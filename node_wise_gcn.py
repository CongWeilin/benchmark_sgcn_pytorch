from samplers import vrgcn_sampler
from utils import *
from packages import *
from optimizers import sgd_step, variance_reduced_step

"""
GraphSage
"""


def graphsage(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
    samp_num_list = np.array([5 for _ in range(args.n_layers)])
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    graphsage_sampler_ = graphsage_sampler(adj_matrix, train_nodes)
    susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                          layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters())

    adjs_full, input_nodes_full, sampled_nodes_full = graphsage_sampler_.full_batch(
        train_nodes, len(feat_data), args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    best_model = copy.deepcopy(susage)
    susage.zero_grad()
    cur_test_loss = susage.calculate_loss_grad(
        feat_data, adjs_full, labels, valid_nodes)
        
    best_val, cnt = 0, 0

    loss_train = [cur_test_loss]
    loss_test = [cur_test_loss]
    grad_variance_all = []
    loss_train_all = [cur_test_loss]
    times = []

    for epoch in np.arange(args.epoch_num):

        train_nodes_p = args.batch_size * \
            np.ones_like(train_nodes)/len(train_nodes)

        # prepare train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, graphsage_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num

        t0 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = sgd_step(susage, optimizer, feat_data, labels,
                                                                     train_nodes, valid_nodes,
                                                                     adjs_full, train_data, inner_loop_num, device,
                                                                     calculate_grad_vars=bool(args.show_grad_norm))
        t1 = time.time()

        times += [t1-t0]
        print('sgcn run time per epoch is %0.3f' % (t1-t0))
        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate test loss
        susage.eval()

        susage.zero_grad()
        cur_test_loss = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)
        val_f1 = susage.calculate_f1(feat_data, adjs_full, labels, valid_nodes)

        if val_f1 > best_val:
            best_model = copy.deepcopy(susage)
        if val_f1 > best_val + 1e-2:
            best_val = val_f1
            cnt = 0
        else:
            cnt += 1
        if cnt == args.n_stops // args.batch_num:
            break

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)

        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss,
              '| test f1: %.8f' % val_f1)

    f1_score_test = best_model.calculate_f1(
        feat_data, adjs_full, labels, test_nodes)
    print('Average time is %0.3f' % np.mean(times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all

"""
Exact inference
"""
def exact_gcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    exact_sampler_ = exact_sampler(adj_matrix, train_nodes)

    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                           layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters())

    adjs_full, input_nodes_full, sampled_nodes_full = exact_sampler_.full_batch(
            train_nodes, len(feat_data), args.n_layers)
    adjs_full = package_mxl(adjs_full, device)
    
    best_model = copy.deepcopy(susage)
    susage.zero_grad()
    cur_test_loss = susage.calculate_loss_grad(
        feat_data, adjs_full, labels, valid_nodes)
        
    best_val, cnt = 0, 0

    loss_train = [cur_test_loss]
    loss_test = [cur_test_loss]
    grad_variance_all = []
    loss_train_all = [cur_test_loss]
    times = []

    for epoch in np.arange(args.epoch_num):
        # calculate gradients
        susage.zero_grad()

        train_nodes_p = args.batch_size * \
            np.ones_like(train_nodes)/len(train_nodes)

        # prepare train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, exact_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers, args.is_ratio)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num
        t1 = t0 = 0
        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = sgd_step(susage, optimizer, feat_data, labels,
                                          train_nodes, valid_nodes,
                                          adjs_full, train_data, inner_loop_num, device,
                                          calculate_grad_vars=bool(args.show_grad_norm))
        t3 = time.time()
        times += [t1-t0+t3-t2]
        print('mvs_gcn run time per epoch is %0.3f' % (t1-t0+t3-t2))

        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate test loss
        susage.eval()

        susage.zero_grad()
        cur_test_loss = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)
        val_f1 = susage.calculate_f1(feat_data, adjs_full, labels, valid_nodes)

        if val_f1 > best_val:
            best_model = copy.deepcopy(susage)
        if val_f1 > best_val + 1e-2:
            best_val = val_f1
            cnt = 0
        else:
            cnt += 1
        if cnt == args.n_stops // args.batch_num:
            break

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)

        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| val loss: %.8f' % cur_test_loss,
              '| val f1: %.8f' % val_f1)

    f1_score_test = best_model.calculate_f1(
        feat_data, adjs_full, labels, test_nodes)
    print('Average time is %0.3f'%np.mean(times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


"""
Wrapper for Variance Reduced GCN
"""


class ForwardWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes):
        super(ForwardWrapper, self).__init__()
        self.n_layers = n_layers
        self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)

    def forward_mini(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes):
        cached_outputs = []
        for ell in range(len(net.gcs)):
            exact_input = input_exact_nodes[ell]
            out_node_num = adjs_exact[ell].size(0)

            if ell == 0:
                x = x_exact[exact_input]
                support = torch.spmm(adjs_exact[ell], x)
            else:
                x_bar = self.hiddens[ell-1, sampled_nodes[ell-1]].to(x)
                x_bar_exact = self.hiddens[ell-1, exact_input].to(x)
                support = torch.spmm(adjs[ell], x-x_bar) \
                    + torch.spmm(adjs_exact[ell], x_bar_exact)

            x = torch.cat([x[:out_node_num], support], dim=1)
            x = net.gcs[ell].linear(x)
            x = net.gcs[ell].lynorm(x)

            x = net.dropout(net.relu(x))
            cached_outputs += [x.cpu().detach()]

        ell = self.n_layers-1
        exact_input = input_exact_nodes[ell]
        out_node_num = adjs_exact[ell].size(0)

        x_bar = self.hiddens[ell-1, sampled_nodes[ell-1]].to(x)
        x_bar_exact = self.hiddens[ell-1, exact_input].to(x)
        support = torch.spmm(adjs[ell], x-x_bar) \
            + torch.spmm(adjs_exact[ell], x_bar_exact)

        x = torch.cat([x[:out_node_num], support], dim=1)
        x = net.gc_out.linear(x)
        x = net.gc_out.lynorm(x)

        for ell in range(self.n_layers-1):
            self.hiddens[ell, sampled_nodes[ell]] = cached_outputs[ell]
        return x

    def partial_grad(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes, targets):
        outputs = self.forward_mini(
            net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes)
        loss = net.loss_f(outputs, targets)
        loss.backward()
        return loss.detach()


"""
Variance Reduced GCN
"""


def vrgcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes, args, device):
    samp_num_list = np.array([2 for _ in range(args.n_layers)])
    wrapper = ForwardWrapper(n_nodes=len(
        feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes)

    vrgcn_sampler_ = vrgcn_sampler(adj_matrix, train_nodes)

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                          layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters())

    adjs_full, input_nodes_full, sampled_nodes_full = vrgcn_sampler_.full_batch(
        train_nodes, len(feat_data), args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    best_model = copy.deepcopy(susage)
    susage.zero_grad()
    cur_test_loss = susage.calculate_loss_grad(
        feat_data, adjs_full, labels, valid_nodes)
        
    best_val, cnt = 0, 0

    loss_train = [cur_test_loss]
    loss_test = [cur_test_loss]
    grad_variance_all = []
    loss_train_all = [cur_test_loss]
    times = []

    for epoch in np.arange(args.epoch_num):

        train_nodes_p = args.batch_size * \
            np.ones_like(train_nodes)/len(train_nodes)

        # prepare train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, vrgcn_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num

        t0 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = variance_reduced_step(susage, optimizer, feat_data, labels,
                                                                                  train_nodes, valid_nodes,
                                                                                  adjs_full, train_data, inner_loop_num, device, wrapper,
                                                                                  calculate_grad_vars=bool(args.show_grad_norm))
        t1 = time.time()

        times += [t1-t0]
        print('sgcn run time per epoch is %0.3f' % (t1-t0))
        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate test loss
        susage.eval()

        susage.zero_grad()
        cur_test_loss = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)
        val_f1 = susage.calculate_f1(feat_data, adjs_full, labels, valid_nodes)

        if val_f1 > best_val:
            best_model = copy.deepcopy(susage)
        if val_f1 > best_val + 1e-2:
            best_val = val_f1
            cnt = 0
        else:
            cnt += 1
        if cnt == args.n_stops // args.batch_num:
            break

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)

        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss,
              '| test f1: %.8f' % val_f1)

    f1_score_test = best_model.calculate_f1(
        feat_data, adjs_full, labels, test_nodes)
    print('Average time is %0.3f' % np.mean(times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all
