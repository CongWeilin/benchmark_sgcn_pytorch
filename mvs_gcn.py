from utils import *
from packages import *
import autograd_wl
from optimizers import boost_step, variance_reduced_boost_step

"""
Minimal Variance Sampling GCN
"""


def mvs_graphsage(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    exact_sampler_ = exact_sampler(adj_matrix, train_nodes)
    susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
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

        mini_batch_nodes = np.random.permutation(
            len(train_nodes))[:int(len(train_nodes)*args.is_ratio)]
        grad_per_sample = np.zeros_like(train_nodes, dtype=np.float32)
        adjs_mini, input_nodes_mini, sampled_nodes_mini = exact_sampler_.large_batch(
            train_nodes[mini_batch_nodes], len(feat_data), args.n_layers)
        adjs_mini = package_mxl(adjs_mini, device)

        t0 = time.time()
        grad_per_sample[mini_batch_nodes] = susage.calculate_sample_grad(
            feat_data[input_nodes_mini], adjs_mini, labels, train_nodes[mini_batch_nodes])
        t1 = time.time()

        thresh = CalculateThreshold(grad_per_sample, args.batch_size)
        train_nodes_p = grad_per_sample/thresh
        train_nodes_p[train_nodes_p > 1] = 1

        # prepare train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, exact_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers, args.is_ratio)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num

        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = boost_step(susage, optimizer, feat_data, labels,
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

def mvs_gcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
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

        mini_batch_nodes = np.random.permutation(
            len(train_nodes))[:int(len(train_nodes)*args.is_ratio)]
        grad_per_sample = np.zeros_like(train_nodes, dtype=np.float32)
        adjs_mini, input_nodes_mini, sampled_nodes_mini = exact_sampler_.large_batch(
            train_nodes[mini_batch_nodes], len(feat_data), args.n_layers)
        adjs_mini = package_mxl(adjs_mini, device)

        t0 = time.time()
        grad_per_sample[mini_batch_nodes] = susage.calculate_sample_grad(
            feat_data[input_nodes_mini], adjs_mini, labels, train_nodes[mini_batch_nodes])
        t1 = time.time()

        thresh = CalculateThreshold(grad_per_sample, args.batch_size)
        train_nodes_p = grad_per_sample/thresh
        train_nodes_p[train_nodes_p > 1] = 1

        # prepare train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, exact_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers, args.is_ratio)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num

        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = boost_step(susage, optimizer, feat_data, labels,
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
Wrapper for Minimal Variance Sampling GCN +
"""


class ForwardWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes):
        super(ForwardWrapper, self).__init__()
        self.n_layers = n_layers
        self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)

    def forward_full(self, net, x, adjs, sampled_nodes):
        for ell in range(len(net.gcs)):
            x = net.gcs[ell](x, adjs[ell])
            x = net.relu(x)
            x = net.dropout(x)
            self.hiddens[ell, sampled_nodes[ell]] = x.cpu().detach()
        x = net.gc_out(x, adjs[self.n_layers-1])
        return x

    def forward_mini(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes):
        cached_outputs = []
        for ell in range(len(net.gcs)):
            exact_input = input_exact_nodes[ell]
            
            if ell == 0:
                x = x_exact[exact_input]
                x = torch.spmm(adjs_exact[ell], x)
            else:
                sample_input = sampled_nodes[ell-1]
                x_bar = self.hiddens[ell-1, sample_input].to(x)
                x_bar_exact = self.hiddens[ell-1, exact_input].to(x)
                x = torch.spmm(adjs[ell], x-x_bar) + torch.spmm(adjs_exact[ell], x_bar_exact)

            x = net.gcs[ell].linear(x)
            x = net.relu(x)
            x = net.dropout(x)
            cached_outputs += [x.cpu().detach()]

        ell = self.n_layers-1
        exact_input = input_exact_nodes[ell]

        x_bar = self.hiddens[ell-1, sampled_nodes[ell-1]].to(x)
        x_bar_exact = self.hiddens[ell-1, exact_input].to(x)
        x = torch.spmm(adjs[ell], x-x_bar) + torch.spmm(adjs_exact[ell], x_bar_exact)
        x = net.gc_out.linear(x)

        for ell in range(self.n_layers-1):
            self.hiddens[ell, sampled_nodes[ell]] = cached_outputs[ell]
        return x

    def calculate_sample_grad(self, net, x, adjs, sampled_nodes, targets, batch_nodes):
        outputs = self.forward_full(net, x, adjs, sampled_nodes)
        loss = net.loss_f(outputs, targets[batch_nodes])
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad()
        return grad_per_sample.cpu().numpy()
    
    def partial_grad(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes, targets, weight=None):
        outputs = self.forward_mini(net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes)
        if weight is None:
            loss = net.loss_f(outputs, targets)
        else:
            loss = net.loss_f_vec(outputs, targets) * weight
            loss = loss.sum()
        loss.backward()
        return loss.detach()

"""
Minimal Variance Sampling GCN +
"""
def mvs_gcn_plus(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    wrapper = ForwardWrapper(n_nodes=len(feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes)
    
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    subgraph_sampler_ = subgraph_sampler(adj_matrix, train_nodes)
    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                 layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    
    susage.to(device)
    print(susage)
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, susage.parameters()))

    adjs_full, input_nodes_full, sampled_nodes_full = subgraph_sampler_.full_batch(
            train_nodes, len(feat_data), args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    best_model = copy.deepcopy(susage)
    susage.zero_grad()
    cur_test_loss = susage.calculate_loss_grad(feat_data, adjs_full, labels, valid_nodes)
        
    best_val, cnt = 0, 0

    loss_train = [cur_test_loss]
    loss_test = [cur_test_loss]
    grad_variance_all = []
    loss_train_all = [cur_test_loss]
    times = []

    for epoch in np.arange(args.epoch_num):
        
        susage.zero_grad()
        mini_batch_nodes = np.random.permutation(len(train_nodes))[:int(len(train_nodes)*args.is_ratio)]
    
        grad_per_sample = np.zeros_like(train_nodes, dtype=np.float32)
        adjs_mini, input_nodes_mini, sampled_nodes_mini = subgraph_sampler_.large_batch(
            train_nodes[mini_batch_nodes], len(feat_data), args.n_layers)
        adjs_mini = package_mxl(adjs_mini, device)
        
        t0 = time.time()
        grad_per_sample[mini_batch_nodes] = wrapper.calculate_sample_grad(
            susage, feat_data[input_nodes_mini], adjs_mini, sampled_nodes_mini, labels, train_nodes[mini_batch_nodes])
        t1 = time.time()
        
        thresh = CalculateThreshold(grad_per_sample, args.batch_size)
        train_nodes_p = grad_per_sample/thresh
        train_nodes_p[train_nodes_p>1] = 1
            
        # prepare train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, subgraph_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers, args.is_ratio)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num

        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = variance_reduced_boost_step(susage, optimizer, feat_data, labels,
                                          train_nodes, valid_nodes,
                                          adjs_full, train_data, inner_loop_num, device, wrapper,
                                          calculate_grad_vars=bool(args.show_grad_norm))
        t3 = time.time()
        
        times += [t1-t0 + t3-t2]
        print('mvs_gcn_plus run time per epoch is %0.3f'%(t1-t0 + t3-t2))


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
    print('Average time is %0.3f'%np.mean(times))
    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all
