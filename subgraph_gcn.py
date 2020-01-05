from utils import *
from packages import *
from optimizers import sgd_step, variance_reduced_step
from mvs_gcn import ForwardWrapper
"""
ClusterGCN
"""


def clustergcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
    samp_num_list = np.array([1 for _ in range(args.n_layers)])
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    cluster_num = int(len(train_nodes)/args.batch_size)
    print(cluster_num)
    cluster_sampler_ = cluster_sampler(adj_matrix, train_nodes, cluster_num)

    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                          layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters(), lr=0.01)

    adjs_full, input_nodes_full, sampled_nodes_full = cluster_sampler_.full_batch(
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
        jobs = prepare_data(pool, cluster_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
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
    
def clustergcn_graphsage(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
    samp_num_list = np.array([1 for _ in range(args.n_layers)])
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    cluster_num = int(len(train_nodes)/args.batch_size)
    cluster_sampler_ = cluster_sampler(adj_matrix, train_nodes, 50)

    susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                          layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters(), lr=0.01)

    adjs_full, input_nodes_full, sampled_nodes_full = cluster_sampler_.full_batch(
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
        jobs = prepare_data(pool, cluster_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
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
Minimal Variance Sampling GCN +
"""
def subgraph_gcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
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
        
        train_nodes_p = args.batch_size * \
            np.ones_like(train_nodes)/len(train_nodes)

        susage.zero_grad()
            
        # prepare train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, subgraph_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers, args.is_ratio)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num

        t1=t0=0
        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = variance_reduced_step(susage, optimizer, feat_data, labels,
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

def subgraph_graphsage(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    wrapper = ForwardWrapper(n_nodes=len(feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes)
    
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    subgraph_sampler_ = subgraph_sampler(adj_matrix, train_nodes)
    susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
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
        
        train_nodes_p = args.batch_size * \
            np.ones_like(train_nodes)/len(train_nodes)

        susage.zero_grad()
            
        # prepare train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, subgraph_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers, args.is_ratio)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num

        t1=t0=0
        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = variance_reduced_step(susage, optimizer, feat_data, labels,
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