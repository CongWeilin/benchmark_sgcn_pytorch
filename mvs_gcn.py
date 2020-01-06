from utils import *
from packages import *
import autograd_wl
from optimizers import boost_step, variance_reduced_boost_step
from forward_wrapper import ForwardWrapper

"""
Minimal Variance Sampling GCN
"""


def mvs_gcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    exact_sampler_ = exact_sampler(adj_matrix, train_nodes)
    if concat:
        susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                          layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    else:
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
    full_batch_times = []
    data_prepare_times = []
    
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
        full_batch_times += [t1-t0]
        
        thresh = CalculateThreshold(grad_per_sample, args.batch_size)
        train_nodes_p = grad_per_sample/thresh
        train_nodes_p[train_nodes_p > 1] = 1

        # prepare train data
        tp_0 = time.time()
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, exact_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers, args.is_ratio)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        tp_1 = time.time()
        data_prepare_times += [tp_1-tp_0]
        
        inner_loop_num = args.batch_num

        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = boost_step(susage, optimizer, feat_data, labels,
                                          train_nodes, valid_nodes,
                                          adjs_full, train_data, inner_loop_num, device,
                                          calculate_grad_vars=bool(args.show_grad_norm))
        t3 = time.time()
        times += [t3-t2]
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
    print('Average full batch time is %0.3f'%np.mean(full_batch_times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


"""
Minimal Variance Sampling GCN +
"""
def mvs_gcn_plus(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    wrapper = ForwardWrapper(n_nodes=len(feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes)
    
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    subgraph_sampler_ = subgraph_sampler(adj_matrix, train_nodes)

    if concat:
        susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                 layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    else:
        susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                 layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    
    susage.to(device)
    print(susage)
        
    optimizer = optim.Adam(susage.parameters())

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
    full_batch_times = []
    data_prepare_times = []
    
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
        full_batch_times = [t1-t0]
        
        thresh = CalculateThreshold(grad_per_sample, args.batch_size)
        train_nodes_p = grad_per_sample/thresh
        train_nodes_p[train_nodes_p>1] = 1
            
        # prepare train data
        tp0 = time.time()
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, subgraph_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers, args.is_ratio)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        tp1 = time.time()
        data_prepare_times += [tp1-tp0]
        
        inner_loop_num = args.batch_num

        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = variance_reduced_boost_step(susage, optimizer, feat_data, labels,
                                          train_nodes, valid_nodes,
                                          adjs_full, train_data, inner_loop_num, device, wrapper,
                                          calculate_grad_vars=bool(args.show_grad_norm))
        t3 = time.time()
        
        times += [t3-t2]
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
    print('Average full batch time is %0.3f'%np.mean(full_batch_times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all
