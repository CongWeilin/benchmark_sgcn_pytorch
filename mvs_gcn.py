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
    print('Average training time is %0.3f'%np.mean(times))
    print('Average full batch time is %0.3f'%np.mean(full_batch_times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all

"""
minimal variance sampling with online learning (on the fly)
"""
def mvs_gcn_otf(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
    from optimizers import calculate_grad_variance
    
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
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
    
    # before started, every nodes has the save probability
    sample_ratio = args.batch_size/len(train_nodes)
    train_nodes_p = np.ones_like(train_nodes, dtype=np.float32) * sample_ratio
    
    for epoch in np.arange(args.epoch_num):
        
        #######################
        #######################
        #######################
        susage.train()
        cur_train_loss_all = []
        grad_variance = []
        
        t2 = time.time()
        for iter_num in range(args.batch_num):
            sample_mask = np.random.uniform(0, 1, len(train_nodes))<= train_nodes_p
            probs_nodes = train_nodes_p[sample_mask] * len(train_nodes)
            batch_nodes = train_nodes[sample_mask]
            adjs, input_nodes, output_nodes, probs_nodes, sampled_nodes = \
                exact_sampler_.mini_batch(iter_num, batch_nodes, probs_nodes, samp_num_list, len(feat_data), adj_matrix, args.n_layers)
            adjs = package_mxl(adjs, device)
            
            optimizer.zero_grad()
            weight = 1.0/torch.FloatTensor(probs_nodes).to(device)
            current_loss, current_grad_norm = susage.partial_grad_with_norm(
                feat_data[input_nodes], adjs, labels[output_nodes], weight)
            
            # only for experiment purpose to demonstrate ...
            if bool(args.show_grad_norm):
                grad_variance.append(calculate_grad_variance(
                    susage, feat_data, labels, train_nodes, adjs_full))

            optimizer.step()
            
            # print statistics
            cur_train_loss_all += [current_loss.cpu().detach()]
            
            # update train_nodes_p
            thresh = CalculateThreshold(current_grad_norm, args.batch_size*sample_ratio)
            current_node_p= current_grad_norm/thresh
            current_node_p[current_node_p > 1] = 1
            train_nodes_p[sample_mask] = current_node_p
            
        t3 = time.time()
        # calculate training loss
        cur_train_loss = np.mean(cur_train_loss_all)
        #######################
        #######################
        #######################

        times += [t3-t2]
        print('mvs_gcn run time per epoch is %0.3f' % (t3-t2))

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
    print('Average training time is %0.3f'%np.mean(times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


"""
Minimal Variance Sampling GCN +
"""
def mvs_gcn_plus(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    wrapper = ForwardWrapper(n_nodes=len(feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes, concat=concat)
    
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
    print('Average training time is %0.3f'%np.mean(times))
    print('Average full batch time is %0.3f'%np.mean(full_batch_times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all

"""
minimal variance sampling with online learning and subgraph sampling (on the fly)
"""
def mvs_gcn_plus_otf(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
    from optimizers import calculate_grad_variance
    wrapper = ForwardWrapper(n_nodes=len(feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes, concat=concat)
    
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
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
    cur_test_loss = susage.calculate_loss_grad(
        feat_data, adjs_full, labels, valid_nodes)
        
    best_val, cnt = 0, 0

    loss_train = [cur_test_loss]
    loss_test = [cur_test_loss]
    grad_variance_all = []
    loss_train_all = [cur_test_loss]
    times = []
    
    # before started, every nodes has the save probability
    sample_ratio = args.batch_size/len(train_nodes)
    train_nodes_p = np.ones_like(train_nodes, dtype=np.float32) * sample_ratio
    
    # warm start
    wrapper.forward_full(susage, feat_data, adjs_full, sampled_nodes_full)
    
    for epoch in np.arange(args.epoch_num):
        #######################
        #######################
        #######################
        susage.train()
        cur_train_loss_all = []
        grad_variance = []
        
        t2 = time.time()
        for iter_num in range(args.batch_num):
            sample_mask = np.random.uniform(0, 1, len(train_nodes))<= train_nodes_p
            probs_nodes = train_nodes_p[sample_mask] * len(train_nodes)
            batch_nodes = train_nodes[sample_mask]
            adjs, adjs_exact, input_nodes, output_nodes, probs_nodes, sampled_nodes, input_exact_nodes = \
                subgraph_sampler_.mini_batch(iter_num, batch_nodes, probs_nodes, samp_num_list, len(feat_data), adj_matrix, args.n_layers)
            adjs, adjs_exact = package_mxl(adjs, device), package_mxl(adjs_exact, device)
            
            optimizer.zero_grad()
            weight = 1.0/torch.FloatTensor(probs_nodes).to(device)
            current_loss, current_grad_norm = wrapper.partial_grad_with_norm(susage, 
                feat_data[input_nodes], adjs, sampled_nodes, feat_data, adjs_exact, input_exact_nodes, labels[output_nodes], weight)
            
            # only for experiment purpose to demonstrate ...
            if bool(args.show_grad_norm):
                grad_variance.append(calculate_grad_variance(
                    susage, feat_data, labels, train_nodes, adjs_full))

            optimizer.step()
            
            # print statistics
            cur_train_loss_all += [current_loss.cpu().detach()]
            
            # update train_nodes_p
            thresh = CalculateThreshold(current_grad_norm, args.batch_size*sample_ratio)
            current_node_p= current_grad_norm/thresh
            current_node_p[current_node_p > 1] = 1
            train_nodes_p[sample_mask] = current_node_p
            
        t3 = time.time()
        # calculate training loss
        cur_train_loss = np.mean(cur_train_loss_all)
        #######################
        #######################
        #######################

        times += [t3-t2]
        print('mvs_gcn run time per epoch is %0.3f' % (t3-t2))

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
    print('Average training time is %0.3f'%np.mean(times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all