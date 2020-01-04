#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

# import sys; sys.argv=['']; del sys
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=UserWarning)


# In[2]:


from utils import *
import argparse
import multiprocessing as mp
from samplers import ladies_sampler, exact_sampler, subgraph_sampler, full_batch_sampler, mini_batch_sampler
from samplers import CalculateThreshold
from model import GCN
from optimizers import boost_step, sgd_step, package_mxl


# In[3]:


"""
Dataset arguments
"""
parser = argparse.ArgumentParser(
    description='Training GCN on Large-scale Graph Datasets')

parser.add_argument('--dataset', type=str, default='reddit',
                    help='Dataset name: cora/citeseer/pubmed/reddit')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default=300,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default=10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default=10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GCN layers')
parser.add_argument('--n_stops', type=int, default=200,
                    help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=512,
                    help='Number of sampled nodes per layer (only for ladies & factgcn)')
parser.add_argument('--sample_method', type=str, default='ladies',
                    help='Sampled Algorithms: ladies/fastgcn/graphsage/graphsaint/exact')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--is_ratio', type=float, default=0.2,
                    help='Importance sampling rate')
parser.add_argument('--show_grad_norm', type=int, default=0,
                    help='Whether show gradient norm 0-False, 1-True')
args = parser.parse_args()
print(args)


# In[4]:


if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

def prepare_data(pool, sampler, process_ids, train_nodes, train_nodes_p, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):
    num_train_nodes = len(train_nodes)
    jobs = []
    for _ in process_ids:
        sample_mask = np.random.uniform(0, 1, num_train_nodes)<= train_nodes_p
        # probs_nodes = train_nodes_p[sample_mask] 
        probs_nodes = train_nodes_p[sample_mask] * len(train_nodes) * args.is_ratio
        batch_nodes = train_nodes[sample_mask]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes, probs_nodes,
                                            samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth))
        jobs.append(p)
    return jobs


lap_matrix, labels, feat_data, train_nodes, valid_nodes, test_nodes = preprocess_data(
    args.dataset)
print("Dataset information")
print(lap_matrix.shape, labels.shape, feat_data.shape,
      train_nodes.shape, valid_nodes.shape, test_nodes.shape)

if type(feat_data) == sp.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device)
else:
    feat_data = torch.FloatTensor(feat_data).to(device)
    
labels = torch.LongTensor(labels).to(device)
num_classes = labels.max().item()+1


# In[5]:


if args.sample_method=='ladies':
    sampler = ladies_sampler
elif args.sample_method=='exact':
    sampler = exact_sampler
    
samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])

results = dict()

calculate_gradient_norm = bool(args.show_grad_norm)


# In[6]:


def mvs_gcn(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars=False):
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)

    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    
    susage.to(device)
    print(susage)
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, susage.parameters()))
    
    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
            train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    loss_train = []
    loss_test = []
    grad_variance_all = []
    loss_train_all = []

    best_model = copy.deepcopy(susage)
    best_val, cnt = 0, 0
    for epoch in np.arange(args.epoch_num):
        # calculate gradients
        susage.zero_grad()
        
        mini_batch_nodes = np.random.permutation(len(train_nodes))[:int(len(train_nodes)*args.is_ratio)]        
        grad_per_sample = np.zeros_like(train_nodes, dtype=np.float32)
        adjs_mini, input_nodes_mini, sampled_nodes_mini = mini_batch_sampler(
            train_nodes[mini_batch_nodes], len(feat_data), lap_matrix, args.n_layers)
        adjs_mini = package_mxl(adjs_mini, device)
        
        t0 = time.time()
        grad_per_sample[mini_batch_nodes] = susage.calculate_sample_grad(feat_data[input_nodes_mini], adjs_mini, labels, train_nodes[mini_batch_nodes])
        t1 = time.time()
    
        thresh = CalculateThreshold(grad_per_sample, args.batch_size)
        train_nodes_p = grad_per_sample/thresh
        train_nodes_p[train_nodes_p>1] = 1
            
        # prepare train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, exact_sampler, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num

        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = boost_step(susage, optimizer, feat_data, labels,
                                          train_nodes, valid_nodes,
                                          adjs_full, train_data, inner_loop_num, device, 
                                          calculate_grad_vars=calculate_gradient_norm)
        t3 = time.time()
        print('mvs_gcn run time per epoch is %0.3f'%(t1-t0+t3-t2))

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
        
    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


# In[7]:


print('>>> mvs_gcn')
susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = mvs_gcn(
    feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_gradient_norm)
results['mvs_gcn'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]


# In[8]:


def sgcn(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars=False):
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)

    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    susage.to(device)
    print(susage)
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, susage.parameters()))

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
            train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    loss_train = []
    loss_test = []
    grad_variance_all = []
    loss_train_all = []

    best_model = copy.deepcopy(susage)
    best_val, cnt = 0, 0

    for epoch in np.arange(args.epoch_num):

        train_nodes_p = args.batch_size*np.ones_like(train_nodes)/len(train_nodes)
        
        # prepare train data
        print(train_nodes_p.min(), train_nodes_p.max(), train_nodes_p.sum())
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num
        
        t0 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = sgd_step(susage, optimizer, feat_data, labels,
                                          train_nodes, valid_nodes,
                                          adjs_full, train_data, inner_loop_num, device, 
                                          calculate_grad_vars=calculate_gradient_norm)
        t1 = time.time()
        
        print('sgcn run time per epoch is %0.3f'%(t1-t0))
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
        
    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


# In[9]:


print('>>> sgcn')
susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = sgcn(
    feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_gradient_norm)
results['sgcn'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]


# In[10]:


import autograd_wl

class ForwardWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes):
        super(ForwardWrapper, self).__init__()
        self.n_layers = n_layers
        self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)
        self.model = None

    def forward_full(self, net, x, adjs, sampled_nodes):
        for ell in range(self.n_layers):
            x = net.gcs[ell](x, adjs[ell])
            self.hiddens[ell, sampled_nodes[ell]] = x.cpu().detach()
            x = net.dropout(net.relu(x))
        x = net.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def forward_mini(self, net, x, adjs, sampled_nodes):
        for ell in range(len(net.gcs)):
            # the feature produced by staled_net correspond to the feature inside hiddens
            x = net.gcs[ell](x, adjs[ell]) - self.model.gcs[ell](x, adjs[ell]) + self.hiddens[ell, sampled_nodes[ell]].to(x)
            x = net.dropout(net.relu(x))
        x = net.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def calculate_sample_grad(self, net, x, adjs, sampled_nodes, targets, batch_nodes):
        self.model = copy.deepcopy(net)
        outputs = self.forward_full(net, x, adjs, sampled_nodes)
        loss = F.nll_loss(outputs[batch_nodes], targets[batch_nodes])
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad(batch_nodes)
        
        return grad_per_sample.cpu().numpy()
    
    def calculate_sample_loss(self, net, x, adjs, sampled_nodes, targets, batch_nodes):
        self.model = copy.deepcopy(net)
        outputs = self.forward_full(net, x, adjs, sampled_nodes)

        loss = F.nll_loss(outputs[batch_nodes], targets[batch_nodes], reduction='none')
        loss = loss.detach()
        return grad_per_sample.cpu().numpy()
    
    def partial_grad(self, net, x, adjs, sampled_nodes, targets, weight=None):
        outputs = self.forward_mini(net, x, adjs, sampled_nodes)
        if weight is None:
            loss = F.nll_loss(outputs, targets)
        else:
            loss = F.nll_loss(outputs, targets, reduction='none') * weight
            loss = loss.sum()
        loss.backward()
        return loss.detach()


# In[11]:


def mvs_gcn_plus(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars=False):
    from optimizers import variance_reduced_boost_step 
    wrapper = ForwardWrapper(n_nodes=len(feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=num_classes)
    
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)

    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    
    susage.to(device)
    print(susage)
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, susage.parameters()))

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
            train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    loss_train = []
    loss_test = []
    grad_variance_all = []
    loss_train_all = []

    best_model = copy.deepcopy(susage)
    best_val, cnt = 0, 0

    for epoch in np.arange(args.epoch_num):
            
        # calculate gradients
        susage.zero_grad()
        mini_batch_nodes = np.random.permutation(len(train_nodes))[:int(len(train_nodes)*args.is_ratio)]
        grad_per_sample = np.zeros_like(train_nodes, dtype=np.float32)
        adjs_mini, input_nodes_mini, sampled_nodes_mini = mini_batch_sampler(
            mini_batch_nodes, len(feat_data), lap_matrix, args.n_layers)
        adjs_mini = package_mxl(adjs_mini, device)
        
        t0 = time.time()
        grad_per_sample[mini_batch_nodes] = wrapper.calculate_sample_grad(susage, feat_data[input_nodes_mini], adjs_mini, sampled_nodes_mini, labels, train_nodes[mini_batch_nodes])
        t1 = time.time()
        
        thresh = CalculateThreshold(grad_per_sample, args.batch_size)
        train_nodes_p = grad_per_sample/thresh
        train_nodes_p[train_nodes_p>1] = 1
            
        # prepare train data
        pool = mp.Pool(args.pool_num)
        print(train_nodes_p.min(), train_nodes_p.max(), train_nodes_p.sum(), len(mini_batch_nodes), len(train_nodes))
        jobs = prepare_data(pool, subgraph_sampler, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()

        inner_loop_num = args.batch_num

        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = variance_reduced_boost_step(susage, optimizer, feat_data, labels,
                                          train_nodes, valid_nodes,
                                          adjs_full, train_data, inner_loop_num, device, wrapper,
                                          calculate_grad_vars=calculate_gradient_norm)
        t3 = time.time()
        
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
        
    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


# In[ ]:


print('>>> mvs_gcn_plus')
susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = mvs_gcn(
    feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_gradient_norm)
results['mvs_gcn_plus'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]


# In[ ]:


prefix = '{}_{}_{}_{}_{}_{}'.format(args.sample_method, args.dataset, args.n_layers, args.batch_size, args.samp_num, args.is_ratio)
with open('results/{}.pkl'.format(prefix),'wb') as f:
    pkl.dump(results, f)


# In[ ]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots()
for key, values in results.items():
    loss_train, loss_test, loss_train_all, f1, grad_vars = values
    print(key, f1)
    y = grad_vars[:200]
    x = np.arange(len(y))
    axs.plot(x,y,label=key)

plt.title('{} - {} - grad_vars/epoch'.format(args.sample_method, args.dataset))
axs.set_xlabel('epoch')
axs.set_ylabel('grad_vars')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('{}_grad_vars.pdf'.format(prefix))
plt.close()


# In[ ]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots()
for key, values in results.items():
    loss_train, loss_test, loss_train_all, f1, grad_vars = values
    print(key, f1)
    y = loss_train
    x = np.arange(len(y))
    axs.plot(x,y,label=key)

plt.title('{} - {} - train_loss/epoch'.format(args.sample_method, args.dataset))
axs.set_xlabel('epoch')
axs.set_ylabel('loss_train')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('{}_train_loss.pdf'.format(prefix))
plt.close()


# In[ ]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots()
for key, values in results.items():
    loss_train, loss_test, loss_train_all, f1, grad_vars = values
    print(key, f1)
    y = loss_test
    x = np.arange(len(y))
    axs.plot(x,y,label=key)

plt.title('{} - {} - loss_test/epoch'.format(args.sample_method, args.dataset))
axs.set_xlabel('epoch')
axs.set_ylabel('loss_test')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('{}_loss_test.pdf'.format(prefix))
plt.close()


# In[ ]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots()
for key, values in results.items():
    loss_train, loss_test, loss_train_all, f1, grad_vars = values
    print(key, f1)
    y = loss_train_all[:200]
    x = np.arange(len(y))
    axs.plot(x,y,label=key)

axs.set_xlabel('epoch')
axs.set_ylabel('loss_train_all')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('{}_loss_all.pdf'.format(prefix))
plt.close()


# In[ ]:




