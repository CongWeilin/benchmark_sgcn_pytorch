from utils import *
from mvs_gcn import mvs_gcn, mvs_gcn_plus
from layer_wise_gcn import ladies, fastgcn
from node_wise_gcn import graphsage, vrgcn, exact_gcn
from subgraph_gcn import clustergcn, graphsaint
import argparse

torch.manual_seed(43)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
Train Configs
"""
parser = argparse.ArgumentParser(
    description='Training GCN on Large-scale Graph Datasets')

parser.add_argument('--dataset', type=str, default='ppi',
                    help='Dataset name: cora/citeseer/pubmed/flickr/reddit/ppi/ppi-large')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default=400,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default=10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default=20,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=256,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GCN layers')
parser.add_argument('--n_stops', type=int, default=1000,
                    help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=256,
                    help='Number of sampled nodes per layer (only for ladies & factgcn)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--is_ratio', type=float, default=1.0,
                    help='Importance sampling rate')
parser.add_argument('--show_grad_norm', type=int, default=0,
                    help='Whether show gradient norm 0-False, 1-True')
parser.add_argument('--cluster_bsize', type=int, default=5,
                    help='how many cluster selected each mini-batch')
args = parser.parse_args()
print(args)


"""
Load Data
"""
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

adj_matrix, labels, feat_data, train_nodes, valid_nodes, test_nodes = preprocess_data(
    args.dataset)
print("Dataset information")
print(adj_matrix.shape, labels.shape, feat_data.shape,
      train_nodes.shape, valid_nodes.shape, test_nodes.shape)

if type(feat_data) == sp.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device)
else:
    feat_data = torch.FloatTensor(feat_data).to(device)
    
args.multi_class = True if args.dataset in ['ppi', 'ppi-large', 'yelp', 'amazon'] else False

if args.multi_class:
    labels = torch.FloatTensor(labels).to(device)
    args.num_classes = labels.shape[1]
else:
    labels = torch.LongTensor(labels).to(device)
    args.num_classes = labels.max().item()+1
    
prefix = '{}_{}_{}_{}_{}'.format(args.dataset, args.n_layers, args.batch_size, args.samp_num, args.is_ratio)

if os.path.exists('results/{}.pkl'.format(prefix)):
    with open('results/{}.pkl'.format(prefix),'rb') as f:
        results = pkl.load(f)
else:
    results = dict()
    
"""
Main
"""
susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = mvs_gcn(
    feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=True)
results['mvs_gcn'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]

susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = mvs_gcn_plus(
    feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=True)
results['mvs_gcn_plus'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]

susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = graphsaint(
    feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=True)
results['graphsaint'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]

susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = clustergcn(
    feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=True)
results['clustergcn'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]

susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = ladies(
    feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device)
results['ladies'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]

susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = graphsage(
    feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device)
results['graphsage'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]

susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = vrgcn(
    feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=True)
results['vrgcn'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]

"""
Save Results
"""
with open('results/{}.pkl'.format(prefix),'wb') as f:
    pkl.dump(results, f)
    
for key, values in results.items():
    loss_train, loss_test, loss_train_all, f1, grad_vars = values
    print(key, f1)