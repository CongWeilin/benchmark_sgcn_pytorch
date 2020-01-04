from utils import *
from layers import GraphConvolution, GraphSageConvolution
import autograd_wl 

"""
This is a plain implementation of GCN
Used for FastGCN, LADIES
"""
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class):
        super(GCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        for _ in range(layers-2):
            self.gcs.append(GraphConvolution(nhid,  nhid))
        self.gc_out = GraphConvolution(nhid, num_classes)
        self.gc_out.linear.register_forward_hook(autograd_wl.capture_activations)
        self.gc_out.linear.register_backward_hook(autograd_wl.capture_backprops)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        
        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, adjs):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adjs[ell])
            x = self.relu(x)
            x = self.dropout(x)
        x = self.gc_out(x, adjs[self.layers-1])
        return x

    def partial_grad(self, x, adjs, targets, weight=None):
        outputs = self.forward(x, adjs)
        if weight is None:
            loss = self.loss_f(outputs, targets)
        else:
            loss = self.loss_f_vec(outputs, targets) * weight
            loss = loss.sum()
        loss.backward()
        return loss.detach()

    def calculate_sample_grad(self, x, adjs, targets, batch_nodes):
        # use smart way
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs, targets[batch_nodes])
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad()

        return grad_per_sample.cpu().numpy()
    
    def calculate_loss_grad(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs[batch_nodes], targets[batch_nodes])
        loss.backward()
        return loss.detach()

    def calculate_f1(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        if self.multi_class:
            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs = outputs.argmax(dim=1)
        return f1_score(outputs[batch_nodes].cpu().detach(), targets[batch_nodes].cpu().detach(), average="micro")
    
    
"""
This is an implementation of GCN with GraphSage convolution layer
Used for GraphSage, VRGCN, ClusterGCN
"""
class GraphSageGCN(nn.Module):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class):
        super(GraphSageGCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphSageConvolution(nfeat, nhid, use_lynorm=True))
        for _ in range(layers-2):
            self.gcs.append(GraphSageConvolution(nhid,  nhid, use_lynorm=True))
        self.gc_out = GraphSageConvolution(nhid,  num_classes, use_lynorm=False) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gc_out.linear.register_forward_hook(autograd_wl.capture_activations)
        self.gc_out.linear.register_backward_hook(autograd_wl.capture_backprops)
        
        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, adjs):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adjs[ell])
            x = self.relu(x)
            x = self.dropout(x)
        x = self.gc_out(x, adjs[self.layers-1])
        return x

    def partial_grad(self, x, adjs, targets, weight=None):
        outputs = self.forward(x, adjs)
        if weight is None:
            loss = self.loss_f(outputs, targets)
        else:
            loss = self.loss_f_vec(outputs, targets) * weight
#             print('>>>', loss.max(), loss.min(), loss.shape, weight.max(), weight.min(), weight.shape)
            loss = loss.sum()
        loss.backward()
        return loss.detach()

    def calculate_sample_grad(self, x, adjs, targets, batch_nodes):
        # use smart way
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs, targets[batch_nodes])
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad()

        return grad_per_sample.cpu().numpy()
    
    def calculate_loss_grad(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs[batch_nodes], targets[batch_nodes])
        loss.backward()
        return loss.detach()

    def calculate_f1(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        if self.multi_class:
            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs = outputs.argmax(dim=1)
        return f1_score(outputs[batch_nodes].cpu().detach(), targets[batch_nodes].cpu().detach(), average="micro")