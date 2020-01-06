from utils import *
import autograd_wl
"""
Wrapper for variance reduction opts
"""

class ForwardWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes):
        super(ForwardWrapper, self).__init__()
        self.n_layers = n_layers
        self.hiddens = torch.zeros(n_layers, n_nodes, 2*n_hid)

    def forward_full(self, net, x, adjs, sampled_nodes):
        for ell in range(len(net.gcs)):
            x = net.gcs[ell](x, adjs[ell])
            x = net.relu(x)
            x = net.dropout(x)
            self.hiddens[ell,sampled_nodes[ell]] = x.cpu().detach()
            
        x = net.gc_out(x)
        return x

    def forward_mini(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes):
        cached_outputs = []
        for ell in range(len(net.gcs)):
            x_bar = x if ell == 0 else self.hiddens[ell-1,sampled_nodes[ell-1]].to(x)
            x_bar_exact = x_exact[input_exact_nodes[ell]] if ell == 0 else self.hiddens[ell-1,input_exact_nodes[ell]].to(x)
            x = net.gcs[ell](x, adjs[ell]) - net.gcs[ell](x_bar, adjs[ell]) + net.gcs[ell](x_bar_exact, adjs_exact[ell])
            x = net.relu(x)
            x = net.dropout(x)
            cached_outputs += [x.detach().cpu()]

        x = net.gc_out(x)
    
        for ell in range(len(net.gcs)):
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
            if net.multi_class:
                loss = net.loss_f_vec(outputs, targets)
                loss = loss.mean(1) * weight
            else:
                loss = net.loss_f_vec(outputs, targets) * weight
            loss = loss.sum()
        loss.backward()
        return loss.detach()