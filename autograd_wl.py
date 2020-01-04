import torch

A, B = None, None

def capture_activations(module, inputs, outputs):
    global A
    A = inputs[0]
def capture_backprops(module, inputs, outputs):
    global B
    B = outputs[0]

def calculate_sample_grad():
    global A, B
    n = A.shape[0]
    B = B * n
    weight_grad = torch.einsum('ni,nj->nij', B, A)
    bias_grad = B
    grad_norm = torch.sqrt(weight_grad.norm(p=2, dim=(1,2)).pow(2) + bias_grad.norm(p=2, dim=1)).squeeze().detach()
    A, B = None, None
    return grad_norm