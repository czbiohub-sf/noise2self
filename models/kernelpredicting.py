import torch
    
class KernelPredicting(torch.nn.Module):
    def __init__(self, channels, BaseNet, kernel_width=5, **base_net_kwargs):
        super(KernelPredicting, self).__init__()
        
        self.kernel_width = kernel_width
        
        self.net = BaseNet(channels, channels*kernel_width*kernel_width, **base_net_kwargs)

        self.unfolder = torch.nn.Unfold(kernel_width, padding = kernel_width//2)
        
    def forward(self, x):
        
        unfolded_shape = list(x.shape)
        unfolded_shape.insert(2, self.kernel_width**2)
        
        out = self.net(x).reshape(unfolded_shape)
        
        weights = torch.softmax(out, 2)
        
        return torch.einsum('ncuij,ncuij->ncij',
                            self.unfolder(x).reshape(unfolded_shape),
                            weights)
