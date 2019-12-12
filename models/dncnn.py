import torch.nn as nn
from torch import cat, tensor
import torch
import math

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return x + out
    
class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            norm_layer = RunningBatchNorm
        self.conv = conv3x3(inplanes, planes, stride)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out
    
class ResCNN(nn.Module):
    def __init__(self, channels, width=64, num_of_layers=8):
        super(ResCNN, self).__init__()
        layers = []
        layers.append(conv3x3(channels, width))
        for _ in range(num_of_layers - 2):
            layers.append(ResBlock(width, width))
        layers.append(conv3x3(width, channels))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
    
class DenseBlock(nn.Module):
    def __init__(self, channels, width=64, num_of_layers=8):
        super(ResCNN, self).__init__()
        layers = []
        layers.append(conv3x3(channels, width))
        for _ in range(num_of_layers - 2):
            layers.append(ResBlock(width, width))
        layers.append(conv3x3(width, channels))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
    
class DenseCNN(nn.Module):
    def __init__(self, channels, width=64, num_of_layers=8):
        super(DenseCNN, self).__init__()
        layers = []
        layers.append(ConvBlock(channels, width))
        
        for l in range(1, num_of_layers - 1):
            layers.append(ConvBlock(width*l + channels, width))
        layers.append(conv3x3(width*(num_of_layers - 1) + channels, channels))
        self.layers = layers
        for i, layer in enumerate(layers):
            self.add_module('name %d' % i, layer)
                            
    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            outputs.append(layer(cat(outputs, dim = 1)))
        return outputs[-1]
    
class RunningBatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.nf = nf
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('sums', torch.zeros(1,nf,1,1))
        self.register_buffer('sqrs', torch.zeros(1,nf,1,1))
        self.register_buffer('batch', tensor(0.))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('step', tensor(0.))
        self.register_buffer('dbias', tensor(0.))

    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        s = x.sum(dims, keepdim=True)
        ss = (x*x).sum(dims, keepdim=True)
        c = self.count.new_tensor(x.numel()/nc)
        mom1 = 1 - (1-self.mom)/math.sqrt(bs-1)
        self.mom1 = self.dbias.new_tensor(mom1)
        self.sums.lerp_(s, self.mom1)
        self.sqrs.lerp_(ss, self.mom1)
        self.count.lerp_(c, self.mom1)
        self.dbias = self.dbias*(1-self.mom1) + self.mom1
        self.batch += bs
        self.step += 1

    def forward(self, x):
        if self.training: self.update_stats(x)
        sums = self.sums
        sqrs = self.sqrs
        c = self.count
        if self.step<100:
            sums = sums / self.dbias
            sqrs = sqrs / self.dbias
            c    = c    / self.dbias
        means = sums/c
        vars = (sqrs/c).sub_(means*means)
        if bool(self.batch < 20): vars.clamp_min_(0.01)
        x = (x-means).div_((vars.add_(self.eps)).sqrt())
        return x.mul_(self.mults).add_(self.adds)

    def extra_repr(self):
        return f'{self.nf}, mom={self.mom}, eps={self.eps}'
    
    
class DnCNNKernel(nn.Module):
    def __init__(self, channels, num_of_layers=17, features = 16, kernel_width=5):
        super(DnCNNKernel, self).__init__()
        kernel_size = 3
        padding = 1
        self.kernel_width = kernel_width
        
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features,
                                out_channels=channels*kernel_width*kernel_width,
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

        self.unfolder = nn.Unfold(kernel_width, padding = kernel_width//2)
        
    def forward(self, x):
        
        unfolded_shape = list(x.shape)
        unfolded_shape.insert(2, self.kernel_width**2)
        
        out = self.dncnn(x).reshape(unfolded_shape)
        
        weights = torch.softmax(out, 2)
        
        return torch.einsum('ncuij,ncuij->ncij',
                            self.unfolder(x).reshape(unfolded_shape),
                            weights)
