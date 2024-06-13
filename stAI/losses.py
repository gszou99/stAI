import torch
from torch import nn


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = nn.Parameter(torch.Tensor((mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))), requires_grad=False)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / 
                         (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)



class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class PcorrLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        x_mean = torch.mean(x, dim=0)
        y_mean = torch.mean(y, dim=0)
        x_std = torch.std(x, dim=0)
        y_std = torch.std(y, dim=0)
        r = torch.mean((x - x_mean) * (y - y_mean), dim=0) / (x_std * y_std+1e-6)
        r = torch.mean(r)
        loss = 1 - r
        return loss


class CosineLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        x_l2_sqr = (x**2).sum(dim=0)
        y_l2_sqr = (y**2).sum(dim=0)
        r = torch.sum(x*y, dim=0) / (x_l2_sqr * y_l2_sqr + 1e-6)**.5
        r = torch.mean(r)
        loss = 1 - r
        return loss

