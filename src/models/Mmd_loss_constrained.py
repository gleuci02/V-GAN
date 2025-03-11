import torch
from torch import nn
import torch_two_sample as tts


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2).to('cuda')
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        
        n_samples = L2_distances.shape[0]
        self.bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return L2_distances.data.sum() / (n_samples ** 2 - n_samples)


    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLossConstrained(nn.Module):
    '''
    Constrained loss by the number of features selected
    '''

    def __init__(self, weight, kernel=RBF()):
        super().__init__()
        self.kernel = kernel
        self.weight = weight

    def forward(self, X, Y, U):
        K = self.kernel(torch.vstack([X, Y]))
        self.bandwidth = self.kernel.bandwidth
        self.bandwidth_multipliers = self.kernel.bandwidth_multipliers
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        return XX - 2 * XY + YY + self.weight*(torch.mean(torch.ones(U.shape[1]).to('cuda') - torch.topk(U, 1, 0).values))
