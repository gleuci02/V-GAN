import torch
import torch_two_sample as tts
from torch import nn


class RBF(nn.Module):

    def __init__(self, n_kernels=1, mul_factor=2.0, bandwidth=100):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2).to('cuda')
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        #if self.bandwidth is None:
        n_samples = L2_distances.shape[0]
        return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        self.bandwidth = self.kernel.bandwidth
        self.bandwidth_multipliers = self.kernel.bandwidth_multipliers

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        return XX - 2 * XY + YY