import torch
from torch import Tensor, nn


class GaussianNoise(nn.Module):
    def __init__(self, mean: float, std: float, *args, **kwargs):
        super().__init__()
        self.noiser = torch.distributions.Normal(mean, std)

    def __call__(self, data: Tensor):
        return data + self.noiser.sample(data.size())
