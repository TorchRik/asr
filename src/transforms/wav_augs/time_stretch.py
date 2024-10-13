import librosa
import torch
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, rate: float, *args, **kwargs):
        super().__init__()
        self.rate = rate

    def __call__(self, data: Tensor):
        augmented_wav = librosa.effects.time_stretch(
            data.numpy().squeeze(), rate=self.rate
        )
        return torch.from_numpy(augmented_wav)
