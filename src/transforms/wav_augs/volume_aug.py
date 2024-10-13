import torchaudio
from torch import Tensor, nn


class VolumeAug(nn.Module):
    def __init__(self, gain: str, *args, **kwargs):
        super().__init__()
        self.vol = torchaudio.transforms.Vol(gain=gain, gain_type="amplitude")

    def __call__(self, data: Tensor):
        return self.vol(data)
