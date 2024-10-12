import torch
from torchaudio.transforms import MelSpectrogram


class LogMelSpectrogram(MelSpectrogram):
    def __init__(self, *args, **kwargs):
        super(LogMelSpectrogram, self).__init__(*args, **kwargs)

    def forward(self, input):
        return torch.log1p(super(LogMelSpectrogram, self).forward(input))
