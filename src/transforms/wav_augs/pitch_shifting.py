import librosa
import torch
from torch import Tensor, nn


class PitchShifting(nn.Module):
    def __init__(self, sr: float, n_steps: float, *args, **kwargs):
        super().__init__()
        self.sr = sr
        self.n_steps = n_steps

    def __call__(self, data: Tensor):
        wav_with_aug = librosa.effects.pitch_shift(
            data.numpy().squeeze(), sr=self.sr, n_steps=-5
        )
        return torch.from_numpy(wav_with_aug)
