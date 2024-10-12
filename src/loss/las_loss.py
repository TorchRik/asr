import torch
import torch.nn.functional as F
from torch import Tensor, nn


def compute_mask(input_ix, eos_ix=0):
    """compute a boolean mask that equals "1" until first EOS (including that EOS)"""
    return F.pad(
        torch.cumsum(input_ix == eos_ix, dim=-1)[..., :-1] < 1,
        pad=(1, 0, 0, 0),
        value=True,
    )


class LassLoss(nn.Module):
    def forward(self, log_probs, text_encoded, **batch) -> dict[str, Tensor]:
        mask = compute_mask(text_encoded)

        batch_indices = torch.arange(log_probs.shape[0])[:, None]
        seq_indices = torch.arange(log_probs.shape[1])[None, :]

        P = log_probs[batch_indices, seq_indices, text_encoded] * mask

        loss = -P.sum() / mask.sum()

        return {"loss": loss}
