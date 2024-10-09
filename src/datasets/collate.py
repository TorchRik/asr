from typing import Any

import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    specs = [
        dataset_item["spectrogram"].permute((0, 2, 1)) for dataset_item in dataset_items
    ]
    encoded_texts = [dataset_item["text_encoded"] for dataset_item in dataset_items]
    max_spec_length = max(spec.shape[1] for spec in specs)
    max_embed_length = max(embeds.shape[1] for embeds in encoded_texts)
    specs = torch.vstack(
        [F.pad(spec, (0, 0, 0, max_spec_length - spec.shape[1])) for spec in specs]
    )

    encoded_texts = torch.vstack(
        [
            F.pad(embeds, (0, max_embed_length - embeds.shape[1]))
            for embeds in encoded_texts
        ]
    )
    return {
        "spectrogram": specs,
        "text_encoded": encoded_texts,
    }
