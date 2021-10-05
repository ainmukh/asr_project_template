import logging
from typing import List
# new imports
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]) -> dict:  # here comes the batch
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    for i in dataset_items:
        for key, value in i.items():
            if key == 'text_encoded':
                result_batch['text_encoded_length'] = result_batch.get(
                    'text_encoded_length', []
                ) + [torch.tensor([value.size(-1)])]
            elif key == 'spectrogram':
                result_batch['spectrogram_length'] = result_batch.get(
                    'spectrogram_length', []
                ) + [torch.tensor([value.size(-1)])]
            result_batch[key] = result_batch.get(key, []) + [
                value.squeeze(0).transpose(0, -1) if torch.is_tensor(value) else value
            ]

    for key, value in result_batch.items():
        if torch.is_tensor(value[0]):
            result_batch[key] = nn.utils.rnn.pad_sequence(value, True).transpose(1, -1)
            if result_batch[key].size(-1) == 1:
                result_batch[key] = result_batch[key].squeeze(-1)
    return result_batch
    # raise NotImplementedError
