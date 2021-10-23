import torch
import librosa
import numpy as np
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretching(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        self.p = p
        self._aug = librosa.effects.time_stretch

    def __call__(self, data: Tensor):
        augment = np.random.binomial(1, self.p)
        if augment:
            augumented = self._aug(data.numpy().squeeze(), 2.0)
            return torch.from_numpy(augumented)
        else:
            augment = np.random.binomial(1, self.p)
            if augment:
                augumented = self._aug(data.numpy().squeeze(), 0.5)
                return torch.from_numpy(augumented)
            else:
                return data
