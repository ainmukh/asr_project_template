import torch_audiomentations
from torch import Tensor
import numpy as np

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, p, sr, *args, **kwargs):
        self.p = p
        self.sr = sr
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        augment = np.random.binomial(1, self.p)
        x = data.unsqueeze(1)
        return self._aug(x, self.sr).squeeze(1) if augment else data
