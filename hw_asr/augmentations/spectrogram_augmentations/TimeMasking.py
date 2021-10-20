import torch_audiomentations
from torch import Tensor
from torchaudio import transforms
import numpy as np

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, p, time, *args, **kwargs):
        self.p = p
        self._aug = transforms.TimeMasking(time)

    def __call__(self, data: Tensor):
        augment = np.random.binomial(1, self.p)
        return self._aug(data) if augment else data
