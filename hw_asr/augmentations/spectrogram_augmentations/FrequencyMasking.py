import torch_audiomentations
from torch import Tensor
from torchaudio import transforms
import numpy as np

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, p, frequency, *args, **kwargs):
        self.p = p
        self._aug = transforms.FrequencyMasking(frequency)

    def __call__(self, data: Tensor):
        augment = np.random.binomial(1, self.p)
        return self._aug(data) if augment else data
