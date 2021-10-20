import torch_audiomentations
import torch
import librosa
import numpy as np
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PitchShifting(AugmentationBase):
    def __init__(self, p, sr, n_steps, *args, **kwargs):
        self.sr = sr
        self.p = p
        self.n_steps = n_steps
        self._aug = librosa.effects.pitch_shift

    def __call__(self, data: Tensor):
        augment = np.random.binomial(1, self.p)
        if augment:
            augumented = self._aug(data.numpy().squeeze(), self.sr, self.n_steps)
            return torch.from_numpy(augumented)
        else:
            return data
