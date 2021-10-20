import torch_audiomentations
from torch import Tensor
import torch
import librosa
import numpy as np

from hw_asr.augmentations.base import AugmentationBase


class Noise(AugmentationBase):
    def __init__(self, p, noise_name, noise_level, *args, **kwargs):
        filename = librosa.ex(noise_name)
        noise, sr = librosa.load(filename)
        self.p = p
        self.noise = torch.from_numpy(noise).unsqueeze(0)
        self.noise_level = torch.Tensor([noise_level])  # [0, 40]
        self.noise_energy = torch.norm(self.noise)
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        augment = np.random.binomial(1, self.p)
        if augment:
            audio_energy = torch.norm(data)
            alpha = (audio_energy / self.noise_energy) * torch.pow(10, -self.noise_level / 20)
            clipped_wav = data[..., :self.noise.size(1)]
            clipped_noise = self.noise[:, :clipped_wav.size(1)]
            res = clipped_wav + alpha * clipped_noise
            return torch.clamp(res, -1, 1)
        else:
            return data
