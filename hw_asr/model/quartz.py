from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class QuartzModel(BaseModel):
    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__()

    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def transform_input_lengths(self, input_lengths):
        """
        Input length transformation function.
        For example: if your NN transforms spectrogram of time-length `N` into an
            output with time-length `N / 2`, then this function should return `input_lengths // 2`
        """
        raise NotImplementedError
