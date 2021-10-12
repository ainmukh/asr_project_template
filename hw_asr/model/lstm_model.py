from torch import nn

from hw_asr.base import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.encoder = nn.LSTM(n_feats, fc_hidden, batch_first=True, bidirectional=True, num_layers=2)
        self.head = nn.Linear(fc_hidden * 2, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        x, _ = self.encoder(spectrogram.transpose(1, 2))
        x = self.head(x)
        # print('output size=', x.size())
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
