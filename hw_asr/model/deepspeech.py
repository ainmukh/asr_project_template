from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential


from hw_asr.base import BaseModel


class DeepSpeech(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.fc_hidden = fc_hidden
        self.fc1 = nn.Linear(n_feats, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc3 = nn.Linear(fc_hidden, fc_hidden)
        self.gru = nn.GRU(
            fc_hidden, fc_hidden, num_layers=3, bidirectional=True
        )
        self.fc4 = nn.Linear(fc_hidden, fc_hidden)
        self.out = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        x = self.fc1(spectrogram.transpose(1, 2))
        x = F.hardtanh(F.relu(x), 0, 20)
        x = self.fc2(x)
        x = F.hardtanh(F.relu(x), 0, 20)
        x = self.fc3(x)
        x = F.hardtanh(F.relu(x), 0, 20)
        x = x.squeeze(1).transpose(0, 1)
        x, _ = self.gru(x)
        x = x[:, :, :self.fc_hidden] + x[:, :, self.fc_hidden:]
        x = self.fc4(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
