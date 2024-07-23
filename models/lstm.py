from torch import nn
import torch
from dataclasses import dataclass


@dataclass
class LSTMParams:
    in_features: int
    hidden_size: int
    num_layers: int
    out_features: int
    dropout: float
    in_noise: float
    hid_noise: float
    bidirectional: bool


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            noise = self.noise.expand(*x.size()).float().normal_() * scale
            return x + noise
        return x


class LSTMModel(nn.Module):
    def __init__(self,
                 params: LSTMParams = LSTMParams(1, 15, 2, 1, 0, 0, 0, True),
                 **kwargs):
        super(LSTMModel, self).__init__()
        self.hidden_size = params.hidden_size
        self.h_n_dim = 2 if params.bidirectional else 1
        self.num_layers = params.num_layers
        self.in_noise = GaussianNoise(params.in_noise)
        rec_drop = params.dropout if params.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=params.in_features,
                            hidden_size=self.hidden_size,
                            num_layers=params.num_layers,
                            batch_first=True,
                            bidirectional=params.bidirectional,
                            dropout=rec_drop)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.fc = nn.Sequential(
            nn.Flatten(),
            GaussianNoise(params.hid_noise),
            nn.Dropout(params.dropout),
            nn.Linear(self.hidden_size * self.h_n_dim * self.num_layers, params.out_features)
        )

    def forward(self, x):
        x = self.in_noise(x)
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.h_n_dim * self.num_layers, batch_size, self.hidden_size).to('cuda')
        c_0 = torch.zeros(self.h_n_dim * self.num_layers, batch_size, self.hidden_size).to('cuda')

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n = torch.permute(h_n, (1, 0, 2))
        # From shape [h_n_dim, batch, hidden_size] -> [batch, h_n_dim, hidden_size]
        # flatten and fully connected layer expects batch to be the first dimension
        return self.fc(h_n)
