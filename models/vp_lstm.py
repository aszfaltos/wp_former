from .vp_layers import vp_layer, ada_hermite
import torch
from torch import nn
from .lstm import LSTMModel, LSTMParams
from dataclasses import dataclass

@dataclass
class VPLSTMParams:
    in_features: int
    hidden_size: int
    num_layers: int
    out_features: int
    dropout: float
    in_noise: float
    hid_noise: float
    bidirectional: bool
    vp_bases: int
    vp_penalty: int


class VPLSTMModel(nn.Module):
    def __init__(self, params: VPLSTMParams):
        super(VPLSTMModel, self).__init__()
        self.vp_layer = vp_layer(ada_hermite, params.in_features, params.vp_bases, 2, params.vp_penalty)
        lstm_params = LSTMParams(
            in_features=params.in_features,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            out_features=params.out_features,
            dropout=params.dropout,
            in_noise=params.in_noise,
            hid_noise=params.hid_noise,
            bidirectional=params.bidirectional
        )
        self.lstm = LSTMModel(lstm_params)

    def forward(self, x, vp_filter=False):
        if vp_filter:
            x = self.vp_layer(x)

        return self.lstm(x)