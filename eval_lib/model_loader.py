import os
import json
from typing import Callable

from torch import nn
from models import Transformer, TransformerParams, VPTransformer, VPTransformerParams, LSTMParams, LSTMModel
from trainer_lib.utils import resume


def load_model(path: str, epoch: int, load_fn: Callable):
    with open(os.path.join(path, 'params.json'), 'r') as fp:
        params = json.load(fp)

    model = load_fn(params)
    resume(model, os.path.join(path, f'{epoch}.pth'))

    with open(os.path.join(path, f'{epoch}.json'), 'r') as fp:
        metrics = json.load(fp)

    return params, model, metrics


def load_lstm(params):
    lstm_params = LSTMParams(
        in_features=params['in_features'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        out_features=params['out_features'],
        dropout=params['dropout'],
        in_noise=params['in_noise'],
        hid_noise=params['hid_noise'],
        bidirectional=params['bidirectional']
    )
    return LSTMModel(lstm_params)


def load_transformer(params):
    transformer_params = TransformerParams(
        src_size=params['src_size'],
        tgt_size=params['tgt_size'],
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        d_ff=params['d_ff'],
        max_seq_length=params['max_seq_length'],
        dropout=params['dropout']
    )

    return Transformer(transformer_params)