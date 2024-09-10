import os
import json
from torch import nn
from models import Transformer, TransformerParams, VPTransformer, VPTransformerParams, LSTMParams, LSTMModel
from trainer_lib.utils import resume



def model_from_params(params: dict, model_type) -> nn.Module:
    params['kind'] = 'lstm'
    print(params['kind'])
    if params['kind'] == 'transformer':
        transformer_params = TransformerParams(
            src_size=params['src_size'] * params['src_window'],
            tgt_size=params['tgt_size'] * params['tgt_window'],
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            d_ff=params['d_ff'] * params['d_model'],
            max_seq_length=max(params['src_seq_length'], params['tgt_seq_length']),
            dropout=params['dropout']
        )

        model = model_type(transformer_params)
        return model
    elif params['kind'] == 'vp_transformer':
        transformer_params = VPTransformerParams(
            vp_bases=params['vp_bases'],
            vp_penalty=params['vp_penalty'],
            src_size=params['src_size'] * params['src_window'],
            tgt_size=params['tgt_size'] * params['tgt_window'],
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            d_ff=params['d_ff'] * params['d_model'],
            max_seq_length=max(params['src_seq_length'], params['tgt_seq_length']),
            dropout=params['dropout']
        )

        model = model_type(transformer_params)
        return model
    elif params['kind'] == 'lstm':
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

        model = model_type(lstm_params)
        return model


def load_model(path: str, epoch: int, load_fn):
    with open(os.path.join(path, 'params.json'), 'r') as fp:
        params = json.load(fp)

    model = load_lstm(params)
    resume(model, os.path.join(path, f'{epoch}.pth'))
    model.eval()

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