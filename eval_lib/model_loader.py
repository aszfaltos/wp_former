import os
import json
from torch import nn
import torch
from models import Transformer, TransformerParams, VPTransformer, VPTransformerParams
from trainer_lib.utils import resume


def load_model(test: str, name: str, epoch: int, model_type, path: str = 'trained'):
    path = os.path.join(path, test, name)
    with open(os.path.join(path, 'params.json'), 'r') as fp:
        params = json.load(fp)

    model = model_from_params(params, model_type)
    resume(model, os.path.join(path, f'{epoch}.pth'))
    model.eval()

    with open(os.path.join(path, f'{epoch}.json'), 'r') as fp:
        metrics = json.load(fp)

    return params, model, metrics


def model_from_params(params: dict, model_type) -> nn.Module:
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
