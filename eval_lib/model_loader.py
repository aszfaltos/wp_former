import os
import json
import torch
from models import TransformerParams
from trainer_lib.utils import resume


def load_model(test: str, name: str, epoch: int, model_type: type, path: str = 'trained'):
    path = os.path.join(path, test, name)
    with open(os.path.join(path, 'params.json'), 'r') as fp:
        params = json.load(fp)

    transformer_params = TransformerParams(
        src_size=params['src_size'] * params['src_window'],
        tgt_size=params['tgt_size'] * params['tgt_window'],
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        d_ff=params['d_ff']*params['d_model'],
        max_seq_length=max(params['src_seq_length'], params['tgt_seq_length']),
        dropout=params['dropout']
    )

    model = model_type(transformer_params)
    resume(model, os.path.join(path, f'{epoch}.pth'))
    model.eval()

    with open(os.path.join(path, f'{epoch}.json'), 'r') as fp:
        metrics = json.load(fp)

    return params, model, metrics
