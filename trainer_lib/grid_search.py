import os.path
from dataclasses import dataclass

from numpy import ndarray
from torch import nn, Tensor
from torch.utils.data import random_split

import utils
from .datasets import TimeSeriesWindowedTensorDataset, TimeSeriesWindowedDatasetConfig
from .permutation_grid import Grid
from .trainer import Trainer, TrainerOptions
from models import Transformer, TransformerParams


@dataclass
class GridSearchOptions:
    root_save_path: str
    valid_split: float
    window_step_size: int
    random_seed: int
    use_start_token: bool


def transformer_grid_search(grid: Grid,
                            data: ndarray | Tensor,
                            trainer_options: TrainerOptions,
                            opts: GridSearchOptions):
    path = os.path.abspath(opts.root_save_path)
    models = []
    names = utils.generate_name(len(grid), opts.random_seed)

    for idx, params in enumerate(grid):
        params = dict(params)  # to help the typechecker not kill itself

        dataset = TimeSeriesWindowedTensorDataset(data,
                                                  TimeSeriesWindowedDatasetConfig(
                                                      params['src_window'],
                                                      params['tgt_window'],
                                                      params['src_seq_length'],
                                                      params['tgt_seq_length'],
                                                      opts.window_step_size,
                                                      opts.use_start_token)
                                                  )

        valid_size = int(round(len(dataset) * opts.valid_split))
        train, valid = random_split(dataset, [len(dataset) - valid_size, valid_size])

        transformer_params = TransformerParams(
            src_size=params['src_size'] * params['src_window'],
            tgt_size=params['tgt_size'] * params['tgt_window'],
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            d_ff=params['d_ff'],
            max_seq_length=max(params['src_seq_length'], params['tgt_seq_length']),
            dropout=params['dropout']
        )
        model = Transformer(transformer_params)

        trainer_options.save_path = os.path.join(path, names[idx])
        trainer = Trainer(model, trainer_options)

        trainer.train(train, valid)
        models.append({'name': names[idx],
                       'model': trainer.model,
                       'params': params,
                       'metrics': trainer.metrics})

    return models
