import os.path
from dataclasses import dataclass
import json

from numpy import ndarray
from torch import Tensor
from torch.utils.data import random_split

import utils
from .datasets import TimeSeriesWindowedTensorDataset, TimeSeriesWindowedDatasetConfig
from .permutation_grid import Grid
from .trainer import Trainer, TrainerOptions
from models import Transformer, TransformerParams
import numpy as np


@dataclass
class GridSearchOptions:
    root_save_path: str
    valid_split: float
    test_split: float
    window_step_size: int
    random_seed: int
    use_start_token: bool


def transformer_grid_search(grid: Grid,
                            data: ndarray,
                            trainer_options: TrainerOptions,
                            opts: GridSearchOptions):
    path = os.path.abspath(opts.root_save_path)
    names = utils.generate_name(len(grid), opts.random_seed)

    for idx, params in enumerate(grid):
        params = dict(params)  # to help the typechecker not kill itself

        valid_size = int(round(len(data) * opts.valid_split))
        test_size = int(round(len(data) * opts.test_split))

        train, valid, test = data[:-valid_size-test_size], data[-valid_size-test_size:-test_size], data[-test_size:]

        train_dataset = TimeSeriesWindowedTensorDataset(train,
                                                        TimeSeriesWindowedDatasetConfig(
                                                          params['src_window'],
                                                          params['tgt_window'],
                                                          params['src_seq_length'],
                                                          params['tgt_seq_length'],
                                                          opts.window_step_size,
                                                          opts.use_start_token)
                                                        )

        valid_dataset = TimeSeriesWindowedTensorDataset(valid,
                                                        TimeSeriesWindowedDatasetConfig(
                                                            params['src_window'],
                                                            params['tgt_window'],
                                                            params['src_seq_length'],
                                                            params['tgt_seq_length'],
                                                            opts.window_step_size,
                                                            opts.use_start_token)
                                                        )

        test_dataset = TimeSeriesWindowedTensorDataset(test,
                                                       TimeSeriesWindowedDatasetConfig(
                                                           params['src_window'],
                                                           params['tgt_window'],
                                                           params['src_seq_length'],
                                                           params['tgt_seq_length'],
                                                           opts.window_step_size,
                                                           opts.use_start_token)
                                                       )

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
        os.makedirs(trainer_options.save_path, exist_ok=True)
        with open(os.path.join(trainer_options.save_path, 'params.json'), "w") as fp:
            json.dump(params, fp)

        trainer = Trainer(model, trainer_options)
        trainer.train(train_dataset, valid_dataset, test_dataset)
