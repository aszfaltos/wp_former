import os.path
from dataclasses import dataclass
import json

from numpy import ndarray
from torch import nn

import utils
from .datasets import TimeSeriesWindowedTensorDataset, TimeSeriesWindowedDatasetConfig
from .permutation_grid import Grid
from .trainer import Trainer, TrainerOptions
from models import Transformer, TransformerParams, VPTransformer, VPTransformerParams, LSTMModel, LSTMParams
import numpy as np
from signal_decomposition.preprocessor import Preprocessor


@dataclass
class GridSearchOptions:
    root_save_path: str
    valid_split: float
    test_split: float
    window_step_size: int
    random_seed: int
    use_start_token: bool
    preprocess_y: bool


def transformer_grid_search(grid: Grid,
                            data: ndarray,
                            trainer_options: TrainerOptions,
                            opts: GridSearchOptions,
                            preprocessor: Preprocessor | None = None):
    path = os.path.abspath(opts.root_save_path)
    names = utils.generate_name(len(grid), opts.random_seed)

    for idx, params in enumerate(grid):
        params = dict(params)  # to help the typechecker not kill itself

        dataset = TimeSeriesWindowedTensorDataset(data, TimeSeriesWindowedDatasetConfig(
                                                          params['src_window'],
                                                          params['tgt_window'],
                                                          params['src_seq_length'],
                                                          params['tgt_seq_length'],
                                                          opts.window_step_size,
                                                          opts.use_start_token,
                                                          preprocess_y=opts.preprocess_y),
                                                  preprocessor=preprocessor
                                                  )

        valid_size = int(round(len(dataset) * opts.valid_split))
        test_size = int(round(len(dataset) * opts.test_split))

        ind = np.random.permutation(len(dataset))

        train_dataset = dataset[ind[:-valid_size - test_size]]
        valid_dataset = dataset[ind[-valid_size - test_size:-test_size]]
        test_dataset = dataset[ind[-test_size:]]

        params['src_size'] = dataset.vec_size_x
        params['tgt_size'] = dataset.vec_size_y

        model = create_model(params, dataset)

        trainer_options.save_path = os.path.join(path, names[idx])
        os.makedirs(trainer_options.save_path, exist_ok=True)
        with open(os.path.join(trainer_options.save_path, 'params.json'), "w") as fp:
            json.dump(params, fp)

        trainer = Trainer(model, trainer_options)
        trainer.train(train_dataset, valid_dataset, test_dataset, lstm=(params['kind'] == 'lstm'))


def create_model(params: dict, dataset: TimeSeriesWindowedTensorDataset) -> nn.Module:
    if params['kind'] == 'vp_transformer':
        transformer_params = VPTransformerParams(
            src_size=dataset.vec_size_x * dataset.ws_x,
            tgt_size=dataset.vec_size_y * dataset.ws_y,
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            d_ff=params['d_ff'] * params['d_model'],
            max_seq_length=max(dataset.sl_x, dataset.sl_y),
            dropout=params['dropout'],
            vp_bases=params['vp_bases'],
            vp_penalty=params['vp_penalty']
        )
        model = VPTransformer(transformer_params)
        return model
    elif params['kind'] == 'transformer':
        transformer_params = TransformerParams(
            src_size=dataset.vec_size_x * dataset.ws_x,
            tgt_size=dataset.vec_size_y * dataset.ws_y,
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            d_ff=params['d_ff'] * params['d_model'],
            max_seq_length=max(dataset.sl_x, dataset.sl_y),
            dropout=params['dropout']
        )
        model = Transformer(transformer_params)
        return model
    elif params['kind'] == 'lstm':
        lstm_params = LSTMParams(features=dataset.vec_size_x,
                                 hidden_size=params['hidden_size'],
                                 num_layers=params['num_layers'],
                                 dropout=params['dropout'],
                                 in_noise=params['in_noise'],
                                 hid_noise=params['hid_noise'],
                                 bidirectional=params['bidirectional'])
        model = LSTMModel(lstm_params)
        return model
