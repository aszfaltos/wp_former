import os.path

from torch import nn, Tensor
from torch.utils.data import random_split
from numpy import ndarray
from .datasets import TimeSeriesWindowedTensorDataset, TimeSeriesWindowedDatasetConfig
from .trainer import Trainer
from .permutation_grid import Grid


def grid_search(grid: Grid, names: list[str], model_type: type[nn.Module], data: ndarray | Tensor, batch_size=8,
                epochs=5, split=0.2, step_size=1, root_path='./'):
    path = os.path.abspath(root_path)
    models = []

    for idx, params in enumerate(grid):
        dataset = TimeSeriesWindowedTensorDataset(data,
                                                  TimeSeriesWindowedDatasetConfig(
                                                        (params['enc_window'], params['dec_window']),
                                                        (params['enc_seq_length'], params['enc_seq_length']),
                                                        step_size))

        valid_size = int(round(len(dataset) * split))
        train, valid = random_split(dataset, [len(dataset) - valid_size, valid_size])

        model = model_type(**params)
        name = names[idx]
        trainer = Trainer(model, os.path.join(path, name), batch_size, epochs)

        trainer.train(train, valid)
        models.append({'name': name,
                       'model': trainer.model,
                       'params': params,
                       'metrics': trainer.metrics})

    return models
