from torch import nn, Tensor
from torch.utils.data import random_split
from numpy import ndarray
from utils import generate_name
from .datasets import TimeSeriesTensorDataset
from .trainer import Trainer
from .permutation_grid import Grid


def grid_search(grid: Grid, model_type: type[nn.Module], x: ndarray | Tensor, y: ndarray | Tensor, batch_size=8,
                epochs=5, split=0.2, rnd_seed=42):
    models = []

    for params in grid:
        dataset = TimeSeriesTensorDataset(x, y, params['enc_seq_length'], params['dec_seq_length'])

        valid_size = int(round(len(dataset) * split))
        train, valid = random_split(dataset, [len(dataset) - valid_size, valid_size])

        model = model_type(**params)
        trainer = Trainer(model, batch_size, epochs)
        trainer.train(train, valid)
        models.append({'name': generate_name(rnd_seed),
                       'model': trainer.model,
                       'params': params,
                       'metrics': trainer.metrics})

    return models
