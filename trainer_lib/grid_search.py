import os.path
from dataclasses import dataclass, asdict
import json

from numpy import ndarray
from torch import nn

from utils import Logger
from .datasets import TimeSeriesWindowedTensorDataset, TimeSeriesWindowedDatasetConfig
from .permutation_grid import Grid
from .trainer import LSTMTrainer, TrainerOptions, TransformerTrainer
from models import Transformer, TransformerParams, VPTransformer, VPTransformerParams, LSTMModel, LSTMParams
import numpy as np
from signal_decomposition.preprocessor import Preprocessor
from abc import ABC, abstractmethod


@dataclass
class GridSearchOptions:
    run_nums: int
    root_save_path: str
    valid_split: float
    test_split: float
    random_seed: int


class GridSearch(ABC):
    def __init__(self,
                 dataset: TimeSeriesWindowedTensorDataset,
                 trainer_options: TrainerOptions,
                 search_options: GridSearchOptions,
                 logger: Logger | None = None):
        self.trainer_options = trainer_options
        self.opts = search_options
        with open(os.path.join(self.opts.root_save_path, 'trainer_opts.json'), "w") as fp:
            json.dump(asdict(self.trainer_options), fp)
        with open(os.path.join(self.opts.root_save_path, 'grid_opts.json'), "w") as fp:
            json.dump(asdict(self.opts), fp)

        if logger is None:
            logger = Logger('grid_search')
        self.logger = logger

        valid_size = int(round(len(dataset) * self.opts.valid_split))
        test_size = int(round(len(dataset) * self.opts.test_split))

        np.random.seed(self.opts.random_seed)
        ind = np.random.permutation(len(dataset))

        self.dataset = dataset
        self.train_dataset = dataset[ind[:-valid_size - test_size]]
        self.valid_dataset = dataset[ind[-valid_size - test_size:-test_size]]
        self.test_dataset = dataset[ind[-test_size:]]

    def search(self, grid: Grid):
        for idx, params in enumerate(grid):
            for n in range(self.opts.run_nums):
                model = self.create_model(dict(params))
                self.logger.info(f"Training model {idx + 1}/{len(grid)} with params: {params}")

                self.trainer_options.save_path = os.path.join(os.path.abspath(self.opts.root_save_path),
                                                                              str(idx), str(n))
                os.makedirs(self.trainer_options.save_path, exist_ok=True)
                with open(os.path.join(self.trainer_options.save_path, 'params.json'), "w") as fp:
                    json.dump(params, fp)

                self.train_model(model)

    @abstractmethod
    def train_model(self, model: nn.Module):
        pass

    @abstractmethod
    def create_model(self, params: dict) -> nn.Module:
        pass


class LSTMGridSearch(GridSearch):
    def train_model(self, model: nn.Module):
        trainer = LSTMTrainer(model, self.trainer_options, self.logger)
        trainer.train_loop(self.train_dataset, self.valid_dataset, self.test_dataset)

    def create_model(self, params: dict) -> nn.Module:
        lstm_params = LSTMParams(**params)
        return LSTMModel(lstm_params)


class TransformerGridSearch(GridSearch):
    def train_model(self, model: nn.Module):
        trainer = TransformerTrainer(model, self.trainer_options, self.logger)
        trainer.train_loop(self.train_dataset, self.valid_dataset, self.test_dataset)

    def create_model(self, params: dict) -> nn.Module:
        transformer_params = TransformerParams(**params)
        return Transformer(transformer_params)
