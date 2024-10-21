from dataclasses import dataclass

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import TensorDataset
from signal_decomposition.preprocessor import Preprocessor, SimplePreprocessor
from abc import ABC, abstractmethod

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class TimeSeriesDatasetConfig:
    src_window_size: int
    tgt_window_size: int
    src_sequence_length: int
    tgt_sequence_length: int
    step_size: int
    add_start_token: bool
    preprocess_y: bool


class TimeSeriesTensorDataset(TensorDataset, ABC):
    def __init__(self, data: ndarray,
                 config: TimeSeriesDatasetConfig,
                 preprocessor: Preprocessor | None = None):
        if preprocessor is None:
            preprocessor = SimplePreprocessor()
        self.preprocessor = preprocessor
        self.ws_x = config.src_window_size
        self.ws_y = config.tgt_window_size
        self.sl_x = config.src_sequence_length
        self.sl_y = config.tgt_sequence_length
        self.step_size = config.step_size
        self.add_start_token = config.add_start_token
        self.preprocess_y = config.preprocess_y
        self.vec_size_x = None
        self.vec_size_y = None

        x, y = self._create_sequences(data)

        super(TimeSeriesTensorDataset, self).__init__(torch.tensor(x), torch.tensor(y))

        self.x = self.tensors[0].to(DEVICE)
        self.y = self.tensors[1].to(DEVICE)

    @abstractmethod
    def _create_sequences(self, data: ndarray) -> (ndarray, ndarray):
        pass

    @abstractmethod
    def get_sequence_from_y_windows(self, window_sequence: Tensor):
        pass

    @abstractmethod
    def get_sequence_from_x_windows(self, window_sequence: Tensor):
        pass

    def reconstruct_preprocessed_sequence(self, preprocessed_sequence):
        return self.preprocessor.reconstruct(preprocessed_sequence)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if type(idx) is int:
            return self.x[idx], self.y[idx]
        else:
            return list(zip(self.x[idx], self.y[idx]))


class TimeSeriesWindowedTensorDataset(TimeSeriesTensorDataset):
    def _create_sequences(self, data: ndarray) -> (ndarray, ndarray):
        x_seqs = []
        y_seqs = []
        for i in range(0, data.shape[0] - (self.ws_x * self.sl_x + self.ws_y * self.sl_y), self.step_size):
            if self.preprocess_y:
                preprocessed = self.preprocessor.process(data[i:i + self.ws_x * self.sl_x + self.ws_y * self.sl_y])
                next_x = preprocessed[:self.ws_x * self.sl_x]
                next_y = preprocessed[self.ws_x * self.sl_x:]
            else:
                next_x = data[i:i + self.ws_x * self.sl_x]
                next_x = self.preprocessor.process(next_x)
                next_y = data[i + self.ws_x * self.sl_x:i + self.ws_x * self.sl_x + self.ws_y * self.sl_y]
                next_y = np.array(next_y[:, np.newaxis, :], dtype=np.float32)

            self.vec_size_x = next_x.shape[-1]
            x_seqs.append(next_x.reshape(self.sl_x, self.vec_size_x * self.ws_x))

            self.vec_size_y = next_y.shape[-1]
            if self.add_start_token:
                y_seqs.append(np.concatenate(
                    [np.ones((1, self.vec_size_y * self.ws_y)),
                     next_y.reshape(self.sl_y, self.vec_size_y * self.ws_y)],
                    dtype=np.float32)
                )
            else:
                y_seqs.append(next_y.reshape(self.sl_y, self.vec_size_y * self.ws_y))

        return np.array(x_seqs), np.array(y_seqs)

    def get_sequence_from_y_windows(self, window_sequence: Tensor):
        return window_sequence.reshape(self.sl_y * self.ws_y, self.vec_size_y)

    def get_sequence_from_x_windows(self, window_sequence: Tensor):
        return window_sequence.reshape(self.sl_x * self.ws_x, self.vec_size_x)


class TimeSeriesInvertedTensorDataset(TimeSeriesTensorDataset):
    def _create_sequences(self, data: ndarray) -> (ndarray, ndarray):
        x_seqs = []
        y_seqs = []
        for i in range(0, data.shape[0] - (self.ws_x * self.sl_x + self.ws_y * self.sl_y), self.step_size):
            if self.preprocess_y:
                preprocessed = self.preprocessor.process(data[i:i + self.ws_x * self.sl_x + self.ws_y * self.sl_y])
                next_x = preprocessed[:self.ws_x * self.sl_x]
                next_y = preprocessed[self.ws_x * self.sl_x:]
            else:
                next_x = data[i:i + self.ws_x * self.sl_x]
                next_x = self.preprocessor.process(next_x)
                next_y = data[i + self.ws_x * self.sl_x:i + self.ws_x * self.sl_x + self.ws_y * self.sl_y]
                next_y = np.array(next_y[:, np.newaxis, :], dtype=np.float32)

            self.vec_size_x = next_x.shape[-1]
            x_seqs.append(next_x.reshape(self.sl_x, self.ws_x, self.vec_size_x)
                                .transpose(0, 2, 1)
                                .reshape(self.sl_x * self.vec_size_x, self.ws_x))

            self.vec_size_y = next_y.shape[-1]
            vec = (next_y.reshape(self.sl_y, self.ws_y, self.vec_size_y)
                         .transpose(0, 2, 1)
                         .reshape(self.sl_y * self.vec_size_y, self.ws_y))

            if self.add_start_token:
                y_seqs.append(np.concatenate([np.ones((self.vec_size_y, self.ws_y)), vec], axis=0))
            else:
                y_seqs.append(vec)

        return np.array(x_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

    def get_sequence_from_y_windows(self, window_sequence: Tensor):
        return (window_sequence.reshape(self.sl_y, self.vec_size_y, self.ws_y)
                               .transpose(1, 2)
                               .reshape(self.sl_y * self.ws_y, self.vec_size_y))

    def get_sequence_from_x_windows(self, window_sequence: Tensor):
        return (window_sequence.reshape(self.sl_x, self.vec_size_x, self.ws_x)
                               .transpose(1, 2)
                               .reshape(self.sl_x * self.ws_x, self.vec_size_x))
