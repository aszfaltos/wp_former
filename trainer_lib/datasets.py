from dataclasses import dataclass

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import TensorDataset
from signal_decomposition.preprocessor import Preprocessor, SimplePreprocessor

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class TimeSeriesWindowedDatasetConfig:
    src_window_size: int
    tgt_window_size: int
    src_sequence_length: int
    tgt_sequence_length: int
    step_size: int
    add_start_token: bool
    preprocess_y: bool


class TimeSeriesWindowedTensorDataset(TensorDataset):
    def __init__(self, data: ndarray,
                 config: TimeSeriesWindowedDatasetConfig,
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

        super(TimeSeriesWindowedTensorDataset, self).__init__(torch.tensor(x), torch.tensor(y))

        self.x = self.tensors[0].to(DEVICE)
        self.y = self.tensors[1].to(DEVICE)

    def _create_sequences(self, data: ndarray) -> (ndarray, ndarray):
        x_seqs = []
        y_seqs = []
        for i in range(0, data.shape[0] - (self.ws_x * self.sl_x + self.ws_y * self.sl_y), self.step_size):
            next_x = data[i:i + self.ws_x * self.sl_x]
            next_x = self.preprocessor.process(next_x)
            self.vec_size_x = next_x.shape[-1]
            x_seqs.append(next_x.reshape(self.sl_x, self.vec_size_x * self.ws_x))

            next_y = data[i + self.ws_x * self.sl_x:i + self.ws_x * self.sl_x + self.ws_y * self.sl_y]
            if self.preprocess_y:
                next_y = self.preprocessor.process(next_y)
            else:
                next_y = np.array(next_y[:, np.newaxis, :], dtype=np.float32)
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

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if type(idx) is int:
            return self.x[idx], self.y[idx]
        else:
            return list(zip(self.x[idx], self.y[idx]))
