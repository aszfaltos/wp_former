from dataclasses import dataclass

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import TensorDataset

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


class TimeSeriesWindowedTensorDataset(TensorDataset):
    def __init__(self, data: ndarray, config: TimeSeriesWindowedDatasetConfig):
        self.ws_x = config.src_window_size
        self.ws_y = config.tgt_window_size
        self.sl_x = config.src_sequence_length
        self.sl_y = config.tgt_sequence_length
        self.step_size = config.step_size
        self.add_start_token = config.add_start_token
        self.original_vector_size = data.shape[-1]

        x, y = TimeSeriesWindowedTensorDataset._create_sequences(data,
                                                                 self.ws_x, self.ws_y,
                                                                 self.sl_x, self.sl_y,
                                                                 self.step_size, self.add_start_token)

        super(TimeSeriesWindowedTensorDataset, self).__init__(torch.tensor(x), torch.tensor(y))

        self.x = self.tensors[0].to(DEVICE)
        self.y = self.tensors[1].to(DEVICE)

    @staticmethod
    def _create_sequences(data: ndarray,
                          ws_x: int, ws_y: int,
                          sl_x: int, sl_y: int,
                          step_size: int, add_start_token: bool) -> (ndarray, ndarray):
        x_seqs = []
        y_seqs = []
        vec_size = data.shape[-1]
        for i in range(0, data.shape[0] - (ws_x * sl_x + ws_y * sl_y), step_size):
            x_seqs.append(data[i:i + ws_x * sl_x].reshape(sl_x, vec_size * ws_x))
            if add_start_token:
                y_seqs.append(np.concatenate(
                    [np.ones((1, vec_size * ws_y)),
                     data[i + ws_x * sl_x:i + ws_x * sl_x + ws_y * sl_y].reshape(sl_y, vec_size * ws_y)],
                    dtype=np.float32)
                )
            else:
                y_seqs.append(data[i + ws_x * sl_x:i + ws_x * sl_x + ws_y * sl_y].reshape(sl_y, vec_size * ws_y))

        return np.array(x_seqs), np.array(y_seqs)

    def get_sequence_from_y_windows(self, window_sequence: Tensor):
        return window_sequence.reshape(self.sl_y * self.ws_y, self.original_vector_size)

    def get_sequence_from_x_windows(self, window_sequence: Tensor):
        return window_sequence.reshape(self.sl_x * self.ws_x, self.original_vector_size)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
