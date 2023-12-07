from torch.utils.data import TensorDataset
import torch
from dataclasses import dataclass
from numpy import ndarray
import numpy as np
from torch import Tensor

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class TimeSeriesWindowedDatasetConfig:
    window_size: (int, int)
    sequence_length: (int, int)
    step_size: int


class TimeSeriesWindowedTensorDataset(TensorDataset):
    def __init__(self, data: ndarray, config: TimeSeriesWindowedDatasetConfig):
        self.ws_x = config.window_size[0]
        self.ws_y = config.window_size[1]
        self.sl_x = config.sequence_length[0]
        self.sl_y = config.sequence_length[1]
        self.step_size = config.step_size
        self.original_vector_size = data.shape[-1]

        x, y = TimeSeriesWindowedTensorDataset._create_sequences(data,
                                                                 self.ws_x, self.ws_y,
                                                                 self.sl_x, self.sl_y,
                                                                 self.step_size)

        super(TimeSeriesWindowedTensorDataset, self).__init__(torch.tensor(x), torch.tensor(y))

        self.x = self.tensors[0].to(DEVICE)
        self.y = self.tensors[1].to(DEVICE)

    @staticmethod
    def _create_sequences(data: ndarray,
                          ws_x: int, ws_y: int,
                          sl_x: int, sl_y: int,
                          step_size: int) -> (ndarray, ndarray):
        x_seqs = []
        y_seqs = []
        vec_size = data.shape[-1]
        for i in range(0, data.shape[0] - (ws_x * sl_x + ws_y * sl_y), step_size):
            x_seqs.append(data[i:i+ws_x*sl_x].reshape(sl_x, vec_size * ws_x))
            y_seqs.append(data[i+ws_x*sl_x:i+ws_x*sl_x+ws_y*sl_y].reshape(sl_y, vec_size * ws_y))
        return np.array(x_seqs), np.array(y_seqs)

    def get_sequence_from_y_windows(self, window_sequence: Tensor):
        return window_sequence.reshape(self.sl_y * self.ws_y, self.original_vector_size)

    def get_sequence_from_x_windows(self, window_sequence: Tensor):
        return window_sequence.reshape(self.sl_x * self.ws_x, self.original_vector_size)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
