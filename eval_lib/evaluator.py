import torch

from models import LSTMModel, Transformer
from .model_loader import load_model
from trainer_lib.datasets import TimeSeriesWindowedTensorDataset
import pandas as pd
from typing import Callable, Tuple
import numpy as np


class Evaluator:
    def __init__(self):
        self.models: dict[str, torch.nn.Module] = {}
        self.forecasts = {}
        self.metrics = {}

    def add_model(self, path: str, epoch: int, load_fn: Callable, key: str) -> dict:
        params, model, metrics = load_model(path, epoch, load_fn)
        if key in self.metrics.keys() or key in self.models.keys():
            self.metrics[key], self.models[key] = [], []
        self.metrics[key].append(metrics)
        self.models[key].append(model)
        return params

    def rename_model(self, key: str, new_key: str):
        self.models[new_key] = self.models[key]
        del self.models[key]
        self.metrics[new_key] = self.metrics[key]
        del self.metrics[key]
        if key in self.forecasts.keys():
            self.forecasts[new_key] = self.forecasts[key]
            del self.forecasts[key]

    def remove_model(self, key: str):
        del self.models[key]
        del self.metrics[key]
        self.forecasts.pop(key, None)

    def list_models(self) -> list[str]:
        return list(self.models.keys())

    def generate_rolling_forecast(self, key: str, dataset: TimeSeriesWindowedTensorDataset, pred_len: int) -> Tuple[list, list]:
        model = self.models[key][0]

        gt, f = [], []

        if isinstance(model, LSTMModel):
            gt, f = self._lstm_forecast(model, dataset, pred_len)
        elif isinstance(model, Transformer):
            gt, f = self._transformer_forecast(model, dataset, pred_len)

        return gt, f


    def generate_rolling_forecasts(self, dataset: TimeSeriesWindowedTensorDataset, pred_len):
        forecasts = []

        for key in self.list_models():
            forecasts.append(self.generate_rolling_forecast(key, dataset, pred_len))

        return forecasts

    def generate_evaluation_table(self):
        def bold_min(col):
            bold = 'font-weight: bold'
            default = ''

            min_in_col = col.min()
            return [bold if v == min_in_col else default for v in col]

        d = {'mse': {'mean': [], 'std': []},
             'rmse': {'mean': [], 'std': []},
             'mae': {'mean': [], 'std': []}}

        for key in self.list_models():
            mse = np.array([metric['test']['MSE'][-1] for metric in self.metrics[key]])
            rmse = np.array([metric['test']['RMSE'][-1] for metric in self.metrics[key]])
            mae = np.array([metric['test']['MAE'][-1] for metric in self.metrics[key]])

            d['mse']['mean'].append(mse.mean())
            d['mse']['std'].append(mse.std())
            d['rmse']['mean'].append(rmse.mean())
            d['rmse']['std'].append(rmse.std())
            d['mae']['mean'].append(mae.mean())
            d['mae']['std'].append(mae.std())

        df = pd.DataFrame(data=d, index=[key for key in self.list_models()])
        df.style.apply(bold_min, axis=0)

    @staticmethod
    def _lstm_forecast(model: LSTMModel, dataset: TimeSeriesWindowedTensorDataset, pred_len: int):
        model.eval()
        with torch.no_grad():
            gt = []
            p = []
            for shift_offset in range(0, len(dataset), pred_len * dataset.ws_y):
                src, tgt = dataset[shift_offset]
                src.unsqueeze(0)
                for i in range(pred_len):
                    out = model(src)
                    in_data = torch.concatenate((dataset[shift_offset][0].unsqueeze(0), out.unsqueeze(-2)), dim=1)

                p.append(dataset.get_sequence_from_y_windows(in_data[:, -pred_len:, :].detach()).to('cpu'))
                gt.append(dataset.get_sequence_from_y_windows(tgt).to('cpu'))

        return gt, p


    @staticmethod
    def _transformer_forecast(model: Transformer, dataset: TimeSeriesWindowedTensorDataset, pred_len: int):
        model.eval()
        start_token = torch.ones(1, 1, dataset[0][1].shape[-1])

        with torch.no_grad():
            out = start_token
            gt = []
            p = []
            for shift_offset in range(0, len(dataset), pred_len * dataset.ws_y):
                src, tgt = dataset[shift_offset]
                src = src.unsqueeze(0)
                for i in range(pred_len):
                    out = torch.concatenate((start_token, model(src, out)), dim=1)

                p.append(dataset.get_sequence_from_y_windows(out[:, 1:, :].detach()).to('cpu'))
                gt.append(dataset.get_sequence_from_y_windows(tgt).to('cpu'))

        return gt, p