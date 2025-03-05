import torch

from models import LSTMModel, Transformer
from models.vp_lstm import VPLSTMModel
from .model_loader import load_model
from trainer_lib.datasets import TimeSeriesWindowedTensorDataset
import pandas as pd
from typing import Callable, Tuple
from matplotlib import pyplot as plt
import numpy as np


class Evaluator:
    def __init__(self):
        self.models: dict[str, list[torch.nn.Module]] = {}
        self.model_params: dict[str, list[dict]] = {}
        self.forecasts = {}
        self.metrics = {}

    def add_model(self, path: str, epoch: int, load_fn: Callable, key: str) -> dict:
        params, model, metrics = load_model(path, epoch, load_fn)
        if key not in self.metrics.keys() or key not in self.models.keys() or key not in self.model_params.keys():
            self.metrics[key], self.models[key], self.model_params[key] = [], [], []
        self.model_params[key].append(params)
        self.models[key].append(model)
        self.metrics[key].append(metrics)
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

        if isinstance(model, LSTMModel) or isinstance(model, VPLSTMModel):
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
        d = {'mse': [],
             'rmse': [],
             'mae': []}

        for key in self.list_models():
            mse = np.array([metric['eval']['MSE'][-1] for metric in self.metrics[key]])
            rmse = np.array([metric['eval']['RMSE'][-1] for metric in self.metrics[key]])
            mae = np.array([metric['eval']['MAE'][-1] for metric in self.metrics[key]])

            d['mse'].append(f"{mse.mean():0.4f} +- {mse.std():0.4f}")
            d['rmse'].append(f"{rmse.mean():0.4f} +- {rmse.std():0.4f}")
            d['mae'].append(f"{mae.mean():0.4f} +- {mae.std():0.4f}")

        df = pd.DataFrame(data=d, index=[key for key in self.list_models()])

        return df

    def plot_learning_curves(self, start: int, end: int, eval_steps: int):
        for key in self.list_models():
            plt.title(f'{key} - learning curve')
            for idx, metric in enumerate(self.metrics[key]):
                print(self.model_params[key][idx])
                plt.plot(np.arange(max(start, 5), min(end, len(metric['train']['MSE']))),
                         metric['train']['MSE'][start:end],
                         label=f'{idx} - train')
                plt.plot(np.arange(max(start, 5), min(end, len(metric['train']['MSE'])), eval_steps),
                         metric['eval']['MSE'][start//eval_steps:end//eval_steps],
                         label=f'{idx} - eval')
            plt.legend()
            plt.show()


    @staticmethod
    def _lstm_forecast(model: LSTMModel | VPLSTMModel, dataset: TimeSeriesWindowedTensorDataset, pred_len: int):
        model.eval()
        with torch.no_grad():
            gt = []
            p = []

            for shift_offset in range(0, len(dataset), pred_len * dataset.ws_y):
                src, tgt = dataset[shift_offset]
                inp = src.unsqueeze(0) # type: ignore
                for _ in range(pred_len):
                    if isinstance(model, VPLSTMModel):
                        out = model(inp, True)
                    else:
                        out = model(inp)
                    inp = torch.concatenate((inp, out.unsqueeze(-2)), dim=1)

                p.append(dataset.get_sequence_from_y_windows(inp[:, -pred_len:, :].detach()).to('cpu'))
                gt.append(dataset.get_sequence_from_y_windows(torch.tensor(tgt)).to('cpu'))

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
