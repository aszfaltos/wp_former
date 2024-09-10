import math
import os.path
import json
from dataclasses import dataclass
import datetime
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .early_stop import EarlyStopper
from .utils import checkpoint
from utils import Logger

from abc import ABC, abstractmethod

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class TrainerOptions:
    batch_size: int
    epochs: int

    learning_rate: float
    learning_rate_decay: float
    weight_decay: float
    gradient_accumulation_steps: int

    early_stopping_patience: int
    early_stopping_min_delta: float

    save_every_n_epochs: int
    save_path: str


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module, opts: TrainerOptions, logger: Logger):
        self.model = model.to(DEVICE)
        self.opts = opts
        self.metrics = {'train': {'MSE': []},
                        'eval': {'MSE': [], 'RMSE': [], 'MAE': []},
                        'test': {'MSE': [], 'RMSE': [], 'MAE': []}}
        self.logger = logger

        self.early_stopper = EarlyStopper(self.opts.early_stopping_patience, self.opts.early_stopping_min_delta)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.opts.learning_rate, betas=(0.9, 0.98), eps=1e-9,
                                    weight_decay=self.opts.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.opts.learning_rate_decay)

        os.makedirs(self.opts.save_path, exist_ok=True)

    def train_loop(self, train_data, valid_data, test_data):
        train_loader = DataLoader(train_data, self.opts.batch_size)
        valid_loader = DataLoader(valid_data, self.opts.batch_size)
        test_loader = DataLoader(test_data, self.opts.batch_size)
        self.logger.info(
            f'Train size: {len(train_data)}, Validation size: {len(valid_data)}, Test size: {len(valid_data)}')

        last_epoch = 0
        time_sum = 0
        for epoch in range(self.opts.epochs):
            start_time = time.monotonic()

            last_epoch = epoch

            self.train(train_loader)

            if (epoch + 1) % self.opts.save_every_n_epochs == 0:
                self.logger.info(f'Epoch: {epoch + 1}, Learning rate: {round(self.scheduler.get_last_lr()[0], 6)}',
                                 extra={'same_line': True, 'delete_prev': True})
                self.evaluate(valid_loader)
                self._save_checkpoint(epoch)
                avg_epoch = time_sum / (epoch + 1)
                etf = avg_epoch * (self.opts.epochs - epoch - 1)
                self.logger.info(f', Avg Epoch: {avg_epoch:.2f}s' +
                                 f', ETF: {self._format_etf(etf)}',
                                 extra={'same_line': True, 'minimal': True})
            else:
                self.logger.debug(f'Epoch: {epoch + 1}, Learning rate: {round(self.scheduler.get_last_lr()[0], 6)}')

            if self.early_stopper.stop:
                self.logger.info(f'Stopped after {epoch + 1} epochs.')
                break

            self.scheduler.step()

            end_time = time.monotonic()

            time_sum += (end_time - start_time)

        self.test(test_loader)
        self._save_checkpoint(last_epoch)

    @abstractmethod
    def train(self, train_loader):
        pass

    def evaluate(self, valid_loader):
        mse_loss, rmse_loss, mae_loss = self._eval(valid_loader)

        self.metrics['eval']['MSE'].append(mse_loss)
        self.metrics['eval']['RMSE'].append(rmse_loss)
        self.metrics['eval']['MAE'].append(mae_loss)
        self.logger.info(f': Eval - MSE: {round(mse_loss, 6)},' +
                         f' RMSE: {round(rmse_loss, 6)},' +
                         f' MAE: {round(mae_loss, 6)}',
                         extra={'same_line': True, 'minimal': True})

        self.early_stopper.step(mse_loss)

    def test(self, test_loader):
        mse_loss, rmse_loss, mae_loss = self._eval(test_loader)

        self.metrics['test']['MSE'].append(mse_loss)
        self.metrics['test']['RMSE'].append(rmse_loss)
        self.metrics['test']['MAE'].append(mae_loss)
        self.logger.info(f'Test - MSE: {round(mse_loss, 6)},' +
                         f' RMSE: {round(rmse_loss, 6)},' +
                         f' MAE: {round(mae_loss, 6)},')

    @abstractmethod
    def _eval(self, data_loader):
        pass

    def _save_checkpoint(self, epoch):
        checkpoint(self.model, os.path.join(self.opts.save_path, f'{epoch + 1}.pth'))
        with open(os.path.join(self.opts.save_path, f'{epoch + 1}.json'), "w") as fp:
            json.dump(self.metrics, fp)

    @staticmethod
    def _format_etf(etf):
        etf_h = int(etf // 3600)
        etf_m = int(etf % 3600 // 60)
        etf_s = int(etf % 60)
        return f'{etf_h}h {etf_m}m {etf_s}s'


class LSTMTrainer(Trainer):
    def train(self, train_loader):
        self.model.train()

        mse = nn.MSELoss()
        train_loss = 0.0

        for (batch_idx, (src_data, tgt_data)) in enumerate(train_loader):
            output = self._inference_step(src_data, tgt_data.shape[1])

            loss = mse(output, tgt_data)
            train_loss += float(loss.item()) / len(train_loader)
            loss = loss / self.opts.gradient_accumulation_steps
            loss.backward()

            if ((batch_idx + 1) % self.opts.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.metrics['train']['MSE'].append(train_loss)
        self.logger.debug(f'Train - MSE: {train_loss}.')

    def _eval(self, data_loader):
        self.model.eval()

        mse = nn.MSELoss()
        mae = nn.L1Loss()
        mse_loss = 0.0
        mae_loss = 0.0

        with torch.no_grad():
            for src_data, tgt_data in data_loader:
                output = self._inference_step(src_data, tgt_data.shape[1])

                loss = mse(output, tgt_data)
                mse_loss += float(loss.item()) / len(data_loader)
                mae_loss += float(mae(output, tgt_data).item()) / len(data_loader)

        return mse_loss, math.sqrt(mse_loss), mae_loss

    def _inference_step(self, src_data, gen_len):
        inp = src_data
        for _ in range(gen_len):
            out = self.model(inp)
            inp = torch.cat((inp, out.unsqueeze(-2)), dim=1)
        return inp[:, -gen_len:]


class TransformerTrainer(Trainer):
    def train(self, train_loader):
        self.model.train()

        mse = nn.MSELoss()
        train_loss = 0.0

        for (batch_idx, (src_data, tgt_data)) in enumerate(train_loader):
            output = self.model(src_data, tgt_data[:, :-1])

            loss = mse(output, tgt_data[:, 1:])
            train_loss += float(loss.item()) / len(train_loader)
            loss = loss / self.opts.gradient_accumulation_steps
            loss.backward()

            if ((batch_idx + 1) % self.opts.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.metrics['train']['MSE'].append(train_loss)
        self.logger.debug(f'Train - MSE: {train_loss}.')

    def _eval(self, data_loader):
        self.model.eval()

        mse = nn.MSELoss()
        mae = nn.L1Loss()
        mse_loss = 0.0
        mae_loss = 0.0

        with torch.no_grad():
            for src_data, tgt_data in data_loader:
                output = self.model(src_data, tgt_data[:, :-1])

                loss = mse(output, tgt_data[:, 1:])
                mse_loss += float(loss.item()) / len(data_loader)
                mae_loss += float(mae(output, tgt_data[:, 1:]).item()) / len(data_loader)

        return mse_loss, math.sqrt(mse_loss), mae_loss
