import math
import os.path
import json
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .early_stop import EarlyStopper
from .utils import checkpoint


@dataclass
class TrainerOptions:
    batch_size: int
    epochs: int

    learning_rate: float
    learning_rate_decay: float
    weight_decay: float
    warmup_steps: int
    warmup_start_factor: float
    gradient_accumulation_steps: int

    early_stopping_patience: int
    early_stopping_min_delta: float

    save_every_n_epochs: int
    save_path: str


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 opts: TrainerOptions):
        self.model = model
        self.opts = opts
        self.metrics = {'train': {'MSE': []},
                        'eval': {'MSE': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
                        'test': {'MSE': .0, 'RMSE': .0, 'MAE': .0, 'MAPE': .0}}
        os.makedirs(self.opts.save_path, exist_ok=True)

    def train(self, train_data, valid_data, test_data):
        early_stopper = EarlyStopper(self.opts.early_stopping_patience, self.opts.early_stopping_min_delta)
        optimizer = optim.Adam(self.model.parameters(), lr=self.opts.learning_rate, betas=(0.9, 0.98), eps=1e-9,
                               weight_decay=self.opts.weight_decay)

        warmup = optim.lr_scheduler.LinearLR(optimizer, self.opts.warmup_start_factor, 1.0,
                                             total_iters=self.opts.warmup_steps)
        exp_sch = optim.lr_scheduler.ExponentialLR(optimizer, self.opts.learning_rate_decay)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, exp_sch], [self.opts.warmup_steps])

        mse = nn.MSELoss()

        train_loader = DataLoader(train_data, self.opts.batch_size)
        valid_loader = DataLoader(valid_data, self.opts.batch_size)
        test_loader = DataLoader(test_data, self.opts.batch_size)
        print(f'Train size: {len(train_data)}, Validation size: {len(valid_data)}, Test size: {len(valid_data)}')

        last_epoch = 0
        for epoch in range(self.opts.epochs):
            last_epoch = epoch
            train_loss: float = 0

            for (batch_idx, (src_data, tgt_data)) in enumerate(train_loader):
                output = self.model(src_data, tgt_data[:, :-1])
                loss = mse(output, tgt_data[:, 1:])
                train_loss += float(loss.item()) / len(train_loader)
                loss = loss / self.opts.gradient_accumulation_steps
                loss.backward()

                if ((batch_idx + 1) % self.opts.gradient_accumulation_steps == 0) or \
                        (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()

            print(
                f"Epoch: {epoch + 1}; Learning rate: {scheduler.get_last_lr()}; Train - MSE: {train_loss}",
                end='')

            self.metrics['train']['MSE'].append(train_loss)
            stop = self._evaluate(valid_loader, early_stopper)

            if (epoch + 1) % self.opts.save_every_n_epochs == 0:
                checkpoint(self.model, os.path.join(self.opts.save_path, f'{epoch + 1}.pth'))
                with open(os.path.join(self.opts.save_path, f'{epoch + 1}.json'), "w") as fp:
                    json.dump(self.metrics, fp)

            if stop:
                print(f'Stopped after {epoch + 1} epochs.')
                break

        self._test_model(test_loader)
        checkpoint(self.model, os.path.join(self.opts.save_path, f'{last_epoch + 1}.pth'))
        with open(os.path.join(self.opts.save_path, f'{last_epoch + 1}.json'), "w") as fp:
            json.dump(self.metrics, fp)

    def _calc_metrics(self, data_loader):
        self.model.eval()

        mse = nn.MSELoss()
        mae = nn.L1Loss()
        mse_loss: float = 0
        rmse_loss: float = 0
        mae_loss: float = 0
        mape_loss: float = 0

        with torch.no_grad():
            for src_data, tgt_data in data_loader:
                ones = tgt_data[:, 0, :].reshape(tgt_data.shape[0], 1, tgt_data.shape[-1])
                out = ones
                for _ in range(tgt_data.shape[1] - 1):
                    out = torch.concat((ones, self.model(src_data, out)), dim=1)

                loss = mse(out[:, 1:], tgt_data[:, 1:])
                mse_loss += float(loss.item()) / len(data_loader)
                rmse_loss += math.sqrt(float(loss.item())) / len(data_loader)
                mae_loss += float(mae(out[:, 1:], tgt_data[:, 1:]).item()) / len(data_loader)
                mape_loss += mae_loss / float(sum(tgt_data[:, 1:].reshape(-1)))

        self.model.train()

        return mse_loss, rmse_loss, mae_loss, mape_loss

    def _evaluate(self, data_loader, early_stopper: EarlyStopper):
        mse, rmse, mae, mape = self._calc_metrics(data_loader)

        print(
            f"; Eval - MSE: {mse}," +
            f" RMSE: {rmse}," +
            f" MAE: {mae}," +
            f" MAPE: {round(mape * 100, 2)}")

        self.metrics['eval']['MSE'].append(mse)
        self.metrics['eval']['RMSE'].append(rmse)
        self.metrics['eval']['MAE'].append(mae)
        self.metrics['eval']['MAPE'].append(mape)

        return early_stopper.early_stop(mse)

    def _test_model(self, data_loader):
        mse, rmse, mae, mape = self._calc_metrics(data_loader)

        print("-----------------------")
        print(
            f"Test - MSE: {mse}," +
            f" RMSE: {rmse}," +
            f" MAE: {mae}," +
            f" MAPE: {round(mape * 100, 2)}")
        print("-----------------------\n")

        self.metrics['test']['MSE'] = mse
        self.metrics['test']['RMSE'] = rmse
        self.metrics['test']['MAE'] = mae
        self.metrics['test']['MAPE'] = mape
