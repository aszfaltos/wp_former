import math
import os.path
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from .utils import checkpoint
from .early_stop import EarlyStopper


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 save_path: str,
                 batch_size=8,
                 epochs=5):
        self.batch_size = batch_size
        self.model = model
        self.epochs = epochs
        self.metrics = {'train': {'MSE': []}, 'eval': {'MSE': [], 'RMSE': [], 'MAE': []}}
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def train(self, train_data, valid_data):
        early_stopper = EarlyStopper(10, 0.1)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

        warmup = optim.lr_scheduler.LinearLR(optimizer, 5*1e-3, 1.0, total_iters=10)
        exp_sch = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, exp_sch], [5])

        criterion = nn.MSELoss()

        train_loader = DataLoader(train_data, self.batch_size)
        valid_loader = DataLoader(valid_data, self.batch_size)
        print(f'Train size: {len(train_data)}, Validation size: {len(valid_data)}')

        for epoch in range(self.epochs):
            train_loss = 0

            for step, (src_data, tgt_data) in enumerate(train_loader):
                optimizer.zero_grad()
                ones = torch.ones(tgt_data.shape[0], 1, tgt_data.shape[-1])
                output = self.model(src_data, torch.concat((ones, tgt_data[:, :-1]), dim=1))
                loss = criterion(output, tgt_data)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() / len(train_loader)
            scheduler.step()

            print(
                f"Epoch: {epoch + 1}; Learning rate: {scheduler.get_last_lr()}; Train - MSE: {train_loss}",
                end='')

            self.metrics['train']['MSE'].append(train_loss)
            stop = self._evaluate(valid_loader, criterion, early_stopper)

            if early_stopper.min_validation_loss == self.metrics['eval']['MSE'][-1]:
                checkpoint(self.model, os.path.join(self.save_path, 'best_mse.pth'))
            if stop:
                print(f'Stopped after {epoch + 1} epochs.')
                break

    def _evaluate(self, data_loader, criterion, early_stopper: EarlyStopper):
        mae = nn.L1Loss()
        self.model.eval()
        eval_loss = 0
        mae_loss = 0
        with torch.no_grad():
            for src_data, tgt_data in data_loader:
                ones = torch.ones(tgt_data.shape[0], 1, tgt_data.shape[-1])
                out = ones
                for _ in range(tgt_data.shape[1]):
                    out = torch.concat((ones, self.model(src_data, out)), dim=1)
                loss = criterion(out[:, 1:], tgt_data)
                eval_loss += loss.item() / len(data_loader)
                mae_loss += mae(out[:, 1:], tgt_data) / len(data_loader)

            print(f"; Eval - MSE: {eval_loss}, RMSE: {math.sqrt(eval_loss)}, MAE: {mae_loss}")

            self.metrics['eval']['MSE'].append(eval_loss)
            self.metrics['eval']['RMSE'].append(math.sqrt(eval_loss))
            self.metrics['eval']['MAE'].append(mae_loss)
        self.model.train()
        return early_stopper.early_stop(eval_loss)
