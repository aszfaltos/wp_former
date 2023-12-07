from torch.utils.data import DataLoader
from torch import nn, optim
import torch


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 batch_size=8,
                 epochs=5):
        self.batch_size = batch_size
        self.model = model
        self.epochs = epochs
        self.metrics = {'train_loss': [], 'eval_loss': []}

    def train(self, train_data, valid_data):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
        # scheduler = optim.lr_scheduler.LinearLR(optimizer, 1.0, 1.0, total_iters=5)
        # cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs * (len(train_data) // self.batch_size) // 256, 1e-05)
        # scheduler = optim.lr_scheduler.ChainedScheduler([linear_scheduler, cos_scheduler])
        train_criterion = nn.MSELoss()
        valid_criterion = nn.MSELoss()

        train_loader = DataLoader(train_data, self.batch_size)
        valid_loader = DataLoader(valid_data, self.batch_size)
        print(len(train_data), len(valid_data))

        for epoch in range(self.epochs):
            self.model.train()
            # TODO: early stop
            train_loss = 0

            for step, (src_data, tgt_data) in enumerate(train_loader):
                optimizer.zero_grad()
                ones = torch.ones(tgt_data.shape[0], 1, tgt_data.shape[-1])
                output = self.model(src_data, torch.concat((ones, tgt_data[:, :-1]), dim=1))
                loss = train_criterion(output, tgt_data)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() / len(train_loader)

            print(
                f"Epoch: {epoch + 1}, Train Loss: {train_loss}",
                end='')

            # scheduler.step()
            self.metrics['train_loss'].append(train_loss)
            self._evaluate(valid_loader, valid_criterion)

    def _evaluate(self, data_loader, criterion):
        self.model.eval()
        eval_loss = 0
        for src_data, tgt_data in data_loader:
            ones = torch.ones(tgt_data.shape[0], 1, tgt_data.shape[-1])
            out = ones
            for _ in range(tgt_data.shape[1]):
                out = torch.concat((ones, self.model(src_data, out)), dim=1)
            loss = criterion(out[:, 1:], tgt_data)
            eval_loss += loss.item() / len(data_loader)

        print(f", Eval Loss: {eval_loss}")
        # TODO: MAE, RMSE, MSE for control

        self.metrics['eval_loss'].append(eval_loss)
        self.model.train()

    # TODO: static eval function
    # TODO: save/load model
