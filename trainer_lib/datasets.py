from torch.utils.data import TensorDataset
import torch


DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class TimeSeriesTensorDataset(TensorDataset):
    def __init__(self, x, y, seq_len, pred_len):
        super(TimeSeriesTensorDataset, self).__init__(torch.tensor(x), torch.tensor(y))
        self.X = self.tensors[0].to(DEVICE)
        self.y = self.tensors[1].to(DEVICE)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len:idx+self.seq_len+self.pred_len]