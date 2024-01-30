import torch


def checkpoint(model: torch.nn.Module, file_name: str):
    torch.save(model.state_dict(), file_name)


def resume(model: torch.nn.Module, file_name: str):
    model.load_state_dict(torch.load(file_name))
