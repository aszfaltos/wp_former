from pandas import DataFrame
from numpy import ndarray


def sample(data: DataFrame, sample_len: int, start_idx=0):
    return data.iloc[start_idx:start_idx+sample_len, :]


def standardize(data: ndarray):
    return (data - data.mean()) / data.std()


def min_max_norm(data: ndarray):
    return ((data - data.min()) / (data.max() - data.min()) - .5) * 2
