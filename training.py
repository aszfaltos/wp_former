import numpy as np
import torch

from data_handling import data_loader
from trainer_lib.datasets import TimeSeriesWindowedTensorDataset, TimeSeriesWindowedDatasetConfig
from trainer_lib.trainer import TrainerOptions
from trainer_lib.grid_search import GridSearchOptions, LSTMGridSearch, TransformerGridSearch
from trainer_lib import Grid
import utils
from signal_decomposition import eemd, wavelet


def load_data(sample_size, start_idx):
    df = data_loader.load_data('data/hourly/regional_aggregated_data.csv')
    df = df.apply(utils.min_max_norm)
    return utils.sample(df, sample_size, start_idx=start_idx)


def train_lstm(training_data, logger):
    training_opts = TrainerOptions(
        batch_size=1024,
        epochs=600,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=1e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=30,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/lstm/',
        run_nums=5,
        valid_split=0.2,
        test_split=0.2,
        random_seed=42,
    )

    dataset_config = TimeSeriesWindowedDatasetConfig(
        src_window_size=1,
        tgt_window_size=1,
        src_sequence_length=24,
        tgt_sequence_length=1,
        step_size=4,
        add_start_token=False,
        preprocess_y=True
    )
    dataset = TimeSeriesWindowedTensorDataset(training_data, dataset_config)

    params = {
        'in_features': [3],
        'hidden_size': [64],
        'num_layers': [2],
        'out_features': [15],
        'dropout': [.03],
        'in_noise': [.0],
        'hid_noise': [.01],
        'bidirectional': [True],
    }
    grid = Grid(params)

    grid_search = LSTMGridSearch(dataset, training_opts, grid_search_opts, logger)
    grid_search.search(grid)


def train_lstm_wavelet(training_data, logger):
    training_opts = TrainerOptions(
        batch_size=1024,
        epochs=600,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=1e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=30,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/lstm_wavelet/',
        run_nums=5,
        valid_split=0.2,
        test_split=0.2,
        random_seed=42,
    )

    dataset_config = TimeSeriesWindowedDatasetConfig(
        src_window_size=1,
        tgt_window_size=1,
        src_sequence_length=24,
        tgt_sequence_length=1,
        step_size=4,
        add_start_token=False,
        preprocess_y=True
    )
    dataset = TimeSeriesWindowedTensorDataset(training_data, dataset_config, wavelet.WaveletPreprocessor(3, 'haar', 4))

    params = {
        'in_features': [15],
        'hidden_size': [64],
        'num_layers': [2],
        'out_features': [15],
        'dropout': [.03],
        'in_noise': [.0],
        'hid_noise': [.01],
        'bidirectional': [True],
    }
    grid = Grid(params)

    grid_search = LSTMGridSearch(dataset, training_opts, grid_search_opts, logger)
    grid_search.search(grid)


def train_vp_lstm(training_data, logger):
    training_opts = TrainerOptions(
        batch_size=1024,
        epochs=600,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=1e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=30,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/vp_lstm/',
        run_nums=5,
        valid_split=0.2,
        test_split=0.2,
        random_seed=42,
    )

    dataset_config = TimeSeriesWindowedDatasetConfig(
        src_window_size=1,
        tgt_window_size=1,
        src_sequence_length=24,
        tgt_sequence_length=1,
        step_size=4,
        add_start_token=False,
        preprocess_y=True
    )
    dataset = TimeSeriesWindowedTensorDataset(training_data, dataset_config)

    params = {
        'in_features': [3],
        'hidden_size': [64],
        'num_layers': [2],
        'out_features': [15],
        'dropout': [.03],
        'in_noise': [.0],
        'hid_noise': [.01],
        'bidirectional': [True],
    }
    grid = Grid(params)

    grid_search = LSTMGridSearch(dataset, training_opts, grid_search_opts, logger)
    grid_search.search(grid)


def train_transformer(training_data, logger):
    training_opts = TrainerOptions(
        batch_size=1024,
        epochs=600,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=1e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=30,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/transformer/',
        run_nums=5,
        valid_split=0.2,
        test_split=0.2,
        random_seed=42,
    )

    dataset_config = TimeSeriesWindowedDatasetConfig(
        src_window_size=1,
        tgt_window_size=1,
        src_sequence_length=24,
        tgt_sequence_length=1,
        step_size=4,
        add_start_token=True,
        preprocess_y=True
    )
    dataset = TimeSeriesWindowedTensorDataset(training_data, dataset_config)

    params = {
        "src_size": [3],
        "tgt_size": [3],
        "d_model": [64],
        "num_heads": [2, 4],
        "num_layers": [2, 4],
        "d_ff": [128, 256],
        "max_seq_length": [24],
        "dropout": [.01, .03, .1]
    }
    grid = Grid(params)

    grid_search = TransformerGridSearch(dataset, training_opts, grid_search_opts, logger)
    grid_search.search(grid)


def train_transformer_wavelet(training_data, logger):
    training_opts = TrainerOptions(
        batch_size=1024,
        epochs=600,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=1e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=30,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/transformer_wavelet/',
        run_nums=5,
        valid_split=0.2,
        test_split=0.2,
        random_seed=42,
    )

    dataset_config = TimeSeriesWindowedDatasetConfig(
        src_window_size=1,
        tgt_window_size=1,
        src_sequence_length=24,
        tgt_sequence_length=1,
        step_size=4,
        add_start_token=True,
        preprocess_y=True
    )
    dataset = TimeSeriesWindowedTensorDataset(training_data, dataset_config, wavelet.WaveletPreprocessor(3, 'haar', 4))

    params = {
        "src_size": [15],
        "tgt_size": [15],
        "d_model": [64],
        "num_heads": [2, 4],
        "num_layers": [2, 4],
        "d_ff": [128, 256],
        "max_seq_length": [24],
        "dropout": [.01, .03, .1]
    }
    grid = Grid(params)

    grid_search = TransformerGridSearch(dataset, training_opts, grid_search_opts, logger)
    grid_search.search(grid)


def train_vp_transformer(training_data, logger):
    training_opts = TrainerOptions(
        batch_size=1024,
        epochs=600,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=1e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=30,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/vp_transformer/',
        run_nums=5,
        valid_split=0.2,
        test_split=0.2,
        random_seed=42,
    )

    dataset_config = TimeSeriesWindowedDatasetConfig(
        src_window_size=1,
        tgt_window_size=1,
        src_sequence_length=24,
        tgt_sequence_length=1,
        step_size=4,
        add_start_token=True,
        preprocess_y=True
    )
    dataset = TimeSeriesWindowedTensorDataset(training_data, dataset_config)

    params = {
        "src_size": [3],
        "tgt_size": [3],
        "d_model": [64],
        "num_heads": [2, 4],
        "num_layers": [2, 4],
        "d_ff": [128, 256],
        "max_seq_length": [24],
        "dropout": [.01, .03, .1],
        'vp_bases': [8],
        'vp_penalty': [0.2],
    }
    grid = Grid(params)

    grid_search = TransformerGridSearch(dataset, training_opts, grid_search_opts, logger)
    grid_search.search(grid)


def main():
    logger = utils.Logger('trainer')

    sample = load_data(100000, 0)
    sample = sample.interpolate(method='linear', axis=0).ffill().bfill()
    sample = sample[['Wind Power [MW] (Net control)', 'Temperature [°C]', 'Wind Speed [m/s]']].to_numpy()

    logger.info('Data loaded.')

    train_transformer(sample, logger)


if __name__ == '__main__':
    main()
