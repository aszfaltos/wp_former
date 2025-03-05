import numpy as np
import pandas as pd
import torch

from data_handling import data_loader
from trainer_lib.datasets import TimeSeriesWindowedTensorDataset, TimeSeriesDatasetConfig
from trainer_lib.trainer import TrainerOptions
from trainer_lib.grid_search import GridSearchOptions, LSTMGridSearch, TransformerGridSearch, VPLSTMGridSearch
from trainer_lib import Grid
import utils
from signal_decomposition import eemd, wavelet


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = data_loader.load_data('data/train/regional_aggregated_data.csv')
    test_df = data_loader.load_data('data/test/regional_aggregated_data.csv')
    train_df = train_df.apply(utils.standardize)
    test_df = test_df.apply(utils.standardize)
    return train_df, test_df # type: ignore


def train_lstm(training_data, test_data, logger):
    training_opts = TrainerOptions(
        batch_size=1024,
        epochs=1200,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=5e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=10,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/lstm/',
        run_nums=1,
        valid_split=0.2,
        random_seed=42,
    )

    dataset_config = TimeSeriesDatasetConfig(
        src_window_size=1,
        tgt_window_size=1,
        src_sequence_length=24,
        tgt_sequence_length=6,
        step_size=1,
        add_start_token=False,
        preprocess_y=False
    )
    train_dataset = TimeSeriesWindowedTensorDataset(training_data, dataset_config)
    test_dataset = TimeSeriesWindowedTensorDataset(test_data, dataset_config)

    params = {
        'in_features': [3],
        'hidden_size': [64],
        'num_layers': [4],
        'out_features': [3],
        'dropout': [.1],
        'in_noise': [.0],
        'hid_noise': [.01],
        'bidirectional': [True],
    }
    grid = Grid(params)

    grid_search = LSTMGridSearch(train_dataset, test_dataset, training_opts, grid_search_opts, logger)
    grid_search.random_search(grid, 5)


def train_lstm_wavelet(training_data, test_data, logger):
    training_opts = TrainerOptions(
        batch_size=1024,
        epochs=1200,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=5e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=10,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/lstm_wavelet/',
        run_nums=1,
        valid_split=0.2,
        random_seed=42,
    )

    dataset_config = TimeSeriesDatasetConfig(
        src_window_size=1,
        tgt_window_size=1,
        src_sequence_length=24,
        tgt_sequence_length=1,
        step_size=1,
        add_start_token=False,
        preprocess_y=True
    )
    train_dataset = TimeSeriesWindowedTensorDataset(training_data, dataset_config, wavelet.WaveletPreprocessor(3, 'haar', 4))

    test_dataset = TimeSeriesWindowedTensorDataset(test_data, dataset_config, wavelet.WaveletPreprocessor(3, 'haar', 4))
    params = {
        'in_features': [15],
        'hidden_size': [64],
        'num_layers': [4],
        'out_features': [15],
        'dropout': [.1],
        'in_noise': [.0],
        'hid_noise': [.01],
        'bidirectional': [True],
    }
    grid = Grid(params)

    grid_search = LSTMGridSearch(train_dataset, test_dataset, training_opts, grid_search_opts, logger)
    grid_search.random_search(grid, 5)


def train_vp_lstm(training_data, test_data, logger):
    training_opts = TrainerOptions(
        batch_size=1024,
        epochs=1200,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=5e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=10,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/vp_lstm/',
        run_nums=1,
        valid_split=0.2,
        random_seed=42,
    )

    dataset_config = TimeSeriesDatasetConfig(
        src_window_size=1,
        tgt_window_size=1,
        src_sequence_length=24,
        tgt_sequence_length=6,
        step_size=4,
        add_start_token=False,
        preprocess_y=True
    )
    train_dataset = TimeSeriesWindowedTensorDataset(training_data, dataset_config)
    test_dataset = TimeSeriesWindowedTensorDataset(test_data, dataset_config)

    params = {
        'vp_bases': [8],
        'vp_penalty': [0.3],
        'in_features': [3],
        'hidden_size': [64],
        'num_layers': [2],
        'out_features': [3],
        'dropout': [.1],
        'in_noise': [.0],
        'hid_noise': [.01],
        'bidirectional': [True],
    }
    grid = Grid(params)

    grid_search = VPLSTMGridSearch(train_dataset, test_dataset, training_opts, grid_search_opts, logger)
    grid_search.random_search(grid, 5)


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

    dataset_config = TimeSeriesDatasetConfig(
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

    dataset_config = TimeSeriesDatasetConfig(
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

    dataset_config = TimeSeriesDatasetConfig(
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

    train_set, test_set = load_data()
    train_set = train_set.interpolate(method='linear', axis=0).ffill().bfill() # type: ignore
    test_set = test_set.interpolate(method='linear', axis=0).ffill().bfill() # type: ignore
    train_set = train_set[['Wind Power [MW] (Net control)', 'Temperature [°C]', 'Wind Speed [m/s]']].to_numpy()
    test_set = test_set[['Wind Power [MW] (Net control)', 'Temperature [°C]', 'Wind Speed [m/s]']].to_numpy()

    logger.info('Data loaded.')

    train_lstm(train_set, test_set, logger)
    train_lstm_wavelet(train_set, test_set, logger)
    train_vp_lstm(train_set, test_set, logger)


if __name__ == '__main__':
    main()
