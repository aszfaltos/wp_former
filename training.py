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


def set_default_options():
    params = {
        'kind': ['transformer'],
        'd_model': [128],
        'num_heads': [2],
        'num_layers': [1],
        'd_ff': [2],
        'src_seq_length': [24],
        'tgt_seq_length': [1],
        'src_window': [8],
        'tgt_window': [1],
        'dropout': [0.1],
    }
    grid = Grid(params)

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

    return grid, training_opts


def train_regular_transformer(training_data):
    grid, training_opts = set_default_options()
    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/regular/',
        valid_split=0.2,
        test_split=0.2,
        window_step_size=4,
        random_seed=50,
        use_start_token=True,
        preprocess_y=False
    )

    transformer_grid_search(grid, training_data, training_opts, grid_search_opts)


def train_eemd_transformer(training_data):
    grid, training_opts = set_default_options()

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/eemd/',
        valid_split=0.2,
        test_split=0.2,
        window_step_size=4,
        random_seed=46,
        use_start_token=True,
        preprocess_y=False
    )

    transformer_grid_search(grid, training_data, training_opts, grid_search_opts,
                            preprocessor=eemd.EEMDPreprocessor(imfs=3, trials=20))


def train_wavelet_transformer(training_data):
    grid, training_opts = set_default_options()

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/wavelet/',
        valid_split=0.2,
        test_split=0.2,
        window_step_size=4,
        random_seed=50,
        use_start_token=True,
        preprocess_y=False
    )

    transformer_grid_search(grid, training_data, training_opts, grid_search_opts,
                            preprocessor=wavelet.WaveletPreprocessor('db2'))


def train_wavelet_vp_transformer(training_data):
    _, training_opts = set_default_options()

    params = {
        'kind': ['vp_transformer'],
        'vp_bases': [8],
        'vp_penalty': [0.2],
        'd_model': [128],
        'num_heads': [2],
        'num_layers': [1],
        'd_ff': [2],
        'src_seq_length': [24*8],
        'tgt_seq_length': [1],
        'src_window': [1],
        'tgt_window': [1],
        'dropout': [0.1],
    }
    grid = Grid(params)

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/vp_wavelet/',
        valid_split=0.2,
        test_split=0.2,
        window_step_size=4,
        random_seed=50,
        use_start_token=True,
        preprocess_y=False
    )

    transformer_grid_search(grid, training_data, training_opts, grid_search_opts,
                            preprocessor=wavelet.WaveletPreprocessor('db2'))


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


def main():
    logger = utils.Logger('trainer')

    sample = load_data(100000, 0)
    sample = sample.interpolate(method='linear', axis=0).ffill().bfill()
    sample = sample[['Wind Power [MW] (Net control)', 'Temperature [°C]', 'Wind Speed [m/s]']].to_numpy()

    logger.info('Data loaded.')

    train_transformer(sample, logger)


if __name__ == '__main__':
    main()
