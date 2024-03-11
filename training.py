import numpy as np
import torch

from data_handling.data_loader import load_mavir_data
from trainer_lib import Grid, transformer_grid_search, TrainerOptions, GridSearchOptions
import utils
from signal_decomposition import eemd, wavelet


def load_data(sample_size, start_idx):
    df = load_mavir_data('data/mavir_data/mavir.csv')
    df['Power'] = utils.min_max_norm(df['Power'])
    return utils.sample(df, sample_size, start_idx=start_idx)


def set_default_options():
    params = {
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
        batch_size=8,
        epochs=30,
        learning_rate=1e-3,
        learning_rate_decay=.999,
        weight_decay=1e-5,
        warmup_steps=10,
        warmup_start_factor=1e-6,
        gradient_accumulation_steps=8,
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


def main():
    sample = load_data(5000, 0)
    sample = sample['Power'].to_numpy()
    print('data loaded')

    print('regular transformer:')
    train_regular_transformer(sample)
    print('eemd transformer:')
    #train_eemd_transformer(sample)
    print('wlt transformer:')
    train_wavelet_transformer(sample)


if __name__ == '__main__':
    main()
    # TODO: test, lr, model sizes, wd, dropout, lr decay
