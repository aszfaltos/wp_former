import numpy as np
import torch

from data_handling.data_loader import load_mavir_data
from trainer_lib import Grid, transformer_grid_search, TrainerOptions, GridSearchOptions
import utils
from signal_decomposition import eemd


def load_data(sample_size, start_idx):
    df = load_mavir_data('data/mavir_data/mavir.csv')
    df['Power'] = utils.min_max_norm(df['Power'])
    return utils.sample(df, sample_size, start_idx=start_idx)


def train_regular_transformer(training_data):
    params = {
        'src_size': [1],
        'tgt_size': [1],
        'd_model': [256],
        'num_heads': [2],
        'num_layers': [2],
        'd_ff': [1024],
        'src_seq_length': [24],
        'tgt_seq_length': [1],
        'src_window': [8],
        'tgt_window': [1],
        'dropout': [0.2],
    }
    grid = Grid(params)

    training_opts = TrainerOptions(
        batch_size=8,
        epochs=30,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_steps=10,
        warmup_start_factor=1e-6,
        gradient_accumulation_steps=8,
        early_stopping_patience=5,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/regular/',
        valid_split=0.35,
        window_step_size=4,
        random_seed=43,
        use_start_token=True
    )

    transformer_grid_search(grid, training_data, training_opts, grid_search_opts)


def train_eemd_transformer(training_data):
    params = {
        'src_size': [training_data.shape[-1]],
        'tgt_size': [training_data.shape[-1]],
        'd_model': [256],
        'num_heads': [2],
        'num_layers': [2],
        'd_ff': [1024],
        'src_seq_length': [24],
        'tgt_seq_length': [1],
        'src_window': [8],
        'tgt_window': [1],
        'dropout': [0.2],
    }
    grid = Grid(params)

    training_opts = TrainerOptions(
        batch_size=8,
        epochs=30,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_steps=10,
        warmup_start_factor=1e-6,
        gradient_accumulation_steps=8,
        early_stopping_patience=5,
        early_stopping_min_delta=0.01,
        save_every_n_epochs=5,
        save_path=''
    )

    grid_search_opts = GridSearchOptions(
        root_save_path='./trained/eemd/',
        valid_split=0.35,
        window_step_size=4,
        random_seed=43,
        use_start_token=True
    )

    transformer_grid_search(grid, training_data, training_opts, grid_search_opts)


def main():
    sample = load_data(10000, 0)
    print('data loaded')
    training_data_regular = np.array(sample['Power'].to_numpy()[..., np.newaxis], dtype=np.float32)
    print('regular train data prepared')

    decomposed = eemd.EEMDWrapper(sample['Power'].to_numpy(), 3)
    training_data_eemd = np.array(np.concatenate([decomposed.get_imfs().transpose(),
                                                  decomposed.get_residue()[..., np.newaxis]], dtype=np.float32, axis=1))
    print('eemd train data prepared')

    print('regular transformer:')
    train_regular_transformer(training_data_regular)
    print('eemd transformer:')
    train_eemd_transformer(training_data_eemd)


if __name__ == '__main__':
    main()
