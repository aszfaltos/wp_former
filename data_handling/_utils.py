import datetime
import os
from shutil import rmtree
import logging
import pandas as pd

from utils import exit_handling

omsz_csv_type_dict = {
    'StationNumber': 'Int64',
    'Time': 'str',
    'r': 'float64',
    'Q_r': 'str',
    't': 'float64',
    'Q_t': 'str',
    'ta': 'float64',
    'Q_ta': 'str',
    'tn': 'float64',
    'Q_tn': 'str',
    'tx': 'float64',
    'Q_tx': 'str',
    'v': 'float64',
    'Q_v': 'str',
    'p': 'float64',
    'Q_p': 'str',
    'u': 'float64',
    'Q_u': 'str',
    'sg': 'float64',
    'Q_sg': 'str',
    'sr': 'float64',
    'Q_sr': 'str',
    'suv': 'float64',
    'Q_suv': 'str',
    'fs': 'float64',
    'Q_fs': 'str',
    'fsd': 'float64',
    'Q_fsd': 'str',
    'fx': 'float64',
    'Q_fx': 'str',
    'fxd': 'float64',
    'Q_fxd': 'str',
    'fxdat': 'str',
    'Q_fxdat': 'str',
    'we': 'Int64',
    'Q_we': 'str',
    'p0': 'float64',
    'Q_p0': 'str',
    'f': 'float64',
    'Q_f': 'str',
    'fd': 'float64',
    'Q_fd': 'str',
    'et5': 'float64',
    'Q_et5': 'str',
    'et10': 'float64',
    'Q_et10': 'str',
    'et20': 'float64',
    'Q_et20': 'str',
    'et50': 'float64',
    'Q_et50': 'str',
    'et100': 'float64',
    'Q_et100': 'str',
    'tsn': 'float64',
    'Q_tsn': 'str',
    'tviz': 'float64',
    'Q_tviz': 'str',
}


def exiting(logger: logging.Logger, temp_path: str):
    if exit_handling.exit_hooks.exit_code != 0 or exit_handling.exit_hooks.exception is not None:
        logger.warning("Download may be corrupted, because program was interrupted or an exception occurred!")

    if not os.path.exists(temp_path):
        return

    logger.info('Cleaning up temporary files...')
    rmtree(temp_path)


def check_datetime(date_str: str):
    try:
        pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False
