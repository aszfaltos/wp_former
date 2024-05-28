import atexit
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from shutil import rmtree
from enum import Enum
import logging
from requests import get as req_get
import pandas as pd

from utils import exit_handling

logger = logging.getLogger(__name__)


def exiting():
    if exit_handling.exit_hooks.exit_code != 0 or exit_handling.exit_hooks.exception is not None:
        logger.warning("Download may be corrupted, because program was interrupted or an exception occurred!")

    if not os.path.exists('temp_data'):
        return

    logger.info('Cleaning up temporary files...')
    rmtree('temp_data')


class PeriodType(Enum):
    MIN = 'min'
    HOUR = 'hour'


def format_mavir(dataframe: pd.DataFrame):
    dataframe.columns = dataframe.columns.str.strip()
    dataframe.index = pd.to_datetime(dataframe['Időpont'], utc=True)
    dataframe.index.name = 'Time'
    dataframe.drop(['Időpont'], axis=1, inplace=True)
    dataframe.dropna(axis=0, inplace=True)
    dataframe.rename(columns={'Szélerőművek tény - bruttó üzemirányítási': 'Wind Power [MW] (actual)',
                              'Szélerőművek becsült termelése (aktuális)': 'Wind Power [MW] (estimated)',
                              'Szélerőművek becsült termelése (dayahead)': 'Wind Power [MW] (dayahead)',
                              'Szélerőművek becsült termelése (intraday)': 'Wind Power [MW] (intraday)'},
                     inplace=True)

    dataframe.drop(['Szélerőművek tény - nettó kereskedelmi elszámolási',
                    'Szélerőművek tény - nettó üzemirányítási',
                    'Szélerőművek tény - bruttó üzemirányítási 1p'],
                   inplace=True, axis=1)
    return dataframe


def download_mavir_data(path: str, from_time: str, to_time: str, period=10):
    """
    Downloads data from MAVIR.
    :param path: path to save the data
    :param from_time: in format 'YYYY-MM-DD hh:mm:ss'
    :param to_time: in format 'YYYY-MM-DD hh:mm:ss'
    :param period: period of the data in minutes
    :return:
    """
    from_in_ms = prep_datetime(from_time)
    to_in_ms = prep_datetime(to_time)
    formatted_period, period_type = (period // 60, PeriodType.HOUR) if period % 60 == 0 else (period, PeriodType.MIN)

    # Splitting the request into n parts, because the request fails if it's too long (more than 60_000 lines).
    period_in_ms = period * 60 * 1000
    lines_in_interval = (to_in_ms - from_in_ms) // period_in_ms + 1
    n = lines_in_interval // 60_000 + 1
    fraction_in_ms = (to_in_ms - from_in_ms) // n

    dfs = []
    # Sleep is needed to not spam the api.
    sleep_time = 5
    for i in range(0, n):
        logger.info(f"Downloading {i}. fragment")
        start = datetime.now()

        df = download_from_to(f'mavir_{i}',
                              from_in_ms + i * fraction_in_ms,
                              from_in_ms + (i + 1) * fraction_in_ms,
                              period=formatted_period,
                              period_type=period_type)
        if isinstance(df, pd.DataFrame):
            dfs.append(df)

        runtime = (datetime.now() - start).total_seconds()
        logger.info(f"Downloaded {i}. fragment in {runtime} seconds")

        time.sleep(sleep_time)

    pd.concat(dfs).to_csv(path, sep=',', decimal='.')

    if not os.path.exists('mavir_data'):
        logger.info('Creating mavir_data directory')
        os.makedirs('mavir_data')


def prep_datetime(dt: str):
    """
    Prepares datetime for MAVIR request.
    :param dt: in format 'YYYY-MM-DD hh:mm:ss'
    :return: datetime in ms
    """
    dt = pd.to_datetime(dt, format='%Y-%m-%d %H:%M:%S', utc=True)
    return int(dt.value / 1e6)


def download_from_to(name: str, from_time: int, to_time: int, period=10,
                     period_type: PeriodType = PeriodType.MIN):
    """
    Time should be given as ms.
    The measurements are taken every 10 minutes.
    """
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    if not os.path.exists('mavir_data'):
        os.makedirs('mavir_data')

    print(from_time, to_time, period, period_type)

    url = (f"https://www.mavir.hu/rtdwweb/webuser/chart/11840/export"
           f"?exportType=xlsx"
           f"&fromTime={from_time}"
           f"&toTime={to_time}"
           f"&periodType={period_type.value}"
           f"&period={period}")

    print(url)

    temp_path = f"temp_data/{name}.xlsx"
    response = req_get(url, timeout=240)

    if response.status_code == 200:
        with open(temp_path, 'wb') as f:
            f.write(response.content)
            print(f"Downloaded {temp_path}")
    else:
        logger.error(f"Error {response.status_code} for request.\nError message: {response.content.decode()}")
        exit(1)

    # suppress default style warning
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pd.read_excel(temp_path, skiprows=0,
                           parse_dates=True, engine='openpyxl')

    return format_mavir(df)


# TODO: this won't be needed when we have meteorological data too
def load_mavir_data():
    df = pd.read_csv(filepath_or_buffer='./mavir_data/mavir.csv',
                     delimiter=',')

    df['Time'] = pd.to_datetime(df['Time'])

    return df


def main():
    # Test downloads
    download_mavir_data('../data/mavir_data/mavir_test_10.csv',
                        '2024-01-01 00:00:00',
                        '2024-01-02 00:00:00',
                        period=10)
    download_mavir_data('../data/mavir_data/mavir_test_30.csv',
                        '2024-01-01 00:00:00',
                        '2024-01-02 00:00:00',
                        period=30)
    download_mavir_data('../data/mavir_data/mavir_test_60.csv',
                        '2024-01-01 00:00:00',
                        '2024-01-02 00:00:00',
                        period=60)
    download_mavir_data('../data/mavir_data/mavir_test_120.csv',
                        '2024-01-01 00:00:00',
                        '2024-01-02 00:00:00',
                        period=120)


if __name__ == '__main__':
    atexit.register(exiting)
    main()
