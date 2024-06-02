import atexit
import os
import time
import warnings
from datetime import datetime
from enum import Enum
import logging
from requests import get as req_get
import pandas as pd

from ._utils import exiting

# TODO: make this env var for docker
TEMP_DATA_PATH = 'temp_data'

# TODO: Create custom logger for the whole codebase should send you emails about errors when run in docker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PeriodType(Enum):
    MIN = 'min'
    HOUR = 'hour'


def download_mavir_data(path: str, from_time: str, to_time: str, period=10):
    """
    Downloads data from MAVIR.
    :param path: path to save the data
    :param from_time: in format 'YYYY-MM-DD hh:mm:ss'
    :param to_time: in format 'YYYY-MM-DD hh:mm:ss'
    :param period: period of the data in minutes
    :return:
    """
    atexit.register(exiting, logger, TEMP_DATA_PATH)

    from_in_ms = prep_datetime(from_time)
    to_in_ms = prep_datetime(to_time)
    formatted_period, period_type = (period // 60, PeriodType.HOUR) if period % 60 == 0 else (period, PeriodType.MIN)

    # Splitting the request into n parts, because the request fails if it's too long (more than 60_000 lines).
    period_in_ms = period * 60 * 1000
    lines_in_interval = (to_in_ms - from_in_ms) // period_in_ms + 1
    n = lines_in_interval // 60_000 + 1
    if lines_in_interval > 30_000:
        n *= 4  # splitting every fragment into 4 parts for memory reasons
    fraction_in_ms = (to_in_ms - from_in_ms) // n

    dfs = []
    # Sleep is needed to not spam the api.
    sleep_time = 5
    for i in range(0, n):
        logger.info(f"Downloading {i+1}. fragment")
        start = datetime.now()

        df = download_from_to(f'mavir_{i}',
                              from_in_ms + i * fraction_in_ms,
                              from_in_ms + (i + 1) * fraction_in_ms,
                              period=formatted_period,
                              period_type=period_type)
        if isinstance(df, pd.DataFrame):
            dfs.append(df)

        runtime = (datetime.now() - start).total_seconds()
        logger.info(f"Downloaded {i+1}. fragment in {runtime} seconds")

        time.sleep(sleep_time)

    if not os.path.exists(path):
        os.makedirs(path)

    pd.concat(dfs).to_csv(os.path.join(path, 'wind.csv'), sep=',', decimal='.', date_format='%Y-%m-%d %H:%M:%S')

    atexit.unregister(exiting)


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
    if not os.path.exists(TEMP_DATA_PATH):
        os.makedirs(TEMP_DATA_PATH)

    url = (f"https://www.mavir.hu/rtdwweb/webuser/chart/11840/export"
           f"?exportType=xlsx"
           f"&fromTime={from_time}"
           f"&toTime={to_time}"
           f"&periodType={period_type.value}"
           f"&period={period}")

    temp_path = os.path.join(TEMP_DATA_PATH, f'{name}.xlsx')
    response = req_get(url, timeout=240)

    if response.status_code == 200:
        with open(temp_path, 'wb') as f:
            f.write(response.content)
            logger.debug(f"Downloaded {temp_path}")
    else:
        logger.error(f"Error {response.status_code} for request.\nError message: {response.content.decode()}")
        exit(1)

    # suppress default style warning
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pd.read_excel(temp_path, skiprows=0,
                           parse_dates=True, engine='openpyxl')

    return format_csv(df)


def format_csv(dataframe: pd.DataFrame):
    dataframe.columns = dataframe.columns.str.strip()
    dataframe.index = pd.to_datetime(dataframe['Időpont'], utc=True)
    dataframe.index.name = 'Time'

    dataframe.drop(['Időpont'], inplace=True, axis=1)

    dataframe.rename(columns={'Szélerőművek tény - bruttó üzemirányítási': 'Wind Power [MW] (Gross control)',
                              'Szélerőművek becsült termelése (aktuális)': 'Estimated Wind Power [MW] (current)',
                              'Szélerőművek becsült termelése (dayahead)': 'Estimated Wind Power [MW] (dayahead)',
                              'Szélerőművek becsült termelése (intraday)': 'Estimated Wind Power [MW] (intraday)',
                              'Szélerőművek tény - nettó kereskedelmi elszámolási':
                                  'Wind Power [MW] (Net commercial settlement)',
                              'Szélerőművek tény - nettó üzemirányítási': 'Wind Power [MW] (Net control)',
                              'Szélerőművek tény - bruttó üzemirányítási 1p': 'Wind Power [MW] (Gross control 1p)'
                              },
                     inplace=True)
    return dataframe


def main():
    # Test downloads
    download_mavir_data('../data/mavir_data',
                        '2021-01-01 00:00:00',
                        '2024-01-02 00:00:00',
                        period=10)
    download_mavir_data('../data/mavir_data',
                        '2024-01-01 00:00:00',
                        '2024-01-02 00:00:00',
                        period=30)
    download_mavir_data('../data/mavir_data',
                        '2024-01-01 00:00:00',
                        '2024-01-02 00:00:00',
                        period=60)
    download_mavir_data('../data/mavir_data',
                        '2024-01-01 00:00:00',
                        '2024-01-02 00:00:00',
                        period=120)


if __name__ == '__main__':
    main()
