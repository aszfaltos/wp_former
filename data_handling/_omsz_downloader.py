import datetime

from requests import get as req_get
import bs4
import re
import sys
from zipfile import ZipFile
import os
import logging
import pandas as pd
import atexit
from enum import Enum
from dataclasses import dataclass

from ._utils import exiting, omsz_csv_type_dict


# TODO: make this env var for docker
TEMP_DATA_PATH = 'temp_data'

# TODO: Create custom logger for the whole codebase should send you emails about errors when run in docker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PeriodType(Enum):
    MIN = '10_minutes'
    HOUR = 'hourly'


@dataclass
class OMSZData:
    base_url: str = 'https://odp.met.hu/climate/observations_hungary/'
    historical_url: str = 'historical/'
    recent_url: str = 'recent/'
    meta_data_file: str = 'station_meta_auto.csv'


def download_omsz_data(path: str, from_time: str, to_time: str, period: int):
    """
    Downloads data from OMSZ.
    :param path: The path to save the data.
    :param from_time: The start time in format "YYYY-MM-DD hh:mm:ss".
    :param to_time: The end time in format "YYYY-MM-DD hh:mm:ss".
    :param period: The period of the data in minutes, data is only available in 10 and 60 minute periods.
    """
    atexit.register(exiting, logger, TEMP_DATA_PATH)

    if period == 10:
        period_type = PeriodType.MIN
    elif period == 60:
        period_type = PeriodType.HOUR
    else:
        logger.error('Invalid period type.')
        sys.exit(1)

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(TEMP_DATA_PATH):
        os.makedirs(TEMP_DATA_PATH)

    meta_data = download_meta_data(TEMP_DATA_PATH, period_type)
    meta_data.to_csv(os.path.join(path, OMSZData.meta_data_file), sep=',')

    omsz_historical_url = os.path.join(OMSZData.base_url, period_type.value, OMSZData.historical_url)
    omsz_recent_url = os.path.join(OMSZData.base_url, period_type.value, OMSZData.recent_url)

    historical_download_links = get_down_links(omsz_historical_url, True)
    recent_download_links = get_down_links(omsz_recent_url, False)

    station_id_regex = re.compile(r'.*_(\d{5})_.*')
    historical_download_links = filter(lambda x: station_id_regex.match(x) is not None, historical_download_links)
    historical_download_dict = {int(station_id_regex.match(link).group(1)): link for link in historical_download_links}
    recent_download_links = filter(lambda x: station_id_regex.match(x) is not None, recent_download_links)
    recent_download_dict = {int(station_id_regex.match(link).group(1)): link for link in recent_download_links}

    station_numbers = list(meta_data.index.values)

    download_stations('historical', historical_download_dict)
    download_stations('recent', recent_download_dict)

    historical_csvs = unpack_stations('historical', station_numbers)
    recent_csvs = unpack_stations('recent', station_numbers)

    concat_csvs(path, historical_csvs, recent_csvs, from_time, to_time)

    atexit.unregister(exiting)


def download_meta_data(path: str, period_type: PeriodType) -> pd.DataFrame | None:
    omsz_meta_url = os.path.join(OMSZData.base_url, period_type.value, OMSZData.meta_data_file)
    omsz_meta_page = req_get(omsz_meta_url)
    omsz_meta_save_path = os.path.join(path, OMSZData.meta_data_file)
    if omsz_meta_page.status_code == 200:
        with open(omsz_meta_save_path, 'wb') as f:
            f.write(omsz_meta_page.content)
            logger.info(f'Downloaded metadata: {omsz_meta_save_path}')

        data = load_meta(omsz_meta_save_path)
        return data

    logger.error(f'Error {omsz_meta_page.status_code} for {omsz_meta_url}\nExiting...')
    sys.exit(1)


def load_meta(path: str) -> pd.DataFrame:
    meta = pd.read_csv(path, sep=';', skipinitialspace=True, na_values='EOR')
    meta.columns = meta.columns.str.strip()

    meta.index = meta['StationNumber']
    meta.drop('StationNumber', axis=1, inplace=True)
    meta.dropna(how='all', axis=1, inplace=True)
    meta = meta[~meta.index.duplicated(keep='last')]

    meta['StartDate'] = pd.to_datetime(meta['StartDate'], format='%Y%m%d')
    meta['EndDate'] = pd.to_datetime(meta['EndDate'], format='%Y%m%d')

    os.remove(path)
    return meta


def get_down_links(url: str, historical: bool) -> list[str]:
    page = req_get(url)
    soup = bs4.BeautifulSoup(page.text, 'html.parser')
    file_download = soup.find_all('a')
    zip_regex = re.compile(r'.*\.zip')
    file_download = [link.get('href').strip() for link in file_download if zip_regex.match(link.get('href'))]
    file_download = list(set(file_download))

    if historical:
        last_year = datetime.datetime(year=datetime.datetime.now().year - 1, month=12, day=31).strftime('%Y%m%d')
        currently_active_regex = re.compile(r'.*' + re.escape(last_year) + r'.*')
        file_download = [f"{url}{file}" for file in file_download if currently_active_regex.match(file)]
        return file_download

    return [f"{url}{file}" for file in file_download]


def download_stations(group: str, download_dict: dict[int, str]):
    for station_number, link in download_dict.items():
        temp_path = os.path.join(TEMP_DATA_PATH, f'{group}_{station_number}.zip')
        download(link, temp_path)


def download(url: str, path: str) -> bool:
    response = req_get(url, timeout=60)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
            logger.info(f'Downloaded {path}')

        return True

    logger.warning(f'Error {response.status_code} for {url}')
    return False


def unpack_stations(group: str, station_numbers: list[int]):
    csv_paths = {}

    for station_number in station_numbers:
        temp_path = os.path.join(TEMP_DATA_PATH, f'{group}_{station_number}.zip')
        try:
            with ZipFile(temp_path, 'r') as zip_obj:
                unzipped_path = os.path.join(TEMP_DATA_PATH, f'{group}_{station_number}')
                zip_obj.extractall(unzipped_path)

            csv_paths[station_number] = os.path.join(unzipped_path, os.listdir(unzipped_path)[0])
            logger.info(f'Unpacked {csv_paths[station_number]}')
        except Exception as e:
            logger.error(f'Exception {e} for {temp_path}')
            continue

    return csv_paths


def concat_csvs(save_path: str,
                historical_csvs: dict[int, str],
                recent_csvs: dict[int, str],
                start_date: str,
                end_date: str):
    station_numbers = list(set(recent_csvs.keys()).intersection(set(historical_csvs.keys())))
    for station_number in station_numbers:
        historical_df = format_csv(historical_csvs[station_number], start_date, end_date)
        recent_df = format_csv(recent_csvs[station_number], start_date, end_date)
        if historical_df is None or recent_df is None:
            logger.warning(f'Throwing away: {station_number}, ')
            continue

        station_data = pd.concat([historical_df, recent_df])
        csv_name = f'omsz_{station_number}.csv'
        station_data.to_csv(os.path.join(save_path, csv_name), sep=',')
        logger.info(f"Extracted and formatted: {csv_name}")


def format_csv(file_path: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    try:
        df: pd.DataFrame = pd.read_csv(file_path,
                                       skiprows=4,  # skip metadata of csv
                                       sep=';',
                                       skipinitialspace=True,
                                       na_values=['EOR', -999],  # End Of Record is irrelevant, -999 means missing value
                                       dtype=omsz_csv_type_dict
                                       )
        df.columns = df.columns.str.strip()

        df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d%H%M', utc=False)
        df.index = df['Time']
        df.drop('Time', axis=1, inplace=True)

        df.dropna(how='all', axis=1, inplace=True)
        # 'suv' column doesn't exist in some instances
        df.drop(['StationNumber', 't', 'tn', 'tx', 'v', 'fs', 'fsd', 'fx', 'fxd', 'fxdat', 'fd', 'et5', 'et10', 'et20',
                 'et50', 'et100', 'tsn', 'suv'], axis=1, inplace=True, errors='ignore')
        # TODO: check which one of these are important for you

        df = df[start_date:end_date]
        return df
    except Exception as e:
        logger.error(f'Exception {e} for {file_path}')
        return None


def main():
    download_omsz_data('../data/omsz_data/test_min',
                       '2021-01-01 00:00:00',
                       '2024-01-02 00:00:00',
                       PeriodType.MIN)
    download_omsz_data('../data/omsz_data/test_hour',
                       '2021-01-01 00:00:00',
                       '2024-01-02 00:00:00',
                       PeriodType.HOUR)


if __name__ == '__main__':
    main()
