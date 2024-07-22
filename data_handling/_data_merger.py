import os
import re

import pandas as pd
from utils import Logger


def merge_data(path: str, name: str, regions: list[str] | None = None, columns: list[str] | None = None, logger=None):
    """
    Merges the OMSZ and mavir data and saves it to a csv file.
    It has to aggregate the OMSZ data since it is taken in multiple stations.

    :param path: The path where the data has been downloaded, the folder has to contain a mavir and an omsz folder.
    At the end the csv will be saved in this folder.
    :param name: The name of the csv file.
    :param regions: The regions to include meteorological data from in the csv, if None all regions will be included.
    :param columns: The columns to include in the csv, if None all columns will be included.
    :param logger: The logger to use.
    """
    if logger is None:
        logger = Logger(__name__)

    mavir_data = load_mavir_data(path, logger)
    omsz_data = load_omsz_data(path, regions, logger)
    merged = pd.merge(mavir_data, omsz_data, left_index=True, right_index=True, how='inner')
    if columns is not None:
        try:
            merged = merged[columns]
        except KeyError as e:
            logger.error(f'Columns were not found in the data. Skipping column selection.\n\tException: {e}')

    merged.to_csv(os.path.join(path, f'{name}.csv'), sep=',', decimal='.', date_format='%Y-%m-%d %H:%M:%S')


def load_mavir_data(path: str, logger=None):
    logger.info('Loading MAVIR data...')
    df = pd.read_csv(os.path.join(path, 'mavir', 'wind.csv'), date_format='%Y-%m-%d %H:%M:%S', index_col='Time')
    logger.info('MAVIR data loaded.')
    return df


def load_omsz_data(path: str, regions: list[str] | None = None, logger=None):
    logger.info('Loading OMSZ data...')

    metadata = pd.read_csv(os.path.join(path, 'omsz', 'station_meta_auto.csv'), index_col='StationNumber')

    stations_to_read = None
    if regions is not None:
        stations_to_read = []
        for index, row in metadata.iterrows():
            if row.get('RegioName').strip() in regions:
                stations_to_read.append(index)

    csvs = os.listdir(os.path.join(path, 'omsz'))
    station_number_regex = re.compile(r'omsz_(\d{5}).csv')

    station_csvs = {int(station_number_regex.match(csv).group(1)): csv
                    for csv in csvs if station_number_regex.match(csv) is not None}

    if stations_to_read is not None:
        station_csvs = {station: csv for station, csv in station_csvs.items() if station in stations_to_read}

    columns = None
    time_column = None
    for station in station_csvs.values():
        df = pd.read_csv(os.path.join(path, 'omsz', station), date_format='%Y-%m-%d %H:%M:%S', index_col='Time')
        if time_column is None:
            time_column = df.index
        if columns is None:
            columns = {column: [] for column in df.columns.to_list() if column != 'Time'}

        for column in df.columns.to_list():
            columns[column].append(df[column])

    combined = pd.DataFrame(index=time_column)
    logger.debug(f'Combining {len(columns)} columns...')
    for key, value in columns.items():
        df = pd.DataFrame(index=time_column)
        df = pd.concat([df, *value], axis=1)
        logger.debug(f'Combining {key}: {df.count(axis=1).max()} stations had data for this column.')
        combined[key] = df.mean(axis=1, skipna=True)

    logger.info('OMSZ data loaded.')
    return combined


if __name__ == '__main__':
    merge_data('../data/hourly', regions=None, columns=None)
