import sys
import os
import time
from requests import get as req_get
import pandas as pd
import atexit
from utils import exit_handling
from datetime import datetime
import warnings
from shutil import rmtree


def exiting():
    if exit_handling.exit_hooks.exit_code != 0 or exit_handling.exit_hooks.exception is not None:
        print("WARNING: download may be corrupted, because program was interrupted or an exception occurred!",
              file=sys.stderr)

    if not os.path.exists('temp_data'):
        return
    print('Cleaning up temporary files...')
    rmtree('temp_data')


def format_mavir(dataframe: pd.DataFrame):
    dataframe.columns = dataframe.columns.str.strip()
    dataframe.index = pd.to_datetime(dataframe['Időpont'], utc=True)
    dataframe.drop(['Időpont'], axis=1, inplace=True)
    dataframe.dropna(axis=0, inplace=True)
    dataframe.drop(['Szélerőművek becsült termelése (aktuális)',
                    'Szélerőművek becsült termelése (dayahead)',
                    'Szélerőművek becsült termelése (intraday)',
                    'Szélerőművek tény - nettó kereskedelmi elszámolási',
                    'Szélerőművek tény - nettó üzemirányítási',
                    'Szélerőművek tény - bruttó üzemirányítási 1p'],
                   inplace=True, axis=1)
    return dataframe


def download_from_to(name: str, from_time: int, to_time: int):
    """Time should be given as ms"""
    if not os.path.exists('temp_data'):
        os.makedirs('temp_data')
    if not os.path.exists('mavir_data'):
        os.makedirs('mavir_data')

    url = (f"https://www.mavir.hu/rtdwweb/webuser/chart/11840/export"
           f"?exportType=xlsx"
           f"&fromTime={from_time}"
           f"&toTime={to_time}"
           f"&periodType=min"
           f"&period=10")

    temp_path = f"temp_data/{name}.xlsx"
    response = req_get(url, timeout=240)

    if response.status_code == 200:
        with open(temp_path, 'wb') as f:
            f.write(response.content)
            print(f"Downloaded {temp_path}")
    else:
        print(f"Error {response.status_code} for request", file=sys.stderr)
        print(f"Error message: {response.content.decode()}", file=sys.stderr)
        return 1

    # suppress default style warning
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pd.read_excel(temp_path, skiprows=0,
                           parse_dates=True, engine='openpyxl')

    return format_mavir(df)


def main():
    from_time = pd.to_datetime(
        '2015-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
    # we need to subtract 2 hours from the date because of the timezone
    from_time = from_time - pd.Timedelta(hours=2)
    from_in_ms = int(from_time.value / 1e6)

    to_time = datetime.now()
    # request goes till end of current day
    to_time = to_time.replace(minute=0, second=0, microsecond=0)
    to_time = pd.to_datetime(
            to_time, format='%Y-%m-%d %H:%M:%S') - pd.Timedelta(hours=2)

    # I'll split the request into n parts, because the request fails if it's too long (more than 60_000 lines)
    n = 10
    fraction_time = pd.to_timedelta((to_time - from_time) / n)
    fraction_in_ms = int(fraction_time.value / 1e6)

    dfs = []
    # Sleep is needed to not spam the api
    sleep_time = 5
    for i in range(0, n):
        print(f"{i}. fragment")
        df = download_from_to(f'mavir_{i}', from_in_ms+i*fraction_in_ms, from_in_ms+(i+1)*fraction_in_ms)
        if isinstance(df, pd.DataFrame):
            dfs.append(df)

        time.sleep(sleep_time)

    pd.concat(dfs).to_csv('../data/mavir_data/mavir.csv', sep=';')

    if not os.path.exists('mavir_data'):
        os.makedirs('mavir_data')


def load_mavir_data():
    df = pd.read_csv(filepath_or_buffer='./mavir_data/mavir.csv',
                     delimiter=';')

    df['Time'] = pd.to_datetime(df['Time'])

    return df


# only run if executed
if __name__ == '__main__':
    atexit.register(exiting)
    main()
