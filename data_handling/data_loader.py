import pandas as pd
import os


def load_mavir_data(path: str):
    path = os.path.abspath(path)
    df = pd.read_csv(filepath_or_buffer=path,
                     delimiter=';')

    df['Time'] = pd.to_datetime(df['Time'])

    return df
