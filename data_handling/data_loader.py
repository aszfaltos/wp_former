import pandas as pd
import os


def load_mavir_data(path: str, start_row: int = 0, num_rows: int = None):
    path = os.path.abspath(path)
    df = pd.read_csv(filepath_or_buffer=path, sep=',', decimal='.', date_format='%Y-%m-%d %H:%M:%S', index_col='Time',
                     skiprows=start_row, nrows=num_rows)

    return df
