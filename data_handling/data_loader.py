import pandas as pd
import os


def load_data(path: str, num_rows: int = None):
    path = os.path.abspath(path)
    df = pd.read_csv(filepath_or_buffer=path, sep=',', decimal='.', date_format='%Y-%m-%d %H:%M:%S', index_col='Time',
                     nrows=num_rows)

    return df
