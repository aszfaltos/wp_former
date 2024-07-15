import argparse as arg
from utils import Logger
import os
import sys

from data_handling import download_omsz_data, download_mavir_data, check_datetime


def download_data(path: str, from_time: str, to_time: str, period: int, logger: Logger):
    download_omsz_data(os.path.join(path, 'omsz'), from_time, to_time, period, logger=logger)
    download_mavir_data(os.path.join(path, 'mavir'), from_time, to_time, period, logger=logger)


def main(args: list):
    parser = arg.ArgumentParser(description='Download wind power data from MAVIR and meteorological data from OMSZ.')
    parser.add_argument('-p', '--path', type=str, help='Path to save the data.')
    parser.add_argument('-f', '--from_time', type=str, help='Start time in format "YYYY-MM-DD hh:mm:ss".')
    parser.add_argument('-t', '--to_time', type=str, help='End time in format "YYYY-MM-DD hh:mm:ss".')
    parser.add_argument('-d', '--period', type=int, choices=[10, 60], help='Period of the data in minutes.')
    args = parser.parse_args(args)

    logger = Logger('data_downloader')

    if not check_datetime(args.from_time):
        logger.error('Invalid from_time format.')
        sys.exit(1)
    if not check_datetime(args.to_time):
        logger.error('Invalid to_time format.')
        sys.exit(1)

    download_data(args.path, args.from_time, args.to_time, args.period, logger)


if __name__ == '__main__':
    main(sys.argv[1:])
