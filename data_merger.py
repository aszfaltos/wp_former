import argparse as arg
import logging
import os
import sys
from utils import Logger

from data_handling import merge_data, regions


def main(args: list):
    logger = Logger('data_merger', logging.INFO)

    parser = arg.ArgumentParser(description='Merge meteorological data from OMSZ with wind power data from MAVIR.')
    parser.add_argument('-p', '--path', type=str, help='Path to save the data.')
    parser.add_argument('-r', '--regions', type=str, help='Region to include meteorological data from.',
                        choices=regions, nargs='+')
    parser.add_argument('-c', '--columns', type=str, help='Columns to include in the csv.', nargs='+')
    args = parser.parse_args(args)

    merge_data(args.path, args.regions, args.columns, logger=logger)


if __name__ == '__main__':
    main(sys.argv[1:])
