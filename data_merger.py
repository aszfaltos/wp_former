import argparse as arg
import logging
import os
import sys

from data_handling import merge_data, regions

# TODO: Create custom logger for the whole codebase should send you emails about errors when run in docker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args: list):
    parser = arg.ArgumentParser(description='Merge meteorological data from OMSZ with wind power data from MAVIR.')
    parser.add_argument('-p', '--path', type=str, help='Path to save the data.')
    parser.add_argument('-r', '--regions', type=str, help='Region to include meteorological data from.',
                        choices=regions, nargs='+')
    parser.add_argument('-c', '--columns', type=str, help='Columns to include in the csv.', nargs='+')
    args = parser.parse_args(args)

    merge_data(args.path, args.regions, args.columns)


if __name__ == '__main__':
    main(sys.argv[1:])
