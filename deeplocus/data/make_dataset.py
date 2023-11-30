# -*- coding: utf-8 -*-
import logging
import argparse
from dotenv import find_dotenv, load_dotenv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeepLocus')
    parser.add_argument('--input', type=str, default='data/raw',
                        help='input directory')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='output directory')
    args = parser.parse_args()
    return args


def main(args):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    # Load environment variables
    load_dotenv(find_dotenv())

    args = parse_args()
    main(args)
