#!/usr/bin/python3
# coding: utf-8

"""
Process raw NOAA climate index data into convenient format.
Mutahar Chalmers, 2018.
"""


from sys import argv
from os import listdir
from os.path import join
import numpy as np
import pandas as pd

def procf(path, fname):
    """Load and process individual file."""

    # Identify name of index from filename
    name = fname.split('.')[0]

    with open(join(path, fname), 'r') as f:
        raw = f.readlines()

    # Id  start and end years from first line, use to get data and missing flag
    yr_start, yr_end = [int(x) for x in raw[0].strip().split()]
    endline_ix = yr_end - yr_start + 2
    missing = float(raw[endline_ix].strip())
    rawdata = [line.strip().split() for line in raw[1:endline_ix]]

    # Replace various different missing flags with one pandas will recognise
    rawdata = [[elem if float(elem)!=missing else '' for elem in row] for row in rawdata]

    # Convert raw data to DataFrame and reshape in convenient format
    cols = ['year'] + [i for i in range(1, 13, 1)]
    df_raw = pd.DataFrame(rawdata, columns=cols)
    series = df_raw.set_index('year').stack()
    series.name = name
    series.index.names = ['year', 'month']
    series[series==missing] = np.nan

    return series


if __name__ == '__main__':
    path = argv[1]
    out = pd.DataFrame([procf(path, f) for f in listdir(path) if 'txt' in f]).T
    out.sort_index(level=[0,1]).to_csv(join(path, 'indices.csv'))
    print('Done.')

