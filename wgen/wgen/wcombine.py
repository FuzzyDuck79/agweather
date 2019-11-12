#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wcombine

Combine all grid-level analytics from historic and stochastic weather output generated
by wgen. Takes advantage of the standard directory structure of wgen output.

Mutahar Chalmers, RMS, 2018-9
"""

import os
import sys
import pandas as pd

sys.path.append('/mnt/gpfs/backup/agri/code/qtree/')
from qtree import cyqtid2ll


def combine(inpath, desc, qsttgs):
    """Load and combine all qid-level results of an analytic and save to NetCDF."""

    # Make combined DataFrame
    combined = pd.concat([pd.read_hdf(inpath+f, desc)
                          for f in os.listdir(inpath) if 'h5' in f], axis=1)

    # Convert qid to lon, lat
    lonlat = [cyqtid2ll(int(qid), **qsttgs, warn=False) for qid in combined.columns]
    combined.columns = pd.MultiIndex.from_tuples(lonlat)
    combined = combined.stack([1,0])

    # Add 'key dimensions'
    if 'quant' in desc:
        combined.index.names = ['month','quantile','variable','lat','lon']
    elif 'xcorr' in desc:
        combined.index.names = ['month','variable1','variable2','lat','lon']
    elif 'clima' in desc or 'ks' in desc:
        combined.index.names = ['month','variable','lat','lon']
    combined.name = 'value'

    combined_xr = combined.to_xarray()
    return combined_xr


if __name__ == '__main__':
    _, inpath, desc, qid_level, qid_offset, outpath = sys.argv

    qsttgs = {'level': qid_level, 'offset': qid_offset}
    out = combine(inpath, desc, qsttgs)
    out.to_netcdf(outpath+'/{0}.nc'.format(desc))

