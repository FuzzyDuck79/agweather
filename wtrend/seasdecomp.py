#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seasdecomp

Classical seasonal decomposition using moving averages.
Mutahar Chalmers, 2019
"""

import pandas as pd


def decompose(x, freq='d', method='additive'):
    """
    Classical seasonal time series decomposition.

    Parameters
    ----------
        x : pandas Series
            If Series is at daily resolution, it is assumed to have a
            pandas DatetimeIndex. If Series is at monthly resolution,
            it is assumed to have a (year, month) MultiIndex.
        freq : string, optional. Default value is 'd'.
            Must be one of ['d','day','daily','m','month','monthly'].
            For daily data, a periodicity of 365 days is inferred.
            For monthly data, a periodicity of 12 months is inferred.
        method : string, optional. Default value is 'additive'.
            Must be one of ['additive','multiplicative'].

    Returns
    -------
        sd : dict
           Dictionary containing trend, seasonal and residual components.
    """

    # Identify seasonal contribution
    if freq.lower() in ['d','day','daily']:
        # Identify and remove trend
        trend = x.rolling(365, center=True).mean()
        if method.lower() == 'multiplicative':
            x_notrend = x / trend
        else:
            x_notrend = x - trend

        # Assume that the time series is indexed by pandas DatetimeIndex
        seas = x_notrend.groupby(x_notrend.index.dayofyear).transform('mean')

    elif freq.lower() in ['m','month','monthly']:
        # Identify and remove trend
        trend = x.rolling(12, center=True).mean()
        if method.lower() == 'multiplicative':
            x_notrend = x / trend
        else:
            x_notrend = x - trend

        # Assume that the time series is indexed by (year, month) MultiIndex
        seas = x_notrend.groupby(level='month').transform('mean')

    # Calculate residuals - common code for daily and monthly
    if method == 'multiplicative':
        residual = x_notrend / seas
    else:
        residual = x_notrend - seas

    sd = {'trend': trend, 'seasonal': seas, 'residual': residual}
    return sd

