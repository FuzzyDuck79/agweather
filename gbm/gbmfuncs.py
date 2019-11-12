#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gbmfuncs

Functions for working with Geometric Brownian Motion (GBM)
Mutahar Chalmers, RMS, 2019

2019-07-10 First version
"""

import numpy as np
import pandas as pd


def gbm_fit(ts, year_from, month_from, year_to, month_to):
    """
    Calculate log-returns from price time series between two points in time,
    and fit GBM model to obtain mean drift and volatility.

    Assumes a pandas Series with MultiIndex comprising date and matyear.
    """

    dates = ts.index.get_level_values('date')
    matyears = ts.index.get_level_values('matyear')

    # Select dates in period of interest and generate mask 
    years_from = matyears + year_from
    dates_from = pd.DatetimeIndex([f'{year}-{month_from}-01' for year in years_from])
    years_to = matyears + year_to
    dates_to = pd.DatetimeIndex([f'{year}-{month_to}-28' for year in years_to])
    mask = (dates >= dates_from) & (dates <= dates_to)

    # Calculate log-returns, drifts and volatilities
    logreturns = np.log(ts[mask]).unstack('matyear').diff().stack()
    drift = logreturns.mean(level='matyear')
    volatility = logreturns.std(level='matyear')

    return {'lr': logreturns, 'drift': drift, 'volatility': volatility}


def gbm_simdaily(drift, volatility, n_days, n_sims, seed=None, **kwargs):
    """
    Simulate Geometric Brownian Motion (GBM) at daily intervals.
    Use closed form solution to SDE. Assumes constant drift and volatility.
    """

    if seed is not None:
        np.random.seed(seed)

    # Add axis 1 to allow correct broadcasting with the Brownian motion array
    t = np.arange(n_days)[:,None]

    # Brownian motion, a.k.a Wiener process (approximate over discrete daily partition)
    b = np.cumsum(np.random.normal(size=(n_days, n_sims)), axis=0)
    b[0] = 0

    return np.exp((drift - 0.5*volatility**2)*t + volatility*b)

