#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qmap

Quantile mapping for gridded weather data using theoretical and
empirical CDFs and QFs.

Mutahar Chalmers, 2019
"""

#TODO
# - Extrapolation for out of bounds quantiles?
# - Option to detrend prior to quantile-mapping?

import os
import sys
from functools import partial
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.interpolate as si
import matplotlib.pyplot as plt


def quantile_map(source, target, to_qmap=None, dist=None, wetday=None):
    """
    Function to do quantile mapping for location-level daily weather data.
    If dist is not specified, empirical CDFs and quantile functions are used.
    Precipitation is assumed to be in mm (for dry day threshold).

    Parameters
    ----------
        source : pandas Series indexed by (year, month, day) MultiIndex,
            or by a pandas DatetimeIndex.
            Data to be used as the source distribution for the mapping. If no
            value is entered for to_qmap, this data also gets transformed.
        target : pandas Series indexed by (year, month, day) MultiIndex,
            or by a pandas DatetimeIndex.
            Data to be used as the target distribution for the mapping.
        to_qmap : pandas Series indexed by (year, month, day) MultiIndex,
            or by a pandas DatetimeIndex. [OPTIONAL]
            Data which gets quantile mapped. If None, then source used.
        dist : string [OPTIONAL]
            Theoretical distribution to be used for distribution-based quantile
            mapping. Must be one of ['normal','gamma','berngamma']
        wetday : Boolean or float [OPTIONAL]
            For precipitation, flag whether to equalise the proportion of wet
            days between the source and target data. If True, the empirical
            probablilty of non-zero target values is calculated, and the
            corresponding precipitation quantile is identified. Source values
            below this value are set to zero. If float, then the above is done,
            in addtion to the float being used to define wet/dry threshold.

    Returns
    -------
        q : pandas Series
            Quantile-mapped values
    """

    # Convert everything to float64 - avoids weird scipy interpolation bug
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    if to_qmap is not None:
        to_qmap = to_qmap.astype(np.float64)

    # Check wet day - assume precipitation if wet day is not None
    if wetday is not None:
        if wetday is True:
            thresh = 1
        else:
            thresh = wetday

        # Apply threshold
        source = source.where(source>=thresh, 0)
        target = target.where(target>=thresh, 0)
        if to_qmap is not None:
            to_qmap = target.where(to_qmap>=thresh, 0)

    # Check if theoretical distribution to be used
    if dist is not None:
        if dist in ['normal','gamma','berngamma']:
            args_source = fit(source, dist, thresh)
            args_target = fit(target, dist, thresh)

            # Handle case where p0_source > p0_target
            if dist == 'berngamma' and args_source['p0'] > args_target['p0']:
                args_source['p0'] = args_target['p0']

            cdf = tcdf(dist, **args_source)
            qf = tqf(dist, **args_target)
        else:
            print('dist must be in [''normal'', ''gamma'', ''berngamma'']')
            return None
    else:
        qf = eqf(target)
        cdf = ecdf(source)

    if to_qmap is None:
        return pd.Series(qf(cdf(source)), index=source.index)
    else:
        return pd.Series(qf(cdf(to_qmap)), index=to_qmap.index)


def ecdf(x, plot=False, **plot_kwargs):
    """
    Generate Python function to evaluate interpolated empirical CDF of data.

    Parameters
    ----------
        x : numpy array
        plot : Boolean
            Flag determining whether a plot is produced or not.

    Returns
    -------
        ecdf : function
           Calculates empirical cumulative probabilites from data
    """

    xx = np.sort(x).astype(np.float64)
    qq = np.arange(1, xx.size+1)/xx.size

    # Define 'naive' empirical CDF function, which needs to be bounded
    ecdf = si.interp1d(xx, qq, bounds_error=False, fill_value=(0, 1))

    if plot:
        plt.plot(xx, ecdf(xx), **plot_kwargs)
    return ecdf


def eqf(x, plot=False, **plot_kwargs):
    """
    Generate Python function to evaluate interpolated empirical QFs of data.

    Parameters
    ----------
        x : numpy array
        plot : Boolean
            Flag determining whether a plot is produced or not.

    Returns
    -------
        eqf : function
            Calculates quantiles from cumulative probabilities
    """

    xx = np.sort(x).astype(np.float64)
    qq = np.arange(1, xx.size+1)/xx.size

    # Define 'naive' empirical CDF function, which needs to be bounded
    eqf = si.interp1d(qq, xx, bounds_error=False, fill_value=(xx.min(), xx.max()))

    if plot:
        plt.plot(qq, eqf(qq), **plot_kwargs)
    return eqf


def fit(data, dist, thresh=1):
    """
    Fit theoretical distributions to data.

    Parameters
    ----------
        data : numpy array
            Raw data for fitting a distribution
        dist : string
            Distribution to be used, one of ['normal', 'gamma', 'berngamma']
        thresh : float
            Threshold applied to Bernoulli-gamma data to identify occurrence or
            non-occurrence. Default 1 (used for precipitation)

    Returns
    -------
        args : tuple of floats
            Appropriate location, scale and shape parameters
    """

    if dist == 'normal':
        args = dict(zip(('loc','scale'), st.norm.fit(data)))
    elif dist == 'gamma':
        args = dict(zip(('a','loc','scale'), st.gamma.fit(data, floc=0)))
    elif dist == 'berngamma':
        # Estimate probability of dry day
        p0 = (data<=thresh).mean()

        # Fit gamma distribution to non-zero values
        args = dict(
                    zip(('a','loc','scale','p0'),
                        st.gamma.fit(data[data>0], floc=0)+(p0,))
                    )
    else:
        print('dist must be in [''normal'', ''gamma'', ''berngamma'']')
        return None
    return args


def tcdf(dist, a=None, loc=None, scale=None, p0=None):
    """
    Calculate cumulative probabilities for data assuming a theoretical
    fitted distribution.

    Parameters
    ----------
        data : numpy array
        dist : string in ['normal', 'gamma', 'berngamma']
            Distribution to be used
        a : float
            Shape parameter (if applicable)
        loc : float
            Location parameter (if applicable)
        scale : float
            Scale parameter (if applicable)
        p0 : float
            Probability of dry day in source Bernoulli distribution

    Returns
    -------
        tcdf : function
            Calculates theoretical cumulative probabilities
    """

    if dist == 'normal':
        return partial(st.norm.cdf, loc=loc, scale=scale)
    elif dist == 'gamma':
        return partial(st.gamma.cdf, a=a, loc=loc, scale=scale)
    elif dist == 'berngamma':
        def tcdf(data):
            return p0 + (1-p0)*st.gamma.cdf(data, a=a, loc=loc, scale=scale)
        return tcdf
    else:
        print('dist must be in [''normal'', ''gamma'', ''berngamma'']')
        return None


def tqf(dist, a=None, loc=None, scale=None, p0=None):
    """
    Calculate quantiles for an array of cumulative probabilities assuming a
    theoretical distribution.

    Parameters
    ----------
        data : numpy array
        dist : string in ['normal', 'gamma', 'berngamma']
            Distribution to be used
        a : float
            Shape parameter (if applicable)
        loc : float
            Location parameter (if applicable)
        scale : float
            Scale parameter (if applicable)
        p0 : float
            Probability of dry day in target Bernoulli distribution

    Returns
    -------
        tqf : function
            Calculates theoretical quantiles from cumulative probabilities
    """

    if dist == 'normal':
        return partial(st.norm.ppf, loc=loc, scale=scale)
    elif dist == 'gamma':
        return partial(st.gamma.ppf, a=a, loc=loc, scale=scale)
    elif dist == 'berngamma':
        def tqf(cumprob):
            q = st.gamma.ppf(np.clip((cumprob-p0)/(1-p0), a_min=0, a_max=None),
                             a=a, loc=loc, scale=scale)
            return q
        return tqf
    else:
        print('dist must be in [''normal'', ''gamma'', ''berngamma'']')
        return None

