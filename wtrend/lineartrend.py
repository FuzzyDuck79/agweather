#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lineartrend

Linear trend analysis of time series data.
Mutahar Chalmers, RMS, 2019
"""

import numpy as np
import scipy.stats as st


def theilsen(x, alpha=0.95):
    """
    Theil-Sen (median slope) trend estimator.

    Parameters
    ----------
        x : 1D numpy array or pandas Series.
        alpha : float, optional
            Confidence level to use to estimate bounds. Default is 0.95.

    Returns
    -------
        tsen : dict
           Dictionary with estimates of the best fit slope and intercept as
           well as upper and lower bounds based on specified confidence level.
    """

    tsen = dict(zip(('slope','intercept','slope_lb','slope_ub'),
                    st.mstats.theilslopes(x)))
    return tsen


def linreg(x):
    """
    Linear regression trend estimator.

    Parameters
    ----------
        x : 1D numpy array or pandas Series.

    Returns
    -------
        lreg : dict
           Dictionary with estimates of the best fit slope and intercept as
           well as upper and lower bounds based on specified confidence level.
    """

    # Trend estimation using Theil-Sen
    lreg = dict(zip(('slope','intercept','r','pval','stderr'),
                    st.linregress(np.arange(x.shape[0]), x)))
    return lreg

