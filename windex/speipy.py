#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spei

Calculate Standardised Precipitation (Evapotranspiration) Index, SP(E)I

Mutahar Chalmers, 2018
"""

import numpy as np
import pandas as pd
from scipy.stats import gamma, genlogistic, norm
from scipy.special import binom, perm


def spi(pre, N=3,  fit='mle'):
    """
    Calculate SPI from monthly precipitation.

    Parameters
    ----------
        pre : pandas Series
            Assumed indexed by (year, month) MultiIndex.
        N : int
            Duration in months.
        fit : str
            Method to use to fit precipitation distribution.
            Dummy variable for spi as will always use scipy's default
            maximum likelihood approach to fit gamma distribution.

    Returns
    -------
        spi : pandas Series
    """

    # Calculate rolling sums and probability of non-zero values
    rollingsum = pre.rolling(N).sum().unstack(level='month')
    p_nonzero = (rollingsum>0).sum()/rollingsum.count()

    # Define SPI DataFrame
    spi = pd.DataFrame().reindex_like(rollingsum)

    # Fit distribution by maximum likelihood and convert to SPI
    for month in range(1, 13):
        # Fit gamma distribution to non-zero values
        params = gamma.fit(rollingsum[month].dropna(), loc=0)
        percentiles = gamma.cdf(rollingsum[month].dropna(), *params)

        # Scale non-zero percentile, add probability mass at zero and invert
        spi_vals = norm.ppf(1-p_nonzero[month] + percentiles*p_nonzero[month])
        spi.loc[~rollingsum[month].isnull(), month] = spi_vals

    return spi.stack().reindex(pre.index)


def spei(cwb, N=3, fit='pwm_ub'):
    """
    Calculate SPEI from monthly climate water balance, pre - eto.
    Reference: http://spei.csic.es/home.html

    Parameters
    ----------
        cwb : pandas Series
            Climatic water balance, i.e. pre - eto.
            Assumed indexed by (year, month) MultiIndex.
        N : int
            Duration in months.
        fit : str
            Method to use to fit cwb distribution.
            Options include ['pwm_ub', 'pwm_pp', 'mle'].
            Default is 'pwm_ub' (unbiased estimator of probability-weighted
            moments).

    Returns
    -------
        spei : pandas Series
    """

    # Calculate rolling sums
    rollingsum = cwb.rolling(N).sum().unstack(level='month')

    # Define SPEI DataFrame
    spei = pd.DataFrame().reindex_like(rollingsum)

    # Fit distribution using L-moments and convert to SPEI
    for month in range(1, 13):
        if fit == 'pwm_ub' or fit == 'pwm_pp':
            params = genlogistic_fit(rollingsum[month].dropna(), method=fit)
            percentiles = genlogistic_cdf(rollingsum[month].dropna(), **params)
    # Fit distribution by maximum likelihood and convert to SPEI
        else:
            params = genlogistic.fit(rollingsum[month].dropna())
            percentiles = genlogistic.cdf(rollingsum[month].dropna(), *params)

        spei_vals = norm.ppf(percentiles)
        spei.loc[~rollingsum[month].isnull(), month] = spei_vals
    return spei.stack().reindex(cwb.index)


def pwm_ub(x, r):
    """
    Unbiased estimator of probability-weighted moments
    Landwehr, 1979a; Hoskings, 1986.
    """

    n = x.size

    # Note that we're calculating a_r, not b_r
    j = np.arange(1, n+1, 1)
    a_r = (x[j-1]*perm(n-j, r)).sum()/perm(n, r+1)

    return a_r


def pwm_pp(x, r):
    """
    Plotting-position estimator of probability-weighted moments
    Landwehr, 1979b; Hoskings, 1986.
    """

    n = x.size

    # Note that we're calculating a_r, not b_r
    j = np.arange(1, n+1, 1)
    a_r = (x[j-1]*(1-(j-0.35)/n)**r).sum()/n

    return a_r


def pwm2lmom(pwms):
    """
    Convert probability-weighted moments to L-moments and ratios.

    Reference: Hosking and Wallis (1995) A comparison of unbiased and
               plotting-position estimators of L-moments.
    """

    # Number of L-moments to calculate
    max_r = len(pwms)

    def pstar(r, k):
        return (-1)**(r-k) * binom(r, k) * binom(r+k, k)

    # L-moments             
    lmoms = {'l'+str(r+1): (-1)**r * sum([pstar(r, k)*pwms[k]
                                          for k in range(0, r+1, 1)])
             for r in range(max_r)}

    # L-moment ratios
    ratios = {'t'+str(r): lmoms['l'+str(r)]/lmoms['l2']
              for r in range(2, max_r, 1)}

    return lmoms, ratios


def genlogistic_fit(x, N=4, method='pwm_ub'):
    """
    Fit Generalised Logistic Distribution by L-moments or maximum likelihood.
    Reference:
    https://cran.r-project.org/web/packages/lmomco/lmomco.pdf, pdfglo, page 363
    """

    # Sort x so estimator can be used
    x = np.sort(x)

    if method == 'pwm_ub' or method =='pwm_pp':
        if method == 'pwm_ub':
            lmoms, ratios = pwm2lmom([pwm_ub(x, r) for r in range(0, N, 1)])
        else:
            lmoms, ratios = pwm2lmom([pwm_pp(x, r) for r in range(0, N, 1)])

        # Shape parameter k
        k = -ratios['t3']
        kpi = k*np.pi

        # Scale parameter alpha
        alpha = lmoms['l2'] * np.sin(kpi)/kpi

        # Location parameter xi
        xi = lmoms['l1'] - alpha*(1/k - np.pi/np.sin(kpi))
    elif method == 'mle':
        print('Maximum likelihood not yet implemented')
        return None
    else:
        print('method must be one of pwm_ub, pwm_pp or mle')
        return None

    return {'k': k, 'alpha': alpha, 'xi': xi}


def genlogistic_cdf(x, k, alpha, xi):
    """
    Generalised Logistic Distribution cdf
    Reference:
    https://cran.r-project.org/web/packages/lmomco/lmomco.pdf, cdfglo, page 72
    """

    Y = -k**-1 * np.log(1 - (k*(x - xi))/alpha)
    cdf = 1/(1 + np.exp(-Y))
    return cdf


def genlogistic_pdf(x, k, alpha, xi):
    """
    Generalised Logistic Distribution pdf
    Reference:
    https://cran.r-project.org/web/packages/lmomco/lmomco.pdf, pdfglo, page 363
    """

    Y = -k**-1 * np.log(1 - (k*(x - xi))/alpha)
    pdf = alpha**-1 * np.exp(-(1-k)*Y)/(1 + np.exp(-Y))**2
    return pdf

