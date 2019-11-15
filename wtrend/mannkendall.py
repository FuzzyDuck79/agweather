#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mannkendall

Mann-Kendall test for monotonic trends.
Mutahar Chalmers, 2019
"""

import scipy.stats as st


def mktest(x):
    """
    Mann-Kendall test, a non-parametric test for a monotonic trend.

    Parameters
    ----------
        x : 1D numpy array or pandas Series.

    Returns
    -------
        mk : dict
           Dictionary containing 'corr' and 'pval' giving the Kendall's tau
           rank correlation and the two-sided p-value for the hypothesis
           test whose null hypothesis is no trend, i.e. tau = 0.
    """

    # Trend hypothesis testing
    mk = st.kendalltau(range(x.shape[0]), x)
    return {'corr': mk.correlation, 'pval': mk.pvalue}

