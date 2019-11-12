#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wtrend

Trend analysis of monthly gridded data.
Mutahar Chalmers, RMS, 2019
"""

import sys
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.tsa.stattools import acf


def trend(df_all, full=False):
    """
    Analyse trends in a DataFrame of monthly weather data for a
    single variable. DataFrame must have year, month MultiIndex,
    and qids for columns.
    """

    # Assess trends in seasonal statistics using Mann-Kendall test
    ktau, tsen, acf_pvals = {}, {}, {}

    # Loop over each month/season/year and calculate trends
    for mth in range(1, 13):
        # Select appropriate month's data
        df = df_all.xs(mth, level='month')

        # Trend hypothesis testing
        ktau[mth] = pd.DataFrame({qid: st.kendalltau(range(df.shape[0]), df[qid])
                                  for qid in df.columns}, index=('corr','pvalue'))

        # Trend estimation using Theil-Sen
        tsen[mth] = pd.DataFrame({qid: st.mstats.theilslopes(df[qid])
                                  for qid in df.columns},
                                 index=('slope','intercept','lo95_slope','hi95_slope'))

        # Test for autocorrelation using Ljung-Box test from statsmodels; lag-1 ACF
        acf_pvals[mth] = {qid: acf(df[qid], qstat=True)[2][0] for qid in df.columns}

    ktau = pd.concat(ktau, names=['month','value'])
    tsen = pd.concat(tsen, names=['month','value'])
    acf_pvals = pd.DataFrame(acf_pvals)

    if full:
        return ktau, tsen, acf_pvals
    else:
        return ktau.xs('pvalue', level='value').T, tsen.xs('slope', level='value').T, acf_pvals


def notrend(df):
    """
    Generate a 'no trend' file based on input DataFrame in standard monthly
    format, i.e. (year, month) MultiIndex and qids for columns.
    """

    return pd.DataFrame(np.zeros((df.columns.size, 12)),
                        columns=range(1, 13), index=df.columns)

