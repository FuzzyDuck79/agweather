#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
windex

Calculate various agri-weather indices.
Mutahar Chalmers, 2018
"""

import sys
import numpy as np
import pandas as pd

from .speipy import spi, spei


def runlen(x):
    """
    Calculate spell lengths for some true or false criterion.
    Spells are encoded as a boolean Series.
    """

    fst, = np.where(x.astype(int).diff().fillna(0)==1)
    lst, = np.where(x.astype(int).diff().fillna(0)==-1)

    # Only run started before and ends after this period
    if fst.size == 0 and lst.size == 0:
        return pd.Series(x.sum(), index=x.index[0:1])

    # Only run started before this period
    if fst.size == 0:
        fst = np.r_[0, fst]
    # Only run ends after this period
    if lst.size == 0:
        lst = np.r_[lst, len(x)]
    # A run ends after this period
    if lst[0] < fst[0]:
        fst = np.r_[0, fst]
    # A run starts after this period
    if fst[-1] > lst[-1]:
        lst = np.r_[lst, len(x)]
    return pd.Series(lst-fst, index=x.index[fst])


def extrema(x, wlen=1):
    """
    Identify (local) maxima and minima of a standardised Series x.
    """

    # First difference
    dx = x.diff()

    # First difference of sign of first difference, shifted
    extrema = (dx/dx.abs()).diff().shift(-1)

    # Assume that maxima are positive and minima are negative
    maxima = x[(extrema==-2)&(x>0)]
    minima = x[(extrema==2)&(x<0)]

    # Local maxima and minima within a window of width wlen
    lmaxima = x[x.rolling(wlen, center=True).max()==maxima[x.index]]
    lminima = x[x.rolling(wlen, center=True).min()==minima[x.index]]
    return lmaxima, lminima


def index_pre(pre, eto=None, wetdry_thresh=1):
    """
    Calculate weather indices derived from precipitation.
    """

    # Concise variable for repeated year-month indexing
    ym = ['year','month']

    # Define groupby index for use with a DateTimeIndex or MultiIndex
    if isinstance(pre.index, pd.DatetimeIndex):
        gbix = [pre.index.year, pre.index.month]
    else:
        # Otherwise assumed to be (year, month, day, [doy]) MultiIndex
        gbix = ym

    # Monthly sum 
    pre_month = pre.groupby(gbix).sum().rename_axis(ym)

    # Monthly rainy days
    rdays = (pre>wetdry_thresh).groupby(gbix).sum().rename_axis(ym)

    # SPI for 'standard' durations
    spi1 = pd.Series(spi(pre_month, N=1), index=pre_month.index)
    spi3 = pd.Series(spi(pre_month, N=3), index=pre_month.index)
    spi6 = pd.Series(spi(pre_month, N=6), index=pre_month.index)

    # Calculate SPEI if eto available
    if eto is not None:
        eto_month = eto.groupby(gbix).sum().rename_axis(ym)
        cwb = pre_month - eto_month

        spei1 = pd.Series(spei(cwb_month, N=1), index=cwb_month.index)
        spei3 = pd.Series(spei(cwb_month, N=3), index=cwb_month.index)
        spei6 = pd.Series(spiei(cwb_month, N=6), index=cwb_month.index)

    # 5th and 95th daily percentiles by month
    r05 = pre.groupby(pre.index.month).quantile(0.05).rename_axis('month')
    r95 = pre.groupby(pre.index.month).quantile(0.95).rename_axis('month')

    # Boolean Series flagging whether <=r05 or >r95 criteria have been satisfied
    r05_bool = pd.concat([(pre_ym<=r05[month])
                          for (year, month), pre_ym in pre.groupby(gbix)])
    r95_bool = pd.concat([(pre_ym>r95[month])
                          for (year, month), pre_ym in pre.groupby(gbix)])

    # Precipitation deficit
    r05_count = r05_bool.groupby(gbix).sum().rename_axis(ym)
    r05_sum = pre.where(r05_bool).groupby(gbix).sum().rename_axis(ym)
    r05_spell = r05_bool.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)

    # Precipitation excess
    r95_count = r95_bool.groupby(gbix).sum().rename_axis(ym)
    r95_sum = pre.where(r95_bool).groupby(gbix).sum().rename_axis(ym)
    r95_spell = r95_bool.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)

    # 1-, 2- and 3-day maximum
    pre24 = pre.groupby(gbix).max().rename_axis(ym)
    pre48 = pre.rolling(2).sum().groupby(gbix).max().rename_axis(ym)
    pre72 = pre.rolling(3).sum().groupby(gbix).max().rename_axis(ym)

    # Combine all indices indexed by (year, month) MultiIndex
    out = pd.DataFrame({
                        'rsum': pre_month,
                        'rday': rdays.astype(int),
                        'spi1': spi1,
                        'spi3': spi3,
                        'spi6': spi6,
                        'r05_count': r05_count.astype(int),
                        'r05_sum': r05_sum,
                        'r05_spell': r05_spell.astype(int),
                        'r95_count': r95_count.astype(int),
                        'r95_sum': r95_sum,
                        'r95_spell': r95_spell.astype(int),
                        'pre24': pre24,
                        'pre48': pre48,
                        'pre72': pre72
                        })

    # Add SPEI if applicable
    if eto is not None:
        out['spei1'] = spei1
        out['spei3'] = spei3
        out['spei6'] = spei6

    return out.rename_axis('v', axis=1).stack()


def index_tmp(tmax, tmin, tmax_gt=[35,40], tmin_gt=[15,20], tmin_lt=[0]):
    """
    Calculate weather indices derived from tmax and tmin.
    """

    # Concise variable for repeated year-month indexing
    ym = ['year','month']

    # Check that indices of tmax and tmin are identical
    if (tmax.index != tmin.index).any():
        print('Indices of tmax and tmin must be the same!')
        return None

    # Define groupby index for use with a DateTimeIndex or MultiIndex
    if isinstance(tmax.index, pd.DatetimeIndex):
        gbix = [tmax.index.year, tmax.index.month]
    else:
        # Otherwise assumed to be (year, month, day, [doy]) MultiIndex
        gbix = ym

    # Monthly average 
    tmax_month = tmax.groupby(gbix).mean().rename_axis(ym)
    tmin_month = tmin.groupby(gbix).mean().rename_axis(ym)

    # Diurnal temperature range
    dtr_month = tmax_month - tmin_month

    # 5th and 95th daily percentiles by month for both tmax and tmin
    tmax05 = tmax.groupby(tmax.index.month).quantile(0.05).rename_axis('month')
    tmax95 = tmax.groupby(tmax.index.month).quantile(0.95).rename_axis('month')
    tmin05 = tmin.groupby(tmin.index.month).quantile(0.05).rename_axis('month')
    tmin95 = tmin.groupby(tmin.index.month).quantile(0.95).rename_axis('month')

    # Boolean Series flagging whether <=t05 or >t95 criteria have been satisfied
    tmax05_bool = pd.concat([(tmax_ym<=tmax05[month])
                             for (year, month), tmax_ym in tmax.groupby(gbix)])
    tmax95_bool = pd.concat([(tmax_ym>tmax95[month])
                             for (year, month), tmax_ym in tmax.groupby(gbix)])
    tmin05_bool = pd.concat([(tmin_ym<=tmin05[month])
                             for (year, month), tmin_ym in tmin.groupby(gbix)])
    tmin95_bool = pd.concat([(tmin_ym>tmin95[month])
                             for (year, month), tmin_ym in tmin.groupby(gbix)])

    # Diurnal heat and cold (tmax)
    tmax05_count = tmax05_bool.groupby(gbix).sum().rename_axis(ym)
    tmax05_sum = tmax.where(tmax05_bool).groupby(gbix).sum().rename_axis(ym)
    tmax05_spell = tmax05_bool.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
    tmax95_count = tmax95_bool.groupby(gbix).sum().rename_axis(ym)
    tmax95_sum = tmax.where(tmax95_bool).groupby(gbix).sum().rename_axis(ym)
    tmax95_spell = tmax95_bool.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)

    # Diurnal heat (tmax); explicit HDD thresholds
    hdd_tmax = {}
    for thresh in tmax_gt:
        tmax_gt_bool = pd.concat([(tmax_ym>thresh) for _, tmax_ym in tmax.groupby(gbix)])
        tmax_gt_count = tmax_gt_bool.groupby(gbix).sum().rename_axis(ym)
        hdd_tmax[f'tmax_hdd_count_{thresh}'] = tmax_gt_count
        tmax_gt_sum = tmax.where(tmax_gt_bool).groupby(gbix).sum().rename_axis(ym)
        hdd_tmax[f'tmax_hdd_sum_{thresh}'] = tmax_gt_sum
        tmax_gt_spell = tmax_gt_bool.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
        hdd_tmax[f'tmax_hdd_spell_{thresh}'] = tmax_gt_spell
    hdd_tmax = pd.DataFrame(hdd_tmax)

    # Nocturnal heat (tmin); explicit HDD thresholds
    hdd_tmin = {}
    for thresh in tmin_gt:
        tmin_gt_bool = pd.concat([(tmin_ym>thresh) for _, tmin_ym in tmin.groupby(gbix)])
        tmin_gt_count = tmin_gt_bool.groupby(gbix).sum().rename_axis(ym)
        hdd_tmin[f'tmin_hdd_count_{thresh}'] = tmin_gt_count
        tmin_gt_sum = tmin.where(tmin_gt_bool).groupby(gbix).sum().rename_axis(ym)
        hdd_tmin[f'tmin_hdd_sum_{thresh}'] = tmin_gt_sum
        tmin_gt_spell = tmin_gt_bool.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
        hdd_tmin[f'tmin_hdd_spell_{thresh}'] = tmin_gt_spell
    hdd_tmin = pd.DataFrame(hdd_tmin)

    # Nocturnal cold (tmin); explicit CDD thresholds
    cdd_tmin = {}
    for thresh in tmin_lt:
        tmin_lt_bool = pd.concat([(tmin_ym<thresh) for _, tmin_ym in tmin.groupby(gbix)])
        tmin_lt_count = tmin_lt_bool.groupby(gbix).sum().rename_axis(ym)
        cdd_tmin[f'tmin_cdd_count_{thresh}'] = tmin_lt_count
        tmin_lt_sum = tmin.where(tmin_lt_bool).groupby(gbix).sum().rename_axis(ym)
        cdd_tmin[f'tmin_cdd_sum_{thresh}'] = tmin_lt_sum
        tmin_lt_spell = tmin_lt_bool.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
        cdd_tmin[f'tmin_cdd_spell_{thresh}'] = tmin_lt_spell
    cdd_tmin = pd.DataFrame(cdd_tmin)

    # Nocturnal heat and cold (tmin)
    tmin05_count = tmin05_bool.groupby(gbix).sum().rename_axis(ym)
    tmin05_sum = tmin.where(tmin05_bool).groupby(gbix).sum().rename_axis(ym)
    tmin05_spell = tmin05_bool.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
    tmin95_count = tmin95_bool.groupby(gbix).sum().rename_axis(ym)
    tmin95_sum = tmin.where(tmin95_bool).groupby(gbix).sum().rename_axis(ym)
    tmin95_spell = tmin95_bool.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)

    # 1-, 2- and 3-day maxima and minima
    tmax24 = tmax.groupby(gbix).max().rename_axis(ym)
    tmax48 = tmax.rolling(2).mean().groupby(gbix).max().rename_axis(ym)
    tmax72 = tmax.rolling(3).mean().groupby(gbix).max().rename_axis(ym)
    tmin24 = tmin.groupby(gbix).min().rename_axis(ym)
    tmin48 = tmin.rolling(2).mean().groupby(gbix).min().rename_axis(ym)
    tmin72 = tmin.rolling(3).mean().groupby(gbix).min().rename_axis(ym)

    # Combine all indices indexed by (year, month) MultiIndex
    out = pd.DataFrame({
                        'tmax': tmax_month,
                        'tmin': tmin_month,
                        'dtr': dtr_month,
                        'tmax05_count': tmax05_count.astype(int),
                        'tmax05_sum': tmax05_sum,
                        'tmax05_spell': tmax05_spell.astype(int),
                        'tmax95_count': tmax95_count.astype(int),
                        'tmax95_sum': tmax95_sum,
                        'tmax95_spell': tmax95_spell.astype(int),
                        'tmin05_count': tmin05_count.astype(int),
                        'tmin05_sum': tmin05_sum,
                        'tmin05_spell': tmin05_spell.astype(int),
                        'tmin95_count': tmin95_count.astype(int),
                        'tmin95_sum': tmin95_sum,
                        'tmin95_spell': tmin95_spell.astype(int),
                        'tmax24': tmax24,
                        'tmax48': tmax48,
                        'tmax72': tmax72,
                        'tmin24': tmin24,
                        'tmin48': tmin48,
                        'tmin72': tmin72
                        })
    out = pd.concat([out, hdd_tmax, hdd_tmin, cdd_tmin], axis=1)
    return out.rename_axis('v', axis=1).stack()

