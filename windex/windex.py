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

    Parameters
    ----------
        x : pandas Series
            Assumed to be of dtype Boolean.

    Returns
    -------
        runlen : pandas Series
            Lengths of spells of Trues, defined at start of spell.
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


def index_pre(pre, eto=None, wetdry_thresh=1):
    """
    Calculate weather indices derived from precipitation.

    Parameters
    ----------
        pre : pandas Series
            Precipitation in mm.
        eto : pandas Series, optional
            Reference evapotranspiration in mm.
        wetdry_thresh : float, optional
            Threshold between dry and wet day in mm. Default is 1 mm.

    Returns
    -------
        out : pandas Series
            Collection of different precipitation indices.
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
        cwb_month = pre_month - eto_month

        spei1 = pd.Series(spei(cwb_month, N=1), index=cwb_month.index)
        spei3 = pd.Series(spei(cwb_month, N=3), index=cwb_month.index)
        spei6 = pd.Series(spei(cwb_month, N=6), index=cwb_month.index)

    # 5th and 95th monthly percentiles
    r_p05 = pre.groupby(pre.index.month).transform(lambda x: x.quantile(0.05))
    r_p95 = pre.groupby(pre.index.month).transform(lambda x: x.quantile(0.95))

    # Boolean Series flagging if r<=p05 or r>p95 criteria have been satisfied
    r_lt_p05 = pre <= r_p05
    r_gt_p95 = pre > r_p95

    # Precipitation deficit - percentile thresholds
    r_count_lt_p05 = r_lt_p05.groupby(gbix).sum().rename_axis(ym)
    r_spell_lt_p05 = r_lt_p05.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
    #r_sum_lt_p05 doesn't make sense, so omit

    # Precipitation excess - percentile thresholds
    r_count_gt_p95 = r_gt_p95.groupby(gbix).sum().rename_axis(ym)
    r_spell_gt_p95 = r_gt_p95.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
    r_sum_gt_p95 = (pre-r_p95).clip(0, None).groupby(gbix).sum().rename_axis(ym)

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
                        'r_count_lt_p05': r_count_lt_p05.astype(int),
                        'r_spell_lt_p05': r_spell_lt_p05.astype(int),
                        'r_count_gt_p95': r_count_gt_p95.astype(int),
                        'r_spell_gt_p95': r_spell_gt_p95.astype(int),
                        'r_sum_gt_p95': r_sum_gt_p95,
                        'pre24': pre24,
                        'pre48': pre48,
                        'pre72': pre72
                        })

    # Add ETo and SPEI if available
    if eto is not None:
        out['eto'] = eto_month
        out['spei1'] = spei1
        out['spei3'] = spei3
        out['spei6'] = spei6

    return out.rename_axis('index_name', axis=1).stack().rename('value')


def index_tmp(tmax, tmin, tmax_gt=[35,40], tmin_gt=[15,20], tmin_lt=[0]):
    """
    Calculate weather indices derived from tmax and tmin.

    Parameters
    ----------
        tmax : pandas Series
            Maximum daily temperature in degrees C.
        tmin : pandas Series
            Minimum daily temperature in degrees C.
        tmax_gt : list of floats
            Thresholds for calculating day heat stress degree days,
            in degrees C.
        tmin_gt : list of floats
            Thresholds for calculating night heat stress degree days,
            in degrees C.
        tmin_lt : list of floats
            Thresholds for calculating cold degree days, in degrees C.

    Returns
    -------
        out : pandas Series
            Collection of different temperature indices.
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
    tmax_p05 = tmax.groupby(tmax.index.month).transform(lambda x: x.quantile(0.05))
    tmax_p95 = tmax.groupby(tmax.index.month).transform(lambda x: x.quantile(0.95))
    tmin_p05 = tmin.groupby(tmin.index.month).transform(lambda x: x.quantile(0.05))
    tmin_p95 = tmin.groupby(tmin.index.month).transform(lambda x: x.quantile(0.95))

    # Boolean Series flagging whether <=t05 or >t95 criteria have been satisfied
    tmax_lt_p05 = tmax <= tmax_p05
    tmax_gt_p95 = tmax > tmax_p95
    tmin_lt_p05 = tmin <= tmin_p05
    tmin_gt_p95 = tmin > tmin_p95

    # Diurnal (tmax) heat and cold - percentile thresholds
    tmax_count_gt_p95 = tmax_gt_p95.groupby(gbix).sum().rename_axis(ym)
    tmax_spell_gt_p95 = tmax_gt_p95.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
    tmax_sum_gt_p95 = (tmax-tmax_p95).clip(0, None).groupby(gbix).sum().rename_axis(ym)
    tmax_count_lt_p05 = tmax_lt_p05.groupby(gbix).sum().rename_axis(ym)
    tmax_spell_lt_p05 = tmax_lt_p05.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
    tmax_sum_lt_p05 = (tmax_p05-tmax).clip(0, None).groupby(gbix).sum().rename_axis(ym)

    # Nocturnal (tmin) heat and cold - percentile thresholds
    tmin_count_gt_p95 = tmin_gt_p95.groupby(gbix).sum().rename_axis(ym)
    tmin_spell_gt_p95 = tmin_gt_p95.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
    tmin_sum_gt_p95 = (tmin-tmin_p95).clip(0, None).groupby(gbix).sum().rename_axis(ym)
    tmin_count_lt_p05 = tmin_lt_p05.groupby(gbix).sum().rename_axis(ym)
    tmin_spell_lt_p05 = tmin_lt_p05.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
    tmin_sum_lt_p05 = (tmin_p05-tmin).clip(0, None).groupby(gbix).sum().rename_axis(ym)

    # Diurnal (tmax) heat - explicit HDD thresholds
    hdd_tmax = {}
    for thresh in tmax_gt:
        # Identify days greater than the threshold with a boolean mask
        tmax_gt_thresh = tmax > thresh
        tmax_count_gt = tmax_gt_thresh.groupby(gbix).sum().rename_axis(ym)
        hdd_tmax[f'tmax_count_gt_{thresh}'] = tmax_count_gt
        tmax_spell_gt = tmax_gt_thresh.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
        hdd_tmax[f'tmax_spell_gt_{thresh}'] = tmax_spell_gt
        tmax_sum_gt = (tmax-thresh).clip(0, None).groupby(gbix).sum().rename_axis(ym)
        hdd_tmax[f'tmax_sum_gt_{thresh}'] = tmax_sum_gt
    hdd_tmax = pd.DataFrame(hdd_tmax)

    # Nocturnal heat (tmin); explicit HDD thresholds
    hdd_tmin = {}
    for thresh in tmin_gt:
        # Identify days greater than the threshold with a boolean mask
        tmin_gt_thresh = tmin > thresh
        tmin_count_gt = tmin_gt_thresh.groupby(gbix).sum().rename_axis(ym)
        hdd_tmin[f'tmin_count_gt_{thresh}'] = tmin_count_gt
        tmin_spell_gt = tmin_gt_thresh.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
        hdd_tmin[f'tmin_spell_gt_{thresh}'] = tmin_spell_gt
        tmin_sum_gt = (tmin-thresh).clip(0, None).groupby(gbix).sum().rename_axis(ym)
        hdd_tmin[f'tmin_sum_gt_{thresh}'] = tmin_sum_gt
    hdd_tmin = pd.DataFrame(hdd_tmin)

    # Nocturnal cold (tmin); explicit CDD thresholds
    cdd_tmin = {}
    for thresh in tmin_lt:
        # Identify days less than the threshold with a boolean mask
        tmin_lt_thresh = tmin <= thresh
        tmin_count_lt = tmin_lt_thresh.groupby(gbix).sum().rename_axis(ym)
        cdd_tmin[f'tmin_count_lt_{thresh}'] = tmin_count_lt
        tmin_spell_lt = tmin_lt_thresh.groupby(gbix).apply(runlen).rename_axis(ym+[0]).max(level=ym)
        cdd_tmin[f'tmin_spell_lt_{thresh}'] = tmin_spell_lt
        tmin_sum_lt = (thresh-tmin).clip(0, None).groupby(gbix).sum().rename_axis(ym)
        cdd_tmin[f'tmin_sum_lt_{thresh}'] = tmin_sum_lt
    cdd_tmin = pd.DataFrame(cdd_tmin)

    # 1-, 2- and 3-day maxima and minima
    tmax24 = tmax.groupby(gbix).max().rename_axis(ym)
    tmax48 = tmax.rolling(2).mean().groupby(gbix).max().rename_axis(ym)
    tmax72 = tmax.rolling(3).mean().groupby(gbix).max().rename_axis(ym)
    tmin24 = tmin.groupby(gbix).min().rename_axis(ym)
    tmin48 = tmin.rolling(2).mean().groupby(gbix).min().rename_axis(ym)
    tmin72 = tmin.rolling(3).mean().groupby(gbix).min().rename_axis(ym)

    # Combine all indices indexed by (year, month) MultiIndex
    out = pd.DataFrame({
                        'tmax_mean': tmax_month,
                        'tmin_mean': tmin_month,
                        'dtr_mean': dtr_month,
                        'tmax_count_lt_p05': tmax_count_lt_p05.astype(int),
                        'tmax_sum_lt_p05': tmax_sum_lt_p05,
                        'tmax_spell_lt_p05': tmax_spell_lt_p05.astype(int),
                        'tmax_count_gt_p95': tmax_count_gt_p95.astype(int),
                        'tmax_sum_gt_p95': tmax_sum_gt_p95,
                        'tmax_spell_gt_p95': tmax_spell_gt_p95.astype(int),
                        'tmin_count_lt_p05': tmin_count_lt_p05.astype(int),
                        'tmin_sum_lt_p05': tmin_sum_lt_p05,
                        'tmin_spell_lt_p05': tmin_spell_lt_p05.astype(int),
                        'tmin_count_gt_p95': tmin_count_gt_p95.astype(int),
                        'tmin_sum_gt_p95': tmin_sum_gt_p95,
                        'tmin_spell_gt_p95': tmin_spell_gt_p95.astype(int),
                        'tmax24': tmax24,
                        'tmax48': tmax48,
                        'tmax72': tmax72,
                        'tmin24': tmin24,
                        'tmin48': tmin48,
                        'tmin72': tmin72
                        })
    out = pd.concat([out, hdd_tmax, hdd_tmin, cdd_tmin], axis=1)
    return out.rename_axis('index_name', axis=1).stack().rename('value')

