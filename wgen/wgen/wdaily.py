#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wdaily

Generation of daily weather from seasonal weather data produced by wgen.
Includes ETo and SSF calculation.

Mutahar Chalmers, RMS, 2018-9
"""

import sys
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from wgen import load_tfile, detrend
import wanalytics

sys.path.append('/mnt/gpfs/backup/agri/code/eto/')
import eto


def qidelev(qid, elev_fpath):
    """Retrieve qid mean elevation from csv."""

    elevs = pd.read_csv(elev_fpath, header=None, names=['z'], index_col=0)
    return elevs.loc[qid,'z']


def season2day(sttgs, histpath, stocpath, qid):
    """Calculate stochastic daily weather data."""

    slen = 12 // len(sttgs['seasons'])
    asttgs, ssttgs, qsttgs = sttgs['analysis'], sttgs['synthesis'], sttgs['qtree']

    # Extract latitude from qid for ETo calculation
    _ = qsttgs.pop('qidsfile')
    _, lat = eto.qtid2ll(int(qid), **qsttgs)

    # Determine elevation above sea level if applicable
    if len(ssttgs['elev_file']) > 0:
        elev = qidelev(int(qid), ssttgs['elev_file'])
    else:
        elev = 0

    # Make season map - assume standard season naming convention
    variables, seasons = sttgs['variables'], np.array(sttgs['seasons'])
    m2s = seasons[(np.digitize(range(1, 13), seasons, right=True) % len(seasons))]
    smap = dict(zip(range(1, 13), m2s))

    # Load analogue mapping and generate unique historic season identifier
    anal = pd.read_parquet(stocpath+'/analogue.parquet').sort_values(['histyear','season'])
    offset = anal.iloc[0]['histyear']*12 + anal.iloc[0]['season']
    anal['key'] = offset + anal['season'].diff().abs().cumsum().fillna(0)

    # Load historic daily data
    histdailypath = ssttgs['histdailypath']
    hist_daily = pd.read_parquet(histdailypath+f'/{qid}.parquet').sort_index()

    # Restrict temporal extent consistently across all variables
    #start, end = asttgs['start'], asttgs['end']
    #yrs = hist_daily.index.get_level_values('year')
    #mths = hist_daily.index.get_level_values('month')
    #hist_daily = hist_daily[(yrs>=start[0])&(yrs<=end[0])&(mths>=start[1])&(mths<=end[1])]

    # Load trend file and detrend
    trends = {v: load_tfile(sttgs['analysis'], v) for v in variables}
    hist_daily = pd.concat([detrend(hist_daily[v], trends[v], asttgs['pivot_year'], qid)
                           for v in variables], axis=1)

    # Calculate reference evapotranspiration (ETo) for detrended data
    hist_daily['eto'] = eto.calc_ETo(hist_daily, lat, elevation=elev)
    hist_daily = hist_daily[variables+['eto']]

    # Generate unique historic season identifiers
    hist_daily = hist_daily.reset_index().rename(columns={'year':'histyear'})
    seas = hist_daily['month'].map(smap.get)
    offset = hist_daily['histyear'][0]*12 + seas[0]
    key = offset + seas.diff().abs().cumsum().fillna(0)
    hist_daily['key'] = key.values
    hist_daily['QID'] = int(qid)

    # Load historic and stochastic seasonal data
    hist_seas = pq.read_pandas(histpath+'/hist_seas.parquet', columns=[qid])
    hist_seas = hist_seas.to_pandas()[qid].unstack('variable').reset_index().rename(columns={'year':'histyear'})
    stoc_seas = pq.read_pandas(stocpath+'/stoc_seas.parquet', columns=[qid])
    stoc_seas = stoc_seas.to_pandas()[qid].unstack('variable').reset_index()

    # Allocate historic daily data to corresponding stochastic year
    hist_anal_daily = anal.merge(hist_daily, on='key', suffixes=('_anal',''))

    # Allocate historic seasonal data to corresponding stochastic year
    hist_anal = anal.merge(hist_seas, on=['histyear','season'])

    # Calculate Seasonal Scaling Factors (SSFs)
    ssfs = hist_anal.merge(stoc_seas, on=['year','season'], suffixes=('_hist','_stoc'))
    for v in variables:
        ssfs[v] = ssfs[v+'_stoc']/ssfs[v+'_hist']
    ssfs = ssfs[['year','key']+variables]

    # Combine historic daily data with SSFs to obtain daily stochastic data
    hist_ssf = hist_anal_daily.merge(ssfs, on=['year','key'], suffixes=('_hist','_ssf'))

    # Adjust stochastic year for those seasons which straddle calendar years
    hist_ssf['year'] = hist_ssf['year']+hist_ssf['histyear']-hist_ssf['histyear_anal']
    hist_ssf = hist_ssf.set_index(['year','month','day','doy','key'])

    # Calculate main stochastic daily weather DataFrame
    stoc = pd.concat({v: (hist_ssf[v+'_hist']*hist_ssf[v+'_ssf']) for v in variables},
                     axis=1).sort_index()

    # Calculate reference evapotranspiration (ETo)
    _, lat = eto.qtid2ll(int(qid), **qsttgs)
    stoc['eto'] = eto.calc_ETo(stoc, np.deg2rad(lat), elevation=elev)

    # Final format for stochastic daily data
    stoc = stoc[variables+['eto']].astype(np.float32)

    # Calculate SSFs for ET0
    hist_eto_seas = hist_ssf['eto'].sum(level=['year','key'])
    stoc_eto_seas = stoc['eto'].sum(level=['year','key'])
    eto_ssf = stoc_eto_seas/hist_eto_seas

    # Add QID and ETo to ssfs, and make 32-bit
    ssfs['QID'] = int(qid)
    ssfs = ssfs.set_index(['year','key','QID'])
    ssfs['eto'] = eto_ssf
    ssfs[variables+['eto']] = ssfs[variables+['eto']].astype(np.float32)
    ssfs = ssfs.sort_index().reset_index()
    ssfs[['year','key','QID']] = ssfs[['year','key','QID']].astype(np.int32)

    # Make hist_daily 32-bit
    cols = ['histyear','month','day','doy','key','QID']
    hist_daily[cols] = hist_daily[cols].astype(np.int32)
    hist_daily[variables+['eto']] = hist_daily[variables+['eto']].astype(np.float32)

    # Check that seasonal aggregation of daily data equals seasonal data for all years
    stoc_seas = stoc_seas.set_index(['year','season']).sort_index()
    stoc_seas_fromdaily = agg_season(stoc, sttgs)
    test = np.allclose(stoc_seas, stoc_seas_fromdaily)

    return stoc, ssfs, hist_daily, test


def histSSF2stoc(hist_daily, ssfs, sttgs):
    """
    Combine historic daily data with Stochastic Seasonal Factors (SSFs) to get
    daily stochastic data.
    """

    variables, seasons = list(sttgs['variables']), np.array(sttgs['seasons'])

    # Add ETo to list of variables
    variables.append('eto')

    # Make season map - assume standard season naming convention
    m2s = seasons[(np.digitize(range(1, 13), seasons, right=True) % len(seasons))]
    smap = dict(zip(range(1, 13), m2s))

    # Combine historic daily data with SSFs to obtain daily stochastic data
    ssf_hist = ssfs.merge(hist_daily, on=['key','QID'], suffixes=('_ssf','_hist'))

    # Subtract 1 from stochastic year for those seasons which straddle calendar years
    ssf_hist['season'] = ssf_hist['month'].map(smap)
    straddle = ssf_hist['month'] > ssf_hist['season']
    ssf_hist.loc[straddle, 'year'] = ssf_hist.loc[straddle, 'year'] - 1
    ssf_hist = ssf_hist.set_index(['year','month','day','doy'])

    # Calculate main stochastic daily weather DataFrame
    stoc = pd.concat({v: (ssf_hist[v+'_hist']*ssf_hist[v+'_ssf']) for v in variables},
                     axis=1).sort_index()
    return stoc


def agg_season(stoc, sttgs):
    """Seasonally aggregate daily data."""

    seasagg = {}
    slen = 12 // len(sttgs['seasons'])
    for v in sttgs['variables']:
        if sttgs['agg'][v] == 'sum':
            seasagg[v] = stoc[v].sum(level=['year','month']).rolling(slen).sum()
        else:
            seasagg[v] = stoc[v].mean(level=['year','month']).rolling(slen).mean()
    seasagg = pd.concat(seasagg, axis=1).sort_index()
    return seasagg[seasagg.index.get_level_values('month').isin(sttgs['seasons'])]


def agg_month(wdata, sttgs):
    """Aggregate daily data by month."""

    # First, generate monthly aggregated DataFrames from daily data
    wdata_m_agg = wdata.mean(level=['year','month'])
    for v in sttgs['variables']:
        if sttgs['agg'][v] == 'sum':
            wdata_m_agg[v] = wdata[v].sum(level=['year','month'])

    wdata_m_max = wdata.max(level=['year','month'])
    wdata_m_min = wdata.min(level=['year','month'])
    wdata_m_var = wdata.var(level=['year','month'])

    # Handle ETo explicitly since it won't be in settings file
    wdata_m_agg['eto'] = wdata['eto'].sum(level=['year','month'])
    wdata_m_max['eto'] = wdata['eto'].max(level=['year','month'])
    wdata_m_min['eto'] = wdata['eto'].min(level=['year','month'])
    wdata_m_var['eto'] = wdata['eto'].var(level=['year','month'])

    return wdata_m_agg, wdata_m_max, wdata_m_min, wdata_m_var


def run_analytics(wdata, sttgs, qid, histflag):
    """Generate analytics on monthly weather data."""

    # Load climate indices and combine with wdata
    clim = wanalytics2.load_climate(sttgs, hist=histflag).reindex(wdata.index)
    wdata = pd.concat([wdata, clim], axis=1).dropna(axis=1)

    # Calculate climatology and quantiles
    wdata_clima, wdata_quant = wanalytics2.climquant(wdata, qid)

    # Calculate cross-correlations
    wdata_xcorr = wanalytics2.xcorr(wdata, qid)
    return wdata_clima, wdata_quant, wdata_xcorr


if __name__ == '__main__':
    _, sttgsfile, histpath, stocpath, qid = sys.argv

    # Load settings file from JSON to dict
    with open(sttgsfile, 'r') as f:
        sttgs = json.load(f)
    seasons = sttgs['seasons']

    # Generate SSFs and historic daily files and write to disk
    stoc, ssfs, hist_daily, test = season2day(sttgs, histpath, stocpath, qid)
    ssfs.to_parquet(stocpath+f'/ssfs/{qid}.parquet')
    hist_daily.to_parquet(stocpath+f'/histdaily/{qid}.parquet')

    # Write QID-level results only if qid is in list of reference QIDs
    if int(qid) in sttgs['analytics']['qids_ref']:
        stoc.to_parquet(stocpath+f'/daily/{qid}.parquet')

    # Run historic and stochastic analytics and write to file
    # Need to modify format of hist_daily to 'standard' format
    hist = hist_daily.rename(columns={'histyear':'year'}
                             ).set_index(['year','month','day','doy']
                                         ).drop(['key','QID'], axis=1)

    # Calculate daily quantiles before aggregating by month
    hist_m_day_quant = wanalytics2.dailyquant(hist, int(qid))
    stoc_m_day_quant = wanalytics2.dailyquant(stoc, int(qid))

    # Calculate monthly aggregates
    hist_m_agg, hist_m_max, hist_m_min, hist_m_var = agg_month(hist, sttgs)
    stoc_m_agg, stoc_m_max, stoc_m_min, stoc_m_var = agg_month(stoc, sttgs)

    # Calculate climatology, quantiles and cross-correlations for historic
    hist_m_agg_clima, hist_m_agg_quant, hist_m_agg_xcorr = run_analytics(hist_m_agg, sttgs, qid, histflag=True)
    hist_m_max_clima, hist_m_max_quant, hist_m_max_xcorr = run_analytics(hist_m_max, sttgs, qid, histflag=True)
    hist_m_min_clima, hist_m_min_quant, hist_m_min_xcorr = run_analytics(hist_m_min, sttgs, qid, histflag=True)
    hist_m_var_clima, hist_m_var_quant, hist_m_var_xcorr = run_analytics(hist_m_var, sttgs, qid, histflag=True)

    # Calculate climatology, quantiles and cross-correlations for stochastic
    stoc_m_agg_clima, stoc_m_agg_quant, stoc_m_agg_xcorr = run_analytics(stoc_m_agg, sttgs, qid, histflag=False)
    stoc_m_max_clima, stoc_m_max_quant, stoc_m_max_xcorr = run_analytics(stoc_m_max, sttgs, qid, histflag=False)
    stoc_m_min_clima, stoc_m_min_quant, stoc_m_min_xcorr = run_analytics(stoc_m_min, sttgs, qid, histflag=False)
    stoc_m_var_clima, stoc_m_var_quant, stoc_m_var_xcorr = run_analytics(stoc_m_var, sttgs, qid, histflag=False)

    # Calculate two-sample K-S p-values
    m_agg_ks = wanalytics2.similarity_KS(hist_m_agg, stoc_m_agg, qid)
    m_max_ks = wanalytics2.similarity_KS(hist_m_max, stoc_m_max, qid)
    m_min_ks = wanalytics2.similarity_KS(hist_m_min, stoc_m_min, qid)
    m_var_ks = wanalytics2.similarity_KS(hist_m_var, stoc_m_var, qid)

    # Write to file
    hist_m_day_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_day_quant')

    hist_m_agg_clima.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_agg_clima')
    hist_m_max_clima.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_max_clima')
    hist_m_min_clima.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_min_clima')
    hist_m_var_clima.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_var_clima')

    hist_m_agg_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_agg_quant')
    hist_m_max_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_max_quant')
    hist_m_min_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_min_quant')
    hist_m_var_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_var_quant')

    hist_m_agg_xcorr.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_agg_xcorr')
    hist_m_max_xcorr.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_max_xcorr')
    hist_m_min_xcorr.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_min_xcorr')
    hist_m_var_xcorr.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'hist_m_var_xcorr')

    stoc_m_day_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_day_quant')

    stoc_m_agg_clima.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_agg_clima')
    stoc_m_max_clima.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_max_clima')
    stoc_m_min_clima.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_min_clima')
    stoc_m_var_clima.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_var_clima')

    stoc_m_agg_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_agg_quant')
    stoc_m_max_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_max_quant')
    stoc_m_min_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_min_quant')
    stoc_m_var_quant.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_var_quant')

    stoc_m_agg_xcorr.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_agg_xcorr')
    stoc_m_max_xcorr.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_max_xcorr')
    stoc_m_min_xcorr.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_min_xcorr')
    stoc_m_var_xcorr.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'stoc_m_var_xcorr')

    m_agg_ks.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'm_agg_ks')
    m_max_ks.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'm_max_ks')
    m_min_ks.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'm_min_ks')
    m_var_ks.to_hdf(stocpath+f'/../analytics/raw/{qid}.h5', 'm_var_ks')

    # Identify QIDs which don't have a perfect match between stochastic seasonal
    # and seasonally aggregated stochastic daily values
    if not test:
        with open(stocpath+f'/checks/{qid}.MISMATCH', 'w') as f:
            f.write('Mismatch between aggregated daily and seasonal data.')

