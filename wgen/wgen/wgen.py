#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wgen

Refined version of stochastic weather generator. Requires monthly gridded
weather data and precomputed linear trends in Parquet format.
All geocoding uses quadtreeIDs (qids).

Historic and stochastic climate indices in csv format, and a settings file
in JSON format.

Mutahar Chalmers, RMS, November 2018

Updates:
------------------------------------
13 July 2019 Modifications include:
             Update path and arguments for updated qtree function
"""

import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as st

# Import qtree module to allow latitudes to be extracted from qids
sys.path.append('/mnt/gpfs/backup/agri/code/')
from qtree import cyqtid2ll


def load_wfile(asttgs, v):
    """Load gridded historical monthly weather data."""

    # Read Parquet file and convert string quadtreeID headers back to int
    data = pd.read_parquet(asttgs['weather_files'][v])
    data.columns = data.columns.astype(int)

    # Restrict temporal extent consistently across all variables
    start, end = asttgs['start'], asttgs['end']
    yrs = data.index.get_level_values('year')
    mths = data.index.get_level_values('month')
    return data[(yrs>=start[0])&(yrs<=end[0])&(mths>=start[1])&(mths<=end[1])]


def load_cfile(asttgs):
    """Load monthly climate index timeseries in standard format."""

    clim = pd.read_csv(asttgs['clim_hist'], index_col=[0,1])[asttgs['cixs']]
    clim.index.names = ['year', 'month']

    # Restrict temporal extent consistently across all variables
    start, end = asttgs['start'], asttgs['end']
    yrs = clim.index.get_level_values('year')
    mths = clim.index.get_level_values('month')
    return clim[(yrs>=start[0])&(yrs<=end[0])&(mths>=start[1])&(mths<=end[1])]


def load_tfile(asttgs, v):
    """Load monthly trend file in standard format."""

    # Read Parquet file and convert string quadtreeID headers back to int
    trend = pd.read_parquet(asttgs['trend_files'][v])
    trend.columns = trend.columns.astype(int)
    return trend


def detrend(data, trend, pivot_year, qid=None):
    """Linearly detrend data using trend DataFrame. Handles DataFrame of monthly data
       (many qids) or Series of daily or monthly data (one qid). """

    df = {}
    years = data.index.get_level_values('year').unique()
    for month in range(1, 13):
        df[month] = pd.DataFrame(np.outer((pivot_year - years), trend[month]),
                                 index=years, columns=trend[month].index)
    trend_df = pd.concat(df, axis=0).swaplevel(0, 1).sort_index()
    trend_df.index.names = ['year', 'month']
    trend_df.columns.name = 'qid'

    # For the standard use-case in this module, detrending monthly data
    if qid is None:
        return data + trend_df
    # For use by wdaily2 for detrending daily data
    else:
        return pd.concat([data_ym + trend_df.loc[ym,int(qid)]
                          for ym, data_ym in data.groupby(level=['year','month'])])


def normality(data, seasons):
    """Kolmogorov-Smirnov normality test on data."""

    ks_pvals = {}
    for s in seasons:
        # Select data by season
        data_seas = data.xs(s, level='season')

        # Calculate exact (scipy) optimal K-S p-values
        data_z = (data_seas - data_seas.mean())/data_seas.std()
        ks_pvals[s] = pd.Series({i: st.kstest(data_z[i], 'norm').pvalue for i in data_z})
    return pd.concat(ks_pvals, axis=1)


def transform(data, shift, seasons):
    """Prepare for PCA by applying optimal Box-Cox transforms."""

    transformed, pwrs_opt = {}, {}

    def boxcox(df, pwr):
        return (df**pwr - 1)/pwr

    for s in seasons:
        # Select data by season
        data_seas = data.xs(s, level='season')

        ks_dict = {}
        # Loop over possible power transform exponents and choose best
        pwrs = np.arange(0.001, 1.001, 0.002)
        for pwr in pwrs:
            # Power transform and standardise
            tr_x = boxcox(data_seas+shift, pwr)
            tr_z = (tr_x - tr_x.mean())/tr_x.std()

            # Fast calculation of Kolomogorov-Smirnov statistics on a DataFrame
            ecdf = (np.argsort(np.argsort(tr_z, axis=0), axis=0)+1)/tr_z.count()
            ks_dict[pwr] = (ecdf - st.norm.cdf(tr_z)).abs().max()
        ks_df = pd.concat(ks_dict, axis=1)

        # Choose optimum exponents (smallest K-S statistics) and apply transform
        pwrs_opt[s] = ks_df.idxmin(axis=1)
        tr_data = boxcox(data_seas+shift, pwrs_opt[s])

        # Calculate exact (scipy) optimal K-S statistics and p-values
        tr_data_z = (tr_data - tr_data.mean())/tr_data.std()

        # Save optimal exponents and transformed data by season
        transformed[s] = tr_data

    # Recombine power transformed seasonal data
    transformed = pd.concat(transformed, names=['season']+data_seas.index.names)
    return transformed.sort_index(), pd.concat(pwrs_opt)


def areaweights(qids, qsttgs):
    """Calculate root-cosine area-weighting to be applied for cells at latitudes
       far from the equator to reflect their reduced area contribution."""

    lats = pd.Series({qid: cyqtid2ll(qid, **qsttgs)[1] for qid in qids})
    return np.sqrt(np.cos(np.deg2rad(lats)))


def pca(anom, seasons, mask, awts=None):
    """Generate EOFs and PCs assuming standardised, normalised data."""

    EOFs, PCs, PC_normal= {}, {}, {}
    for s in seasons:
        # Filter out cells that are not Gaussian enough
        anom_seas = anom.xs(s, level='season').loc[:,mask[s]]

        # Apply latitude weighting if required
        if awts is not None:
            anom_seas = anom_seas.mul(awts, axis=1).dropna(axis=1)

        # Carry out PCA using SVD
        U, S, VT = np.linalg.svd(anom_seas.values.T, full_matrices=False)
        EOFs[s] = pd.DataFrame(U, index=anom_seas.columns)
        PCs[s] = pd.DataFrame(np.diag(S).dot(VT), columns=anom_seas.index)

        # Standardise PCs for Kolmogorov-Smirnov normality test
        z_PCs = PCs[s].sub(PCs[s].mean(axis=1), axis=0).div(PCs[s].std(axis=1), axis=0)
        PC_normal[s] = pd.Series({i: st.kstest(vals, 'norm')[1]
                                  for i, vals in z_PCs.iterrows()})

    # Combine all seasons into single DataFrames
    EOFs = pd.concat(EOFs, names=['season','qid'])
    PCs = pd.concat(PCs, names=['season']+anom_seas.index.names, axis=1)
    PC_normal = pd.concat(PC_normal, names=['season','year'], axis=1)
    return EOFs, PCs, PC_normal


def find_analogue(PCs_stoc, PCs_hist):
    """Determine analogue historical years for given historic and stochastic PCs."""

    d2_wt = {}

    # Variance and variance proportions of historic PCs
    sig2 = PCs_hist.var()
    sig2_wt = sig2.div(sig2.sum(level=0), level=0)

    # For each historic year, calculate variance-weighted squared Euclidean distances
    # to all stochastic years in PC-space
    for yr in PCs_hist.index.get_level_values(0).unique():
        d2 = PCs_stoc.sub(PCs_hist.loc[yr], axis=1)**2
        d2_wt[yr] = d2.mul(sig2_wt, axis=1).sum(axis=1)
    return pd.concat(d2_wt).unstack(level=0).idxmin(axis=1)


def seas_agger_mth(df, seasons, slen, agg='mean'):
    """Fast aggregation of sorted monthly data by season."""

    if agg == 'sum':
        df_agg = df.rolling(slen).sum()
    else:
        df_agg = df.rolling(slen).mean()
    df_agg = df_agg[df_agg.index.get_level_values('month').isin(seasons)]
    df_agg.index.names = ['year','season']
    return df_agg.dropna(axis=1, how='all').dropna()


def analyse(sttgs, outpath):
    """Analyse weather data."""

    # Load seasons and settings, and calculate season length assuming equal lengths
    seasons = sttgs['seasons']
    slen = 12 // len(seasons)
    asttgs, ssttgs, qsttgs = sttgs['analysis'], sttgs['synthesis'], sttgs['qtree']

    # Load all qids and calculate area weights for cells away from the equator
    with open(qsttgs.pop('qidsfile', None), 'r') as f:
        qids = np.array([int(row) for row in f.readlines()])
    awts = areaweights(qids, qsttgs)

    # Load monthly historic climate indices, and aggregate by season
    clim_hist = load_cfile(asttgs)
    clim_hist = seas_agger_mth(clim_hist, seasons, slen)
    clim_hist = pd.concat({'clim': clim_hist}, axis=1)

    # Dicts of EOFs and PCs to simplify generation of correlation matrix
    mean, std, EOFs, PCs_hist= {}, {}, {}, {}
    hist_seas_orig, pwrs = {}, {}

    # Loop over each weather variable
    for v in sttgs['variables']:
        print('Analysing {0}...'.format(v))

        # Load historic weather and trends at monthly resolution
        hist_month = load_wfile(asttgs, v)
        trend = load_tfile(asttgs, v)

        # Identify qids which are in the nominal list AND in the data AND in the trend
        qids = np.intersect1d(qids, hist_month.columns)
        qids = np.intersect1d(qids, trend.index)

        # Detrend linearly as required (default trend file has no trend) 
        hist_month = detrend(hist_month[qids], trend, asttgs['pivot_year'])

        # Aggregate by season - keep a seasonally-aggregated version of original data
        hist_seas_orig[v] = seas_agger_mth(hist_month, seasons, slen, sttgs['agg'][v])
        hist_seas = seas_agger_mth(hist_month, seasons, slen, sttgs['agg'][v])

        # Apply Box-Cox power transform if required, and calculate anomalies
        print('  Box-Cox...')
        if asttgs['trans'][v]:
            hist_seas, pwrs[v] = transform(hist_seas, asttgs['shift'][v], seasons)

        # Calculate normality of qid-level data and generate mask for PCA
        print('  Normality by qid of transformed data...')
        qid_ks_pvals = normality(hist_seas, seasons)
        mask = qid_ks_pvals >= asttgs['ks_pval_min']

        # Calculate standardised anomalies
        mean[v] = hist_seas.mean(level='season')
        std[v] = hist_seas.std(level='season')
        anom = hist_seas.subtract(mean[v], level='season').div(std[v], level='season')

        # Perform PCA decomposition
        print('  PCA...')
        EOFs[v], PCs_hist[v], PC_ks_pvals = pca(anom, seasons, mask, awts)

        # Write intermediate output to disk
        EOFs[v].to_csv(outpath+'/EOFs_{}.csv'.format(v), float_format='%.6f')
        PCs_hist[v].to_csv(outpath+'/PCs_hist_{}.csv'.format(v), float_format='%.6f')
        PC_ks_pvals.to_csv(outpath+'/PC_KS_pvals_{}.csv'.format(v), float_format='%.6f')
        qid_ks_pvals.to_csv(outpath+'/qid_KS_pvals_{}.csv'.format(v), float_format='%.6f')

    # Concatenate dicts into DataFrames and write to disk
    hist_seas_orig = pd.concat(hist_seas_orig, names=['variable'])
    hist_seas_orig.columns = hist_seas_orig.columns.astype(str)
    hist_seas_orig.to_parquet(outpath+'/hist_seas.parquet')
    PCs_hist = pd.concat(PCs_hist).T

    # Write EOFs to NetCDF for easy plotting
    EOFs_df = pd.concat(EOFs)
    EOFs_df = EOFs_df.stack().unstack('qid')
    EOFs_df.columns = pd.MultiIndex.from_tuples([cyqtid2ll(int(qid), **qsttgs)
                                                 for qid in EOFs_df.columns])
    EOFs_df = EOFs_df.stack([1,0])
    EOFs_df.index.names = ['variable','season','EOF','lat','lon']
    EOFs_df.name = 'value'
    EOFs_df.to_xarray().to_netcdf(outpath+'/EOFs.nc')

    # Correlations between PCs and climate indices, and Cholesky matrices
    cholmat = {}
    for s in seasons:
        PCcix = pd.concat([clim_hist.xs(s, level='season'),
                           PCs_hist.xs(s, level='season').dropna(axis=1)], axis=1)
        rho = PCcix.dropna().corr()
        rho.to_csv(outpath+'/PC_corrmat_s{0}.csv'.format(s), float_format='%.6f')

        # Calculate p-values
        cols, pval = PCcix.columns, {}
        for ci in cols:
            pval[ci] = [st.pearsonr(PCcix[ci], PCcix[cj])[1] for cj in cols]
        pval = pd.DataFrame(pval, index=cols)
        pval.to_csv(outpath+'/PC_pvalmat_s{0}.csv'.format(s), float_format='%.6f')

        L = np.linalg.cholesky(rho + np.eye(rho.shape[0])*1e-9)
        cholmat[s] = pd.DataFrame(L, index=rho.columns, columns=rho.columns)

    # Write qid list and N_qids to disk
    with open(outpath+'/qids.txt', 'w') as f:
        f.write('\n'.join(map(str, qids))+'\n')

    # Generate NetCDF of elevation
    elev_df = pd.read_csv(ssttgs['elev_file'], header=None, names=['qid','elevation'])
    elev_df = pd.DataFrame([cyqtid2ll(int(row['qid']), **qsttgs) + (row['elevation'],)
                            for ix, row in elev_df.iterrows() if row['qid'] in qids],
                            columns=['lon','lat','elevation']).set_index(['lat','lon'])
    elev_df.to_xarray().to_netcdf(outpath+'/../analytics/elevation.nc')

    kwargs = {'EOFs': EOFs, 'PCs_hist': PCs_hist, 'cholmat': cholmat,
              'mean': mean, 'std': std, 'pwrs': pwrs}
    return kwargs


def PC_gen(sttgs, N_PCs, seasons):
    """Simulate standard normal random PCs."""

    seed, N_years = sttgs['synthesis']['seed'], sttgs['synthesis']['N_years']

    # Seed the RNG and generate all standard normal random PC samples
    np.random.seed(seed)

    # Define index for the stochastic PCs
    N = len(sttgs['seasons'])
    ssns_map = dict(zip(range(N), seasons))
    ix0 = np.array(range(N_years*N))
    yrs, ssns_temp  = ix0 // N + 1, ix0 % N
    ssns = np.fromiter((ssns_map.get(ssn) for ssn in ssns_temp), int)
    ix = pd.MultiIndex.from_arrays([yrs, ssns], names=['year','season'])

    z_PCs = {}
    for v in sttgs['variables']:
        if v not in sttgs['synthesis']['z_files']:
            # Simulate on the fly
            z_PCs[v] = pd.DataFrame(np.random.normal(size=(N_years*N, N_PCs)), index=ix)
        else:
            # Load from file
            z_PCs[v] = pd.read_hdf(sttgs['synthesis']['z_files'][v], v).loc[ix]
    return pd.concat(z_PCs, axis=1)


def synthesise(sttgs, EOFs, PCs_hist, cholmat, mean, std, pwrs, outpath):
    """Synthesise weather from stochastic PCs and EOFs."""

    # Load seasons and settings, and calculate mappings
    seasons = sttgs['seasons']
    slen = 12 // len(seasons)
    asttgs, ssttgs, qsttgs = sttgs['analysis'], sttgs['synthesis'], sttgs['qtree']
    avars = ssttgs['analogue_vars_wt']

    # N_PCs can vary by season if offset !=0, so choose the largest value
    N_PCs = max([len(PCs_hist.loc[s]) for s in seasons])
    N_qids = mean[sttgs['variables'][0]].columns.size

    # Load stochastic climate indices and add column level
    clim_stoc_mth = pd.read_csv(ssttgs['clim_stoc'], index_col=[0, 1])
    clim_stoc = seas_agger_mth(clim_stoc_mth[asttgs['cixs']], seasons, slen, agg='mean')
    clim_stoc = pd.concat({'clim': clim_stoc}, axis=1)

    # Generate random PCs, combine with clim_stoc and standardise
    z = PC_gen(sttgs, N_PCs, seasons)
    z_stoc = pd.concat([clim_stoc.reindex(z.index), z], axis=1).dropna()
    z_stoc = (z_stoc - z_stoc.mean(level='season'))/z_stoc.std(level='season')

    # Loop over seasons and generate stochastic PCs and analogue years
    PCs_stoc, analogue, out = {}, {}, {}
    for s in seasons:
        print('Generating PCs for season {0}'.format(s))
        # Apply correlations to z_stoc and scale PCs by std deviations
        z_stoc_seas = z_stoc.xs(s, level='season')[cholmat[s].columns]
        z_stoc_corr =  z_stoc_seas.dot(cholmat[s].T).drop('clim', axis=1)
        PCs_stoc[s] = z_stoc_corr.mul(PCs_hist.loc[s].std(), axis=1).dropna(axis=1)

        # Determine analogue historic month and write to file
        analogue[s] = find_analogue(PCs_stoc[s][avars], PCs_hist.loc[s][avars])

        # Generate stochastic seasonal weather by combining EOFs and PCs
        stoc_seas = {}
        for v in sttgs['variables']:
            print('Synthesising {}'.format(v))

            # Combine PCs and EOFs, std deviations and means, and convert anomalies
            anom = PCs_stoc[s][v].dot(EOFs[v].loc[s].dropna(axis=1).T)
            stoc = anom.mul(std[v].loc[s], axis=1).add(mean[v].loc[s], axis=1)
            if asttgs['trans'][v]:
                # Load exponents, invert Box-Cox transform and set nulls to 0
                pwr, shift = pwrs[v].loc[s], asttgs['shift'][v]
                stoc = ((stoc.mul(pwr, axis=1) + 1).pow(1/pwr, axis=1) - shift).fillna(0)

            # Clip values between allowable bounds and cast to float32
            stoc_seas[v] = stoc.clip(*ssttgs['bounds'][v]).astype(np.float32)
        out[s] = pd.concat(stoc_seas, names=['variable'])

    print('Writing to file...')

    # Concatenate stochastic PCs and analogue mapping to DataFrames, and write to file
    PCs_stoc = pd.concat(PCs_stoc, names=['season']).stack(level=1)
    PCs_stoc.to_parquet(outpath+'/PCs_stoc.parquet')

    stoc_seas = pd.concat(out, names=['season'])
    stoc_seas.columns = stoc_seas.columns.astype(str)
    stoc_seas.to_parquet(outpath+'/stoc_seas.parquet')

    # Add unique historic season identifier
    analogue = pd.concat(analogue, names=['season']).to_frame('histyear').reset_index()
    analogue = analogue.sort_values(['histyear','season'])
    analogue.to_parquet(outpath+'/analogue.parquet')
    return stoc_seas


if __name__ == '__main__':
    _, sttgsfile, histpath, stocpath = sys.argv

    # Load settings file from JSON to dict
    with open(sttgsfile, 'r') as f:
        sttgs = json.load(f)

    kwargs = analyse(sttgs, histpath)
    if kwargs is not None:
        stoc = synthesise(sttgs, outpath=stocpath, **kwargs)
        print('Complete.')
    else:
        print('Analysis failed; exiting.')

