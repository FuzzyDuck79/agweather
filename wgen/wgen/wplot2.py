#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wplot2

Plotting functions for analytics produced by wanalytics2.

Mutahar Chalmers, RMS, 2018-2019
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

# Import qtree module to allow latitudes to be extracted from qids
sys.path.append('/mnt/gpfs/backup/agri/team/mutahar/code/qtree/')
from qtree import qtid2ll

months = range(1, 13, 1)
mmap = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
        7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}


def basemap(nrow, ncol, sttgs, axdim=4, cbar=True, noticks=True):
    """Layout basemap in M x N configuration."""

    # Load country and admin1 shapefiles - assume Natural Earth columns
    world = gpd.read_file(sttgs['analytics']['countryshp'])
    country = world[world['ISO_A2']==sttgs['analytics']['country']]
    admin1 = gpd.read_file(sttgs['analytics']['admin1shp'])
    admin1 = admin1[admin1['iso_a2']==sttgs['analytics']['country']]

    # Longitude and latitude bounds
    lonbounds = sttgs['analytics']['lonbounds']
    latbounds = sttgs['analytics']['latbounds']

    # Geo-slice world GeoDataFrame to the bounding box
    world = world.cx[slice(*lonbounds), slice(*latbounds)]

    if cbar:
        fig = plt.figure(figsize=(ncol*axdim+1, nrow*axdim))
    else:
        fig = plt.figure(figsize=(ncol*axdim, nrow*axdim))

    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot2grid((nrow, ncol), (i, j))
            if noticks:
                ax.set_xticklabels([]); ax.set_yticklabels([]);
                ax.set_xticks([]); ax.set_yticks([]);

            # Add world, country and admin1 boundaries, and set bbox
            country.boundary.plot(ax=ax, linewidth=0.5, color='k')
            admin1.boundary.plot(ax=ax, linewidth=0.1, color='0.75')
            world.boundary.plot(ax=ax, color='0.5', linewidth=0.2)
            ax.set_xlim(lonbounds); ax.set_ylim(latbounds)

    # Add colourbar
    if cbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    return fig


def qidrefplot(sttgs):
    """Plot map with reference qids highlighted and labelled."""

    # Load reference qids and URG shapefile
    qids = sttgs['analytics']['qids_ref']
    URG = gpd.read_file(sttgs['analytics']['qidshp']).set_index('geoid').reindex(qids)

    fig = basemap(1, 1, sttgs, axdim=12, cbar=False, noticks=False)
    URG.plot(ax=fig.axes[0])
    for qid in qids:
        xy = (URG.loc[qid,'geometry'].centroid.x, URG.loc[qid,'geometry'].centroid.y)
        fig.axes[0].annotate(qid, xy, fontsize='small')
    fig.axes[0].set_title('Reference QIDs')
    return fig


def qidnorm(sttgs, inpath, v, cmap='jet'):
    """Plot map with normality of qid-level seasonal anomalies."""

    # Make mapping from season code to more descriptive season label (e.g. DJF)
    seasons = sttgs['seasons']
    smap = {s: ''.join(map(lambda x: mmap.get(x)[0],
                       np.roll(months, -s)[-np.diff(seasons)[0]:])) for s in seasons}

    # Load reference qids and URG shapefile
    with open(sttgs['qtree']['qidsfile']) as f:
        qids = [int(qid) for qid in f.read().strip().split('\n')]
    URG = gpd.read_file(sttgs['analytics']['qidshp']).set_index('geoid').reindex(qids)

    # Load Kolmogorov-Smirnov p-values
    ks_pvals = pd.read_csv(inpath+'/hist/qid_KS_pvals_{}.csv'.format(v), index_col=0)

    # Set up colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []

    # Make plot
    fig = basemap(1, len(seasons), sttgs)
    for i, s in enumerate(seasons):
        URG['temp'] = ks_pvals[str(s)]
        URG.dropna().plot('temp', vmin=0, vmax=1, cmap=cmap, ax=fig.axes[i])
        fig.axes[i].set_title(smap[s])

    fig.colorbar(sm, cax=fig.axes[-1])
    fig.suptitle('Kolmogorov-Smirnov normality test p-value | {}'.format(v))

    return fig


def eof(sttgs, inpath, v, N=5):
    """Load EOFs for plotting."""

    # Make mapping from season code to more descriptive season label (e.g. DJF)
    seasons = sttgs['seasons']
    smap = {s: ''.join(map(lambda x: mmap.get(x)[0],
                       np.roll(months, -s)[-np.diff(seasons)[0]:])) for s in seasons}

     # Load EOFs 
    EOFs = pd.read_csv(inpath+'hist/EOFs_{}.csv'.format(v), index_col=[0,1])

    kwargs = {}
    kwargs['keys'] = [(s, str(n)) for s in seasons for n in range(N)]
    kwargs['titles'] = ['{0}, EOF{1}'.format(smap[s], n) for s in seasons for n in range(N)]
    kwargs['nrow'] = len(seasons)
    kwargs['ncol'] = N
    kwargs['suptitle'] = 'EOFs | {}'.format(v)

    # Rearrange structure of EOFs for plotting using same spatial routines as above
    kwargs['data'] = EOFs[EOFs.columns[:N]].stack().unstack('qid')

    # Extra kwargs specially for EOFs
    kwargs['cmap'] = 'RdBu'
    kwargs['symm'] = True
    return kwargs


def splot(sttgs, data, nrow, ncol, keys, titles, suptitle, cmap='viridis_r', symm=False):
    """Plot spatial analytics."""

    # Load qids and URG shapefile
    with open(sttgs['qtree']['qidsfile']) as f:
        qids = [int(qid) for qid in f.read().strip().split('\n')]
    URG = gpd.read_file(sttgs['analytics']['qidshp']).set_index('geoid').reindex(qids)

    # Set colour scale bounds
    vmax = max([data.max().max(), data.max().max()])
    vmin = min([data.min().min(), data.min().min()])
    if symm:
        vmax = max(abs(vmax), abs(vmin))
        vmin = -vmax

    # Set up colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    # Generate basemap
    fig = basemap(nrow, ncol, sttgs)

    # Plot heatmap
    for i, key, title in zip(range(len(keys)), keys, titles):
        URG['temp'] = data.loc[key]
        URG.dropna().plot('temp', vmin=vmin, vmax=vmax, cmap=cmap, ax=fig.axes[i])
        fig.axes[i].set_title(title)

    fig.colorbar(sm, cax=fig.axes[-1])
    fig.suptitle(suptitle)
    return fig


def eof(sttgs, inpath, v, N=5):
    """Load EOFs for plotting."""

    # Make mapping from season code to more descriptive season label (e.g. DJF)
    seasons = sttgs['seasons']
    smap = {s: ''.join(map(lambda x: mmap.get(x)[0],
                       np.roll(months, -s)[-np.diff(seasons)[0]:])) for s in seasons}

     # Load EOFs 
    EOFs = pd.read_csv(inpath+'hist/EOFs_{}.csv'.format(v), index_col=[0,1])

    kwargs = {}
    kwargs['keys'] = [(s, str(n)) for s in seasons for n in range(N)]
    kwargs['titles'] = ['{0}, EOF{1}'.format(smap[s], n) for s in seasons for n in range(N)]
    kwargs['nrow'] = len(seasons)
    kwargs['ncol'] = N
    kwargs['suptitle'] = 'EOFs | {}'.format(v)

    # Rearrange structure of EOFs for plotting using same spatial routines as above
    kwargs['data'] = EOFs[EOFs.columns[:N]].stack().unstack('qid')

    # Extra kwargs specially for EOFs
    kwargs['cmap'] = 'RdBu'
    kwargs['symm'] = True
    return kwargs


def pc_varexp(sttgs, inpath, v):
    """Generate PC plots for variance explained."""

    seasons = sttgs['seasons']
    smap = {s: ''.join(map(lambda x: mmap.get(x)[0],
                       np.roll(months, -s)[-np.diff(seasons)[0]:])) for s in seasons}

    # Load PCs and associated K-S normality test p-values
    PCs = pd.read_csv(inpath+'/hist/PCs_hist_{}.csv'.format(v), index_col=0, header=[0,1])

    # Calculate PC variance by season
    PCvar = PCs.var(axis=1, level=0)
    PCvar_norm = PCvar/PCvar.sum()

    # Plot variance explained
    fig = plt.figure(figsize=(4, len(seasons)*4))
    for i, s in enumerate(seasons):
        ax = plt.subplot2grid((len(seasons), 1), (i, 0))
        PCvar_norm.cumsum()[str(s)].plot(ax=ax, color='r')
        ax2 = PCvar_norm[str(s)].plot.bar(ax=ax, secondary_y=True, color='b')
        ax.set_ylim([0,1])
        ax2.set_ylim([0, 1.05*PCvar_norm.max().max()])
        ax.set_title('{}'.format(smap[s]))
        ax.set_xlabel('PC')
        ax.axhline(0.8, color='0.5')

        ax.set_ylabel('Cumulative variance explained')
        ax2.set_ylabel('Variance explained')

        # Fix x-axis label density
        for j, label in enumerate(ax.get_xticklabels()):
            if j % 5 != 0:
                label.set_visible(False)
    fig.suptitle('PC variance explained | {}'.format(v))
    return fig


def pc_norm(sttgs, inpath, v):
    """Generate plots for PC normality."""

    seasons = sttgs['seasons']
    smap = {s: ''.join(map(lambda x: mmap.get(x)[0],
                       np.roll(months, -s)[-np.diff(seasons)[0]:])) for s in seasons}

    # Load PCs and associated K-S normality test p-values
    PC_norm_pvals = pd.read_csv(inpath+'/hist/PC_KS_pvals_{}.csv'.format(v), index_col=0)
    PC_norm_pvals.columns = PC_norm_pvals.columns.astype(int)

    # Plot PC normality
    PC_norm_pvals = PC_norm_pvals.rename(columns=smap)
    ax = PC_norm_pvals.plot(figsize=(12,6))
    ax.set_xlabel('PC')
    ax.set_ylabel('p-value')
    ax.set_title('Kolmogorov-Smirnov normality test: p-values by PC | {}'.format(v))
    ax.set_ylim([0,1])

    return ax.figure


def pc_corrmat(sttgs, inpath, N=4):
    """Generate PC correlation matrix heatmap."""

    seasons = sttgs['seasons']
    smap = {s: ''.join(map(lambda x: mmap.get(x)[0],
                       np.roll(months, -s)[-np.diff(seasons)[0]:])) for s in seasons}

    # Load PC correlation matrices for each season
    PC_corrmat = {s: pd.read_csv(inpath+'/hist/PC_corrmat_s{}.csv'.format(s),
                                 index_col=[0,1], header=[0,1]) for s in seasons}

    # Plot PC correlation matrix
    # Select only first few PCs for each variable
    ix = []
    for level0 in ['clim']+sttgs['variables']:
        if level0 == 'clim':
            ix.append((level0, PC_corrmat[seasons[0]].loc[level0].index[0]))
        else:
            ix.extend([(level0, str(i)) for i in range(N)])

    fig = plt.figure(figsize=(len(seasons)*8+1, 6))
    for i, s in enumerate(seasons):
        ax = plt.subplot2grid((1, len(seasons)), (0, i))
        sns.heatmap(PC_corrmat[s].loc[ix, ix], square=True, vmin=-1, vmax=1, cmap='RdBu', ax=ax)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.set_title(smap[s])
    return fig


def plot_clima(v, hist_clima_m, stoc_clima_m, basemap, cmap='viridis_r'):
    """Plot historic and stochastic climatology maps on the same figure, by month."""

    mmap = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    # Get limits across all subplots as min and max 
    histmax = hist_clima_m.sel(variable=v).max(dim=['month','lon','lat'])['value'].values
    stocmax = stoc_clima_m.sel(variable=v).max(dim=['month','lon','lat'])['value'].values
    histmin = hist_clima_m.sel(variable=v).min(dim=['month','lon','lat'])['value'].values
    stocmin = stoc_clima_m.sel(variable=v).min(dim=['month','lon','lat'])['value'].values
    vmax, vmin = max(histmax, stocmax), min(histmin, stocmin)

    # Set up colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    for m in range(1,13):
        # Calculate axis index for historic (LHS) (assumes 3*8 monthly layout)
        n = (m-1)%4 + 8*((m-1)//4)

        # Make actual plots
        hist_clima_m.sel(variable=v, month=m)['value'].plot(ax=basemap.axes[n], vmin=vmin, vmax=vmax,
                                                            add_colorbar=False, cmap=cmap)
        stoc_clima_m.sel(variable=v, month=m)['value'].plot(ax=basemap.axes[n+4], vmin=vmin, vmax=vmax,
                                                            add_colorbar=False, cmap=cmap)
        # Tweak subplot options
        basemap.axes[n].set_xlabel(None)
        basemap.axes[n].set_ylabel(None)
        basemap.axes[n+4].set_xlabel(None)
        basemap.axes[n+4].set_ylabel(None)
        basemap.axes[n].set_title('Historic | {0}'.format(mmap[m]))
        basemap.axes[n+4].set_title('Stochastic | {0}'.format(mmap[m]))
        plt.setp(basemap.axes[n+4].spines.values(), color='r')
        plt.setp([basemap.axes[n+4].get_xticklines(), basemap.axes[n+4].get_yticklines()], color='r')

    # Add colorbar and title
    basemap.colorbar(sm, cax=basemap.axes[-1])
    basemap.suptitle('Climatology | {}'.format(v))
    return basemap


def plot_quant(v, q, hist_quant_m, stoc_quant_m, basemap, cmap='viridis_r'):
    """Plot historic and stochastic quantile maps on the same figure, by month."""

    mmap = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    # Get limits across all subplots as min and max 
    histmax = hist_quant_m.sel(variable=v, quantile=q).max(dim=['month','lon','lat'])['value'].values
    stocmax = stoc_quant_m.sel(variable=v, quantile=q).max(dim=['month','lon','lat'])['value'].values
    histmin = hist_quant_m.sel(variable=v, quantile=q).min(dim=['month','lon','lat'])['value'].values
    stocmin = stoc_quant_m.sel(variable=v, quantile=q).min(dim=['month','lon','lat'])['value'].values
    vmax, vmin = max(histmax, stocmax), min(histmin, stocmin)

    # Set up colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    for m in range(1,13):
        # Calculate axis index for historic (LHS) (assumes 3*8 monthly layout)
        n = (m-1)%4 + 8*((m-1)//4)

        # Make actual plots
        hist_quant_m.sel(variable=v, month=m, quantile=q)['value'].plot(ax=basemap.axes[n], vmin=vmin, vmax=vmax,
                                                                        add_colorbar=False, cmap=cmap)
        stoc_quant_m.sel(variable=v, month=m, quantile=q)['value'].plot(ax=basemap.axes[n+4], vmin=vmin, vmax=vmax, 
                                                                        add_colorbar=False, cmap=cmap)
        # Tweak subplot options
        basemap.axes[n].set_xlabel(None)
        basemap.axes[n].set_ylabel(None)
        basemap.axes[n+4].set_xlabel(None)
        basemap.axes[n+4].set_ylabel(None)
        basemap.axes[n].set_title('Historic | {0}'.format(mmap[m]))
        basemap.axes[n+4].set_title('Stochastic | {0}'.format(mmap[m]))

        plt.setp(basemap.axes[n+4].spines.values(), color='r')
        plt.setp([basemap.axes[n+4].get_xticklines(), basemap.axes[n+4].get_yticklines()], color='r')

    # Add colorbar and title
    basemap.colorbar(sm, cax=basemap.axes[-1])
    basemap.suptitle('Quantile {0:.2f} | {1}'.format(q, v))

    return basemap


def plot_cdfs(v, qid, hist_quant_m, stoc_quant_m, nrow=3, ncol=8, axdim=4):
    """Plot historic and stochastic quantile maps on the same figure, by month."""

    mmap = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    lon, lat = qtid2ll(int(qid))

    fig = plt.figure(figsize=(ncol*axdim, nrow*axdim))
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot2grid((nrow, ncol), (i, j))

    for m in range(1,13):
        # Calculate axis number for historic (LHS) (assumes 3*8 monthly layout)
        n = (m-1)%4 + 8*((m-1)//4)

        # Extract quantiles
        hvals = hist_quant_m.sel(variable=v, month=m, lon=lon, lat=lat)['value'].values
        hqnts = hist_quant_m.sel(variable=v, month=m, lon=lon, lat=lat)['quantile'].values
        svals = stoc_quant_m.sel(variable=v, month=m, lon=lon, lat=lat)['value'].values
        sqnts = stoc_quant_m.sel(variable=v, month=m, lon=lon, lat=lat)['quantile'].values

        # Make CDF plots
        fig.axes[n].plot(hvals, hqnts, label='Historic')
        fig.axes[n].plot(svals, sqnts, label='Stochastic')
        fig.axes[n].set_title('CDF | {0} '.format(mmap[m]))

        hist = pd.Series(dict(zip(hqnts, hvals)), name='hist')
        stoc = pd.Series(dict(zip(sqnts, svals)), name='stoc')
        qq = pd.concat([hist, stoc], axis=1).dropna()
        qq.plot.scatter('hist', 'stoc', ax=fig.axes[n+4])

        # Calculate axis bounds
        mins, maxs = zip(fig.axes[n+4].get_xlim(), fig.axes[n+4].get_ylim())
        fig.axes[n+4].set_xlim((min(mins), max(maxs)))
        fig.axes[n+4].set_ylim((min(mins), max(maxs)))

        # Add 1:1 line
        fig.axes[n+4].plot((min(mins), max(maxs)), (min(mins), max(maxs)), '0.75')
        fig.axes[n+4].set_title('QQ | {0} '.format(mmap[m]))

        # Tweak subplot options
        if (m-1)//8 < 1:
            fig.axes[n+4].set_xlabel(None)
        fig.axes[n+4].set_ylabel(None)

    fig.suptitle('CDF and QQ plots for {0} | QID {1} | ({2:.3f}, {3:.3f})'.format(v, qid, lon, lat))

    return fig


def plot_ks2(v, ks_m, basemap, cmap='viridis_r'):
    """Plot K-S two sample p-values for similarity of historic and stochastic."""

    mmap = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    # Set limits across all subplots as min and max 
    vmax, vmin = 1, 0

    # Set up colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    for m in range(1,13):
        # Make actual plots
        ks_m.sel(variable=v, month=m)['value'].plot(ax=basemap.axes[m-1], vmin=vmin, vmax=vmax,
                                                    add_colorbar=False, cmap=cmap)
        # Tweak subplot options
        basemap.axes[m-1].set_xlabel(None)
        basemap.axes[m-1].set_ylabel(None)
        basemap.axes[m-1].set_title('{}'.format(mmap[m]))

    # Add colorbar
    basemap.colorbar(sm, cax=basemap.axes[-1])
    basemap.suptitle('Two sample (historic and stochastic) Kolmogorov-Smirnov p-value | {0}'.format(v))
    return basemap


def plot_xcorr(v1, v2, hist_xcorr_m, stoc_xcorr_m, basemap, cmap='RdBu'):
    """Plot historic and stochastic cross-correlation maps on the same figure, by month."""

    mmap = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    # Get limits across all subplots as min and max 
    vmax, vmin = 1, -1

    # Set up colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    for m in range(1,13):
        # Calculate axis index for historic (LHS) (assumes 3*8 monthly layout)
        n = (m-1)%4 + 8*((m-1)//4)

        # Make actual plots
        hist_xcorr_m.sel(variable1=v1, variable2=v2, month=m)['value'].plot(ax=basemap.axes[n], vmin=vmin, vmax=vmax,
                                                                            add_colorbar=False, cmap=cmap)
        stoc_xcorr_m.sel(variable1=v1, variable2=v2, month=m)['value'].plot(ax=basemap.axes[n+4], vmin=vmin, vmax=vmax,
                                                                            add_colorbar=False, cmap=cmap)
        # Tweak subplot options
        basemap.axes[n].set_xlabel(None)
        basemap.axes[n].set_ylabel(None)
        basemap.axes[n+4].set_xlabel(None)
        basemap.axes[n+4].set_ylabel(None)
        basemap.axes[n].set_title('Historic | {0}'.format(mmap[m]))
        basemap.axes[n+4].set_title('Stochastic | {0}'.format(mmap[m]))

        plt.setp(basemap.axes[n+4].spines.values(), color='r')
        plt.setp([basemap.axes[n+4].get_xticklines(), basemap.axes[n+4].get_yticklines()], color='r')

    # Add colorbar and title
    basemap.colorbar(sm, cax=basemap.axes[-1])
    basemap.suptitle('Cross-correlation between {0} and {1}'.format(v1, v2))
    return basemap


def proc(sttgs, inpath, outpath):
    """Convenience function to generate all plots."""

    # Open NetCDF files output by wdaily2
    hist_clima_m = xr.open_dataset(inpath+'/analytics/hist_m_agg_clima.nc')
    stoc_clima_m = xr.open_dataset(inpath+'/analytics/stoc_m_agg_clima.nc')
    hist_quant_m = xr.open_dataset(inpath+'/analytics/hist_m_agg_quant.nc')
    stoc_quant_m = xr.open_dataset(inpath+'/analytics/stoc_m_agg_quant.nc')
    hist_quant_d = xr.open_dataset(inpath+'/analytics/hist_m_day_quant.nc')
    stoc_quant_d = xr.open_dataset(inpath+'/analytics/stoc_m_day_quant.nc')
    hist_xcorr_m = xr.open_dataset(inpath+'/analytics/hist_m_agg_xcorr.nc')
    stoc_xcorr_m = xr.open_dataset(inpath+'/analytics/stoc_m_agg_xcorr.nc')
    ks_m = xr.open_dataset(inpath+'/analytics/m_agg_ks.nc')

    # Plot reference QIDs
    fig = qidrefplot(sttgs)
    plt.savefig(outpath+'/qids_ref.png', dpi=300, bbox_inches='tight')

    # Normality of anomalies
    for v in sttgs['variables']:
        fig = qidnorm(sttgs, inpath, v)
        plt.savefig(outpath+f'/qidnorm_{v}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    # EOFs
    for v in sttgs['variables']:
        kwargs = eof(sttgs, inpath, v)
        fig = splot(sttgs, **kwargs)
        plt.savefig(outpath+f'/EOFs_{v}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    # PC variance explained
    for v in sttgs['variables']:
        fig = pc_varexp(sttgs, inpath, v)
        plt.savefig(outpath+f'/PCvarexp_{v}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    # Normality of PCs
    for v in sttgs['variables']:
        fig = pc_norm(sttgs, inpath, v)
        plt.savefig(outpath+f'/PCnorm_{v}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    # PC correlation matrix
    fig = pc_corrmat(sttgs, inpath)
    plt.savefig(outpath+f'/PCcorrmat.png', dpi=300, bbox_inches='tight')

    # Climatologies
    for v in sttgs['variables']:
        fig = basemap(3, 8, sttgs)
        fig = plot_clima(v, hist_clima_m, stoc_clima_m, fig)
        plt.savefig(outpath+f'/clima_{v}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    # Quantiles
    for v in sttgs['variables']:
        for q in [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]:
            fig = basemap(3, 8, sttgs)
            fig = plot_quant(v, q, hist_quant_m, stoc_quant_m, fig)
            plt.savefig(outpath+f'/quant_{v}_{q}.png', dpi=300, bbox_inches='tight')
        plt.close('all')

    # ECDFs and QQ plots
    for v in sttgs['variables']:
        for qid in sttgs['analytics']['qids_ref']:
            # Plot monthly aggregates
            fig = plot_cdfs(v, qid, hist_quant_m, stoc_quant_m)
            plt.savefig(outpath+f'/ecdfQQ_{v}_{qid}.png', dpi=300, bbox_inches='tight')

            # Plot daily ECDFs/quantiles ========================================================
            fig = plot_cdfs(v, qid, hist_quant_d, stoc_quant_d)
            plt.savefig(outpath+f'/ecdfQQ_{v}_{qid}_daily.png', dpi=300, bbox_inches='tight')

        plt.close('all')

    # K-S similarity
    for v in sttgs['variables']:
        fig = basemap(3, 4, sttgs)
        fig = plot_ks2(v, ks_m, fig)
        plt.savefig(outpath+f'/ks_histstoc_{v}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    # Cross-correlations
    for v1 in sttgs['variables'] + ['mei']:
        for v2 in sttgs['variables']:
            fig = basemap(3, 8, sttgs)
            fig = plot_xcorr(v1, v2, hist_xcorr_m, stoc_xcorr_m, fig)
            plt.savefig(outpath+f'/xcorr_{v1}_{v2}.png', dpi=300, bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    _, sttgsfile, inpath = sys.argv

    # Identify, load and parse settings file
    with open(sttgsfile, 'r') as f:
        sttgs = json.load(f)

    # Generate plots
    proc(sttgs, inpath, inpath+'/plots/')

