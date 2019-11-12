#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eto

Calculate reference evapotranspiration using the Penman-Monteith or Hargreaves
methods. This script relies on the PyETo (https://github.com/woodcrafty/PyETo)
by Mark Richards, and is essentially a wrapper to simplify the calculation of
ETo for gridded historic and stochastic weather data.

Input data is a single weather file in csv or Parquet format, with standard
structure as follows:

    Index: year, month, day, doy
    Columns:  pre (mm), Tmax (degrees C), Tmin (degrees C)
    Optional columns for Penman-Monteith: RH (%), Rs (MJ/m2), u2 (m/s)

Mutahar Chalmers, RMS, 2018-9
"""

import sys
import numpy as np
import pandas as pd

sys.path.append('/mnt/gpfs/backup/agri/code/eto/PyETo')
sys.path.append('/mnt/gpfs/backup/agri/code/qtree')
from pyeto import fao
from qtree import qtid2ll # cy


def calc_ETo_Hargreaves(tmax, tmin, doys, lat):
    """
    Calculate reference evapotranspiration using 1985 Hargreaves equation
    for a time series of weather data with standard structure.

        tmax :: ndarray or pd.Series of daily Tmax (degrees C)
        tmin :: ndarray or pd.Series of daily Tmin (degrees C)
        doys :: ndarray or pd.Series of days of the year corresponding
                to tmax and tmin, and of the same length
    """

    # Convert latitude to radians
    lat = np.deg2rad(lat)

    # Estimate Tmean
    tmean = (tmin + tmax)/2

    # Calculate the following quantities for doys in [1, 2, ...,  365, 366] only
    # Solar declination, sunset hour angle and inverse relative Earth-Sun distance
    sol_dec = [fao.sol_dec(doy) for doy in range(1, 367)]
    sha = [fao.sunset_hour_angle(lat, sdec) for sdec in sol_dec]
    ird = [fao.inv_rel_dist_earth_sun(doy) for doy in range(1, 367)]

    # Calculate extra-terrestrial radiation and map from doy
    et_rad = [fao.et_rad(lat, *ssi) for ssi in zip(sol_dec, sha, ird)]
    et_rad_doy = {doy: etr for doy, etr in zip(range(1, 367), et_rad)}
    et_rad_all = np.array([et_rad_doy[doy] for doy in doys])

    return fao.hargreaves(tmin, tmax, tmean, et_rad_all)


def calc_ETo_PenmanMonteith(tmax, tmin, rs, rh, u2, doys, lat, elevation=0):
    """
    Calculate reference evapotranspiration using Penman-Monteith/FAO56 method
    for a time series of weather data with standard structure.

        tmax :: ndarray or pd.Series of daily Tmax (degrees C)
        tmin :: ndarray or pd.Series of daily Tmin (degrees C)
        rs   :: ndarray or pd.Series of daily Rs (MJ/m2)
        rh   :: ndarray or pd.Series of daily relative humidity (%)
        u2   :: ndarray or pd.Series of daily wind speed at 2m (m/s)
        doys :: ndarray or pd.Series of days of the year corresponding
                to tmax and tmin, and of the same length
    """

    # Convert latitude to radians
    lat = np.deg2rad(lat)

    # Calculate actual and (mean) saturated vapour pressures
    avp = fao.avp_from_rhmean(fao.svp_from_t(tmin), fao.svp_from_t(tmax), rh)
    svp = (fao.svp_from_t(tmin) + fao.svp_from_t(tmax))/2

    # Calculate the following quantities for doys in [1, 2, ...,  365] only
    # Solar declination, sunset hour angle and inverse relative Earth-Sun distance
    sol_dec = [fao.sol_dec(doy) for doy in range(1, 367)]
    sha = [fao.sunset_hour_angle(lat, sdec) for sdec in sol_dec]
    ird = [fao.inv_rel_dist_earth_sun(doy) for doy in range(1, 367)]

    # Calculate extra-terrestrial radiation
    et_rad = [fao.et_rad(lat, *ssi) for ssi in zip(sol_dec, sha, ird)]

    # Calculate clear sky radiation from elevation and extraterrestrial radiation
    csrs = {doy: fao.cs_rad(elevation, etr)
            for doy, etr in zip(range(1, 367), et_rad)}

    # Generate clear sky radiation for full dataset
    cs_rad = np.array([csrs[doy] for doy in doys])

    # Calculate net outgoing longwave radiation
    no_lw_rad = fao.net_out_lw_rad(tmin+273.15, tmax+273.15, rs, cs_rad, avp)

    # Calculate net incoming shortwave (solar) radiation
    ni_sw_rad = fao.net_in_sol_rad(rs)

    # Net radiation
    net_rad = fao.net_rad(ni_sw_rad, no_lw_rad)

    # Delta svp at mean temperature
    d_svp = fao.delta_svp((tmin + tmax)/2)

    # Psychrometric constant
    atmos_pres = fao.atm_pressure(elevation)
    psy = fao.psy_const(atmos_pres)

    # Calculate reference evapotranspiration using mean temperature
    t = (tmin+273.15 + tmax+273.15)/2

    return fao.fao56_penman_monteith(net_rad, t, u2, svp, avp, d_svp, psy)


def main(argv=None):
    """
    Main routine. Expects the following arguments:
        qid, qid_level, qid_offset, elev, fformat, inpath, outpath
    """

    if argv is None:
        argv = sys.argv
        _, qid, qid_level, qid_offset, elev, fformat, inpath, outpath = sys.argv
    else:
        qid, qid_level, qid_offset, elev, fformat, inpath, outpath = argv

    # Load weather data
    if fformat == 'csv': # Specify index_col assuming [year, month, day, doy]
        data = pd.read_csv(inpath+'/{0}.csv'.format(qid), index_col=[0,1,2,3])
    elif fformat == 'parquet':
        data = pd.read_parquet(inpath+'/{0}.parquet'.format(qid))
    else:
        print('File format must be csv or parquet.')
        return None

    # Extract latitude from qid and doys for use in ETo calculation
    _, lat = cyqtid2ll(int(qid), level=qid_level, offset=qid_offset)
    doys = data.index.get_level_values('doy')

    # Define columns variable in order to add eto to it
    cols = data.columns.tolist()

    # Determine method to use and do calculation
    if len(set(('tmax','tmin','rs','rh','u2')).intersection(cols)) == 5:
        pcols = [data[v] for v in ['tmax', 'tmin', 'rs', 'rh', 'u2']] + [doys]
        data['eto'] = calc_ETo_PenmanMonteith(*pcols, lat, float(elev))
    else:
        hcols = [data[v] for v in ['tmax', 'tmin']] + [doys]
        data['eto'] = calc_ETo_Hargreaves(*hcols, lat).astype(np.float32)
    data['eto'] = data['eto'].astype(np.float32)
    cols.append('eto')

    # Write to file
    if fformat == 'csv':
        data[cols].to_csv(outpath+'/{0}.csv'.format(qid))
    else:
        data[cols].to_parquet(outpath+'/{0}.parquet'.format(qid))

if __name__ == '__main__':
    sys.exit(main())

