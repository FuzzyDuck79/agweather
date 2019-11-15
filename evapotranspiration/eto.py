#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eto

Calculate reference evapotranspiration using the Penman-Monteith or Hargreaves
methods. This module is a wrapper around the fao module from Mark Richards'
PyETo library (https://github.com/woodcrafty/PyETo).

Mutahar Chalmers, 2018-9
"""

import numpy as np
import evapotranspiration.fao as fao


def hargreaves(tmax, tmin, doys, lat):
    """
    Calculate reference evapotranspiration using 1985 Hargreaves equation
    for a time series of weather data with standard structure.

    Parameters
    ----------
        tmax : numpy array or pandas Series
           Daily Tmax (degrees C).
        tmin : numpy array or pandas Series
            Daily Tmin (degrees C).
        doys : numpy array or pandas Series
            Day of the year labels corresponding to tmax and tmin.
        lat : float
            Latitude in decimal degrees.

    Returns
    -------
        eto : numpy array or pandas Series
            Reference evapotranspiration (mm).
    """

    # Convert latitude to radians
    lat = np.deg2rad(lat)

    # Estimate Tmean
    tmean = (tmin + tmax)/2

    # Calculate the following quantities for doys in [1, 2, ...,  365, 366] only
    # Solar declination, sunset hour angle, inverse relative Earth-Sun distance
    sol_dec = [fao.sol_dec(doy) for doy in range(1, 367)]
    sha = [fao.sunset_hour_angle(lat, sdec) for sdec in sol_dec]
    ird = [fao.inv_rel_dist_earth_sun(doy) for doy in range(1, 367)]

    # Calculate extra-terrestrial radiation and map from doy
    et_rad = [fao.et_rad(lat, *ssi) for ssi in zip(sol_dec, sha, ird)]
    et_rad_doy = {doy: etr for doy, etr in zip(range(1, 367), et_rad)}
    et_rad_all = np.array([et_rad_doy[doy] for doy in doys])

    return fao.hargreaves(tmin, tmax, tmean, et_rad_all)


def penmanmonteith(tmax, tmin, rs, rh, u2, doys, lat, elevation=0):
    """
    Calculate reference evapotranspiration using Penman-Monteith/FAO56 method
    for a time series of weather data with standard structure.

    Parameters
    ----------
        tmax : numpy array or pandas Series
            Daily Tmax (degrees C).
        tmin : numpy array or pandas Series
            Daily Tmin (degrees C).
        rs   : numpy array or pandas Series
            Daily Rs (MJ/m2).
        rh   : numpy array or pandas Series
            Daily relative humidity (%).
        u2   : numpy array or pandas Series
            Daily wind speed at 2m (m/s).
        doys : numpy array or pandas Series
            Day of the year labels corresponding to tmax and tmin.
        lat : float
            Latitude in decimal degrees.
        elevation : float, optional
            Elevation above mean sea level (m).

    Returns
    -------
        eto : numpy array or pandas Series
            Reference evapotranspiration (mm).
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

