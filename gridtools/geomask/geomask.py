#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geomask

Generate raster masks for each country and region at arbitrary grid resolution.
Mutahar Chalmers, 2019
"""

import numpy as np
import shapely.vectorized as shpv
import xarray as xr


def qidmask(poly, res, buff):
    """
    Generate raster mask from rectangular lon, lat grid resolution
    and Shapely Polygon or MultiPolygon object.
    """

    # Generate global lon and lat arrays, and meshgrid for vectorisation
    lons = np.arange(-180+res/2, 180+res/2, res)
    lats = np.arange(-90+res/2, 90+res/2, res)
    llons, llats = np.meshgrid(lons, lats)

    # Generate the mask, transpose, convert to DataArray and return it
    mask = shpv.contains(poly.buffer(buff), llons, llats)
    return xr.DataArray(mask, dims=['lat','lon'],
                        coords={'lon': lons, 'lat': lats}, name='mask')

