#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
urggen

Generate global qid-level Uniform Resolution Grids (URG).
Mutahar Chalmers, 2019
"""

import sys
import numpy as np
import xarray as xr
import shapely as shp
import geopandas as gpd

sys.path.append('/mnt/gpfs/backup/agri/code/')
import qtree


def makeURG_DataArray(res, lon_bounds=None, lat_bounds=None):
    """
    Make a URG in xarray DataArray format.

    Parameters
    ----------
        res : float
            Grid resolution in decimal degrees.
        lon_bounds : tuple, optional
            2-tuple of floats containing longitude extent of URG centroids.
            If omitted, defaults to (-180+res/2, 180+res/2).
        lat_bounds : tuple, optional
            2-tuple of floats containing latitude extent of URG centroids.
            If omitted, defaults to (-90+res/2, 90+res/2).

    Returns
    -------
        da : DataArray
            xarray DataArray of qids with longitude and latitude dimensions.
        df : DataFrame
            pandas DataFrame with qid, longitude and latitude.
    """

    if lon_bounds is None:
        lon_args = (-180+res/2, 180+res/2, res)
    else:
        lon_args = lon_bounds + (res,)

    if lat_bounds is None:
        lat_args = (-90+res/2, 90+res/2, res)
    else:
        lat_args = lat_bounds + (res,)

    # Generate arrays of qid centroid coordinates
    lons = np.arange(*lon_args)
    lats = np.arange(*lat_args)

    # Generate actual qids and store in DataArray
    qids = np.array([qtree.cyll2qid(lon, lat, res)
                     for lat in lats for lon in lons]
                     ).reshape((lats.size, lons.size))
    da = xr.DataArray(qids, coords={'lon': lons, 'lat': lats},
                      dims=['lat','lon'], name='qid')

    # Convert to DataFrame
    df = da.to_dataframe()
    return da, df


def makeURG_GeoDataFrame(res, lon_bounds=None, lat_bounds=None):
    """
    Make a URG in geopandas GeoDataFrame format.

    Parameters
    ----------
        res : float
            Grid resolution in decimal degrees.
        lon_bounds : tuple, optional
            2-tuple of floats containing longitude extent of URG centroids.
            If omitted, defaults to (-180+res/2, 180+res/2).
        lat_bounds : tuple, optional
            2-tuple of floats containing latitude extent of URG centroids.
            If omitted, defaults to (-90+res/2, 90+res/2).

    Returns
    -------
        gdf : GeoDataFrame
            geopandas GeoDataFrame with qid and geometry columns.
    """

    if lon_bounds is None:
        lon_args = (-180+res/2, 180+res/2, res)
    else:
        lon_args = lon_bounds + (res,)

    if lat_bounds is None:
        lat_args = (-90+res/2, 90+res/2, res)
    else:
        lat_args = lat_bounds + (res,)

    # Generate arrays of qid centroid coordinates
    lons = np.arange(*lon_args)
    lats = np.arange(*lat_args)

    # Generate qid Polygons and store in GeoDataFrame
    gdf_raw = [{'qid': qtree.cyll2qid(lon, lat, res),
                'lon': lon, 'lat': lat,
                'geometry': shp.geometry.Polygon([(lon+res/2, lat+res/2),
                                                  (lon+res/2, lat-res/2),
                                                  (lon-res/2, lat-res/2),
                                                  (lon-res/2, lat+res/2)])}
               for lat in lats for lon in lons]
    gdf = gpd.GeoDataFrame(gdf_raw)
    gdf.crs = {'init': 'epsg:4326'}
    return gdf

