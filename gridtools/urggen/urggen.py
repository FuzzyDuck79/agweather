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


def makeURG_DataArray(res):
    """
    Make a global qid-grid in xarray DataArray format.

    Parameters
    ----------
        res : float
            Grid resolution in decimal degrees.

    Returns
    -------
        da : DataArray
            xarray DataArray of qids with longitude and latitude dimensions.
        df : DataFrame
            pandas DataFrame with qid, longitude and latitude.
    """

    # Generate arrays of qid centroid coordinates
    lons = np.arange(-180+res/2, 180+res/2, res)
    lats = np.arange(-90+res/2, 90+res/2, res)

    # Generate actual qids and store in DataArray
    qids = np.array([qtree.cyll2qid(lon, lat, res)
                     for lat in lats for lon in lons]
                     ).reshape((lats.size, lons.size))
    da = xr.DataArray(qids, coords={'lon': lons, 'lat': lats},
                      dims=['lat','lon'], name='qid')

    # Convert to DataFrame
    df = da.to_dataframe()
    return da, df


def makeURG_GeoDataFrame(res):
    """
    Make a global qid-grid in xarray DataArray format.

    Parameters
    ----------
        res : float
            Grid resolution in decimal degrees.

    Returns
    -------
        gdf : GeoDataFrame
            geopandas GeoDataFrame with qid and geometry columns.
    """

    # Generate arrays of qid centroid coordinates
    lons = np.arange(-180+res/2, 180+res/2, res)
    lats = np.arange(-90+res/2, 90+res/2, res)

    # Generate qid Polygons and store in GeoDataFrame
    gdf_raw = [{'qid': qtree.cyll2qid(lon, lat, res),
                'geometry': shp.geometry.Polygon([(lon+res/2, lat+res/2),
                                                  (lon+res/2, lat-res/2),
                                                  (lon-res/2, lat-res/2),
                                                  (lon-res/2, lat+res/2)])}
               for lat in lats for lon in lons]
    gdf = gpd.GeoDataFrame(gdf_raw)
    gdf.crs = {'init': 'epsg:4326'}
    return gdf

