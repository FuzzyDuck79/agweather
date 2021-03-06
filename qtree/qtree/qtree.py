#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qtree

Quadtree conversion tools
Mutahar Chalmers, 2018

2019-10-12 Added improved functions ll2qid and qid2ll.
2019-07-12 Renamed variables. Option to print details.
2018-11-06 Modified to incorporate basic input validation for ll2qtid.

"""

import numpy as np


def ll2qid(lon, lat, res_target, verbose=False):
    """
    Simplifed function which calculates qid based on only on resolution.

    Assume that longitude and latitude origin is always (0, 0).

    The first step classifies a point into one of the main quadrants on the
    Earth's surface. It then shifts the origin to a point at half the current
    resolution (res/2, res/2) in that quadrant. The initial |resolution| is
    chosen such that it is:
      (a) larger by a factor of a power of 2 than the grid resolution
      (b) >=180 degrees

    We automate this evaluation in this function.
    """

    # Check input lon and lat
    if lon>180 or lon<-180:
        print('Longitude must be between -180 and 180')
        return None
    if lat>=90 or lat<=-90:
        print('Latitude must be strictly between -90 and 90')
        return None

    # Determine resolution of top-level quadrants, given target resolution
    i_max = int(np.ceil(np.log2(180/res_target)))
    res = res_target * 2**i_max

    i, qid, origin_lon, origin_lat = 0, 0, 0, 0
    while res >= res_target:
        delta = 4**(i_max-i)
        if lon >= origin_lon and lat >= origin_lat:
            # Do nothing to the qid
            origin_lon, origin_lat = (origin_lon+res/2, origin_lat+res/2)
        elif lon < origin_lon and lat >= origin_lat:
            qid = qid + delta
            origin_lon, origin_lat = (origin_lon-res/2, origin_lat+res/2)
        elif lon < origin_lon and lat < origin_lat:
            qid = qid + 2*delta
            origin_lon, origin_lat = (origin_lon-res/2, origin_lat-res/2)
        elif lon >= origin_lon and lat < origin_lat:
            qid = qid + 3*delta
            origin_lon, origin_lat = (origin_lon+res/2, origin_lat-res/2)
        else:
            print(f'Error:\n lon: {lon}\n lat: {lat}')
            return None

        # Halve the resolution and update counter
        res /= 2
        i += 1

    if verbose:
        print(f'({lon}, {lat}) -> {qid}')
        print(f'Resolution={2*res} | centroid=({origin_lon}, {origin_lat})')
    return qid


def qid2ll(qid, res_target, verbose=False):
    """
    Converts quadtree id to the (lon, lat) of the centroid of the quadcell.
    """

    qid0, lon, lat, delta = qid*1, 0, 0, res_target/2

    while delta <= 180:
        qid, mod = divmod(qid, 4)
        if mod == 0:
            lon += delta
            lat += delta
        elif mod == 1:
            lon -= delta
            lat += delta
        elif mod == 2:
            lon -= delta
            lat -= delta
        elif mod == 3:
            lon += delta
            lat -= delta
        delta *= 2

    if verbose:
        print(f'{qid0} -> ({lon}, {lat})')
        print(f'Resolution={res_target}')
    return lon, lat


# --- Older functions, preserved for backwards compatibility -----------------
def ll2qtid(lon, lat, level=11, offset=128, origin_lon=0, origin_lat=0, verbose=False):
    """Converts (lon, lat) to quadtree id."""

    qtid = 0

    if lon>180 or lon<-180:
        print('Longitude must be between -180 and 180')
        return None
    if lat>=90 or lat<=-90:
        print('Latitude must be strictly between -90 and 90')
        return None

    for i in range(level):
        delta = 4**(level-i-1)
        if lon >= origin_lon and lat >= origin_lat:
            # Do nothing to the qtid
            origin_lon, origin_lat = (origin_lon+offset, origin_lat+offset)
        elif lon < origin_lon and lat >= origin_lat:
            qtid = qtid + delta
            origin_lon, origin_lat = (origin_lon-offset, origin_lat+offset)
        elif lon < origin_lon and lat < origin_lat:
            qtid = qtid + 2*delta
            origin_lon, origin_lat = (origin_lon-offset, origin_lat-offset)
        elif lon >= origin_lon and lat < origin_lat:
            qtid = qtid + 3*delta
            origin_lon, origin_lat = (origin_lon+offset, origin_lat-offset)
        else:
            print('Error:\n lon: {1}\n lat: {2}'.format(lon, lat))
            return None
        offset /= 2

    if verbose:
        print(f'({lon}, {lat}) -> {qtid}')
        print(f'Level={level} | resolution={offset*4} | centroid=({origin_lon}, {origin_lat})')
    return qtid


def qtid2ll(qtid, level=11, offset=128, origin_lon=0, origin_lat=0, verbose=False):
    """Converts quadtree id to the (lon, lat) of the centroid of the quadcell."""

    qtid0, lon, lat, delta = qtid*1, origin_lon, origin_lat, offset*2**(1-level)

    for i in range(level):
        qtid, mod = divmod(qtid, 4)
        if mod == 0:
            lon += delta
            lat += delta
        elif mod == 1:
            lon -= delta
            lat += delta
        elif mod == 2:
            lon -= delta
            lat -= delta
        elif mod == 3:
            lon += delta
            lat -= delta
        delta *= 2

    if verbose:
        print(f'{qtid0} -> ({lon}, {lat})')
        print(f'Level={level} | offset={offset} | origin=({origin_lon}, {origin_lat})')
    return lon, lat

