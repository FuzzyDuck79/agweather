#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cyqtree

Cython-based Quadtree conversion tools
Mutahar Chalmers, RMS, 2019

"""


def cyll2qtid(float lon, float lat, int level=11, float offset=128,
            float origin_lon=0, float origin_lat=0, bint warn=True):
    """Converts (lon, lat) to quadtree id."""

    cdef int qtid, delta, i

    qtid = 0

    if lon>=180 or lon<=-180:
        print('Longitude must be between -180 and 180')
        return None
    if lat>=90 or lat<=-90:
        print('Latitude must be between -90 and 90')
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

    if warn:
        print(f'({lon}, {lat}) -> {qtid}')
        print(f'Level={level} | resolution={offset*4} | centroid=({origin_lon}, {origin_lat})')
    return qtid


def cyqtid2ll(int qtid, int level=11, float offset=128,
              float origin_lon=0, float origin_lat=0, bint warn=True):
    """Converts quadtree id to the (lon, lat) of the centroid of the quadcell."""

    cdef int i, mod, qtid0
    cdef float lon, lat, delta
    qtid0, lon, lat, delta = qtid*1, origin_lon, origin_lat, offset*2.0**(1-level)

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

    if warn:
        print(f'{qtid0} -> ({lon}, {lat})')
        print(f'Level={level} | offset={offset} | origin=({origin_lon}, {origin_lat})')
    return lon, lat

