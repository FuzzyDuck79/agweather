#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
haver

Calculate distance matrix of longitude-latitude arrays
using the Haversine metric.

Mutahar Chalmers, RMS, 2019
"""

import numpy as np


def dmat_haver_1d(lon, lat, radius=6371):
    """
    Haversine 'self' distance matrix from lon, lat arrays.
    Assumes 1D arrays of lon and lat in decimal degrees.

    lon and lat must have the same length n.
    Returns an (n, n) distance matrix in kilometers.

    Sources:
        http://en.wikipedia.org/wiki/Haversine_formula
        http://stackoverflow.com/questions/34502254
    """

    lon, lat = np.array(lon), np.array(lat)

    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    dlon = lon[:, None] - lon
    dlat = lat[:, None] - lat

    d = np.sin(dlat/2)**2 + np.cos(lat[:, None])*np.cos(lat)*np.sin(dlon/2)**2
    return 2*radius*np.arcsin(np.sqrt(d))


def dmat_haver_2d(ll_A, ll_B=None, radius=6371):
    """
    Haversine distance matrix in km from two different (n, 2) shaped
    lon, lat arrays. Assumes both arrays in decimal degrees.

    ll_A has shape (nA, 2), ll_B has shape (nB, 2).
    Returns an (nA, nB) distance matrix in kilometers.

    Sources:
        http://en.wikipedia.org/wiki/Haversine_formula
        http://stackoverflow.com/questions/34502254
    """

    if ll_B is None:
        ll_B = ll_A

    ll_A, ll_B = np.array(ll_A), np.array(ll_B)

    ll_A, ll_B = np.deg2rad(ll_A), np.deg2rad(ll_B)
    dlon = ll_A[:,0][:, None] - ll_B[:,0]
    dlat = ll_A[:,1][:, None] - ll_B[:,1]

    d = np.sin(dlat/2)**2 + np.cos(ll_A[:,1][:, None])*np.cos(ll_B[:,1])*np.sin(dlon/2)**2
    return 2*radius*np.arcsin(np.sqrt(d))

