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
    Sources:
        http://en.wikipedia.org/wiki/Haversine_formula
        http://stackoverflow.com/questions/34502254

    Parameters
    ----------
        lon : numpy array
            Longitude in decimal degrees.
        lat : numpy array
            Latitude in decimal degrees.
        radius : float
            Radius of the Earth in km.

    Returns
    -------
        dmat : numpy array
           Distance matrix in km.
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

    Sources:
        http://en.wikipedia.org/wiki/Haversine_formula
        http://stackoverflow.com/questions/34502254

    Parameters
    ----------
        ll_A : numpy array of shape (nA, 2)
            Longitude, latitude pairs in decimal degrees.
        ll_B : numpy array of shape (nB, 2)
            Longitude, latitude pairs in decimal degrees.
        radius : float
            Radius of the Earth in km.

    Returns
    -------
        dmat : numpy array of shape (nA, nB)
           Distance matrix in km.
    """

    if ll_B is None:
        ll_B = ll_A

    ll_A, ll_B = np.array(ll_A), np.array(ll_B)

    ll_A, ll_B = np.deg2rad(ll_A), np.deg2rad(ll_B)
    dlon = ll_A[:,0][:, None] - ll_B[:,0]
    dlat = ll_A[:,1][:, None] - ll_B[:,1]

    d = np.sin(dlat/2)**2 + np.cos(ll_A[:,1][:, None])*np.cos(ll_B[:,1])*np.sin(dlon/2)**2
    return 2*radius*np.arcsin(np.sqrt(d))

