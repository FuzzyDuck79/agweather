#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kumafuncs

Functions for working with the Kumaraswamy distribution.
Mutahar Chalmers, RMS, 2019

2019-07-10 First version
"""

import numpy as np
import scipy.special as ss
import scipy.optimize as so


def fkuma(x, a, b, lb=0, ub=1):
    """PDF of Kumaraswamy distribution."""

    xx = (x-lb)/(ub-lb)
    return a*b*xx**(a-1)*(1-xx**a)**(b-1) / (ub-lb)


def Fkuma(x, a, b):
    """CDF of Kumaraswamy distribution."""

    return 1 - (1-x**a)**b


def Qkuma(q, a, b):
    """Quantile function of Kumaraswamy distribution."""

    return (1 - (1 - q)**(1/b))**(1/a)


def kuma_mn(n, a, b):
    """Raw moments of Kumaraswamy distribution."""

    return b*ss.beta(1+n/a, b)


def fitkuma_m(mu, sigma, wt_mu=1, wt_sigma2=1, tol=1e-6):
    """Fit Kumaraswamy distribution by matching moments."""

    def objkuma(ab, muvar):
        """Objective function for use in optimisation."""

        mean = kuma_mn(1, *ab)
        variance = kuma_mn(2, *ab) - mean**2
        return wt_mu*(mean - muvar[0])**2 + wt_sigma2*(variance - muvar[1])**2

    soln = so.minimize(objkuma, x0=[1,1], args=[mu, sigma**2], method='BFGS',
                       tol=tol)
    a, b = soln.x
    return a, b


def fitkuma_q(x, tol=1e-6):
    """Fit Kumaraswamy distribution by matching quantiles."""

    def objkuma(ab):
        """Objective function for use in optimisation."""

        ref_cumprobs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])

        ref_quantiles = np.quantile(x, ref_cumprobs, axis=0)
        return np.sum((Fkuma(ref_quantiles, *ab) - ref_cumprobs)**2)

    soln = so.minimize(objkuma, x0=[1,1], method='BFGS', tol=tol)
    a, b = soln.x
    return a, b


def kumagrid(mean_min, mean_max, mean_delta, cv_min, cv_max, cv_delta):
    """
    Generate a 3D grid of Kumaraswamy a and b parameters as a function of
    a range of combinations of mean and CV.
    """

    # Define arrays for means and CVs
    means = np.arange(mean_min, mean_max, mean_delta)
    cvs = np.arange(cv_min, cv_max, cv_delta)

    # Calculate Kumaraswamy as and bs
    kumagrid_np = np.array([[fitkuma_m(mean, mean*cv) for cv in cvs]
                            for mean in means])
    return kumagrid_np

