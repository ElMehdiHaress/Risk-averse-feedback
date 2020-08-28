import numpy as np
from matplotlib import pyplot as plt
from math import *
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy import integrate
from scipy import special
from numpy import median
from numpy import linspace
from copy import deepcopy


def loss_function(w, x, y):
    """
        squared loss function

        Parameters
        ----------
        w : d-array, candidate

        x : d-n array, sample

        y : n array, sample

        Returns
        -------
        n-array, loss function at w
        """

    return (np.dot(w, x) - y) ** 2


def modified_loss(w, v, x, y, alpha):
    """
            modified loss function for the joint CVaR minimization

            Parameters
            ----------
            w : d-array, candidate

            v : 1-array

            x : d-n array, sample

            y : n array, sample

            alpha : float, quantile level

            Returns
            -------
            n-array, modified loss function at w
            """

    ll = loss_function(w, x, y)
    return np.where(ll >= v, v + (1 / alpha) * (ll - v), v)


def gradient_loss(w, x, y):
    """
        gradient of the loss function

        Parameters
        ----------
        w : d-array, candidate

        x : d-array, one value from the sample

        y : float, one value from the sample

        Returns
        -------
        d-array, gradient of the loss function at w
        """

    return [2 * x[i] * (np.dot(w, x) - y) for i in range(len(w))]


def stagewise_grad(w, X, Y, alpha, valpha):
    """
        the gradient loss used in the stage-wise gradient descent (where the value-at-risk is fixed)

        Parameters
        ----------
        w : d-array, candidate

        X : d-n array, sample

        Y : n array, sample

        alpha : float, quantile level

        valpha : float, value-at-risk

        Returns
        -------
        d-array, gradient loss at w
        """

    d = len(X)
    g = [(1 / alpha) * 2 * X[i] * (np.dot(w, X) - Y) for i in range(d)]

    return np.where(loss_function(w, X, Y) >= valpha, g, 0)


def value_at_risk(L, alpha):
    """
            estimate of the value at risk at level alpha

            Parameters
            ----------
            L: n-array, loss function

            alpha : quantile level

            Returns
            -------
            float, VaR
            """

    L.sort()
    return L[int(len(L) * (1 - alpha))]
