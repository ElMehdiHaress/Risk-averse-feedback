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


def emp_grad(w, X, Y, alpha, valpha):
    """
        empirical gradient-risk estiamtes

        Parameters
        ----------
        w : d-array, candidate

        X : d-n array, sample

        Y : n array, sample

        alpha : float, quantile level

        valpha : float, value-at-risk

        Returns
        -------
        d-array, gradient-risk estimates
        """

    d = len(X)
    n = len(Y)
    g = np.dot(w, X) - Y
    g = [(1 / alpha) * np.where(g >= 0, X[i], -X[i]) for i in range(d)]
    g = np.where(loss_function(w, X, Y) >= valpha, g, 0)

    return [sum(g[i])/n for i in range(d)]


def erm(initial_w, X1, Y1, X2, Y2, alpha, eta,  max_iterations, max_gradient_descents):
    """
        A stage-wise gradient descent (using empirical risk minimization)

        Parameters
        ----------
        initial_w : d-array, initial value of w

        X1 : d-n array, first half of the x-sample

        X2 : d-n array, second half of the x-sample used to compute the value at risk

        Y1 : n array, first half of the y-sample

        Y2 : n-array, second half of the y-sample used to compute the value-at-risk

        alpha : float, quantile level

        eta : float, stepsize for each gradient descent

        max_iterations : int, maximum number of iterations for each gradient descent

        max_gradient_descents : int, maximum number of gradient descents

        Returns
        -------
        ws : d-array, estimate of the optimal candidate
        """

    d = len(initial_w)
    ws = initial_w

    w_history = []
    S = max_gradient_descents
    T = max_iterations

    for s in range(S):
        w_0 = ws
        valpha = value_at_risk(loss_function(ws, X2, Y2), alpha)

        wt = w_0

        for t in range(T):
            gradient = stagewise_grad(wt, X1, Y1, alpha, valpha)

            wt = [wt[i] - eta * np.mean(gradient[i]) for i in range(d)]

            w_history += [wt]
            print(wt)

        ws = wt

    return ws
