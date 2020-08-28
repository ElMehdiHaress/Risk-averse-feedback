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


def catoni(w, X, Y, delta, alpha, valpha):
    """
        Catoni estimator of the gradient

        Parameters
        ----------
        w : d-array, candidate

        X : d-n array, sample

        Y : n-array, sample

        delta : float, a parameter that determines the precision of the gradient estimates

        alpha : float, quantile level

        valpha : float, value-at-risk

        Returns
        -------
        theta : d-array, gradient estimates
        """

    d = len(w)
    n = len(Y)

    ll = stagewise_grad(w, X, Y, alpha, valpha)

    variance = [np.var(ll[k]) for k in range(d)]
    s = [sqrt(variance[k] * n / log(2 / delta)) for k in range(d)]
    for i in range(d):
        if s[i] == 0:
            s[i] = 0.00001
    theta = np.zeros(d)
    for i in range(5):
        xx = [(ll[k] - theta[k]) / s[k] for k in range(d)]
        xx = [np.where(xx[k] >= 0, np.log(1 + xx[k] + xx[k] * xx[k]), -np.log(1 - xx[k] + xx[k] * xx[k])) for k in
              range(d)]
        theta = [theta[k] + (s[k] / n) * sum(xx[k]) for k in range(d)]

    return theta


def rgd(initial_w, X1, Y1, X2, Y2, alpha, delta, eta, max_iterations, max_gradient_descents):
    """
        A stage-wise robust gradient descent (uses the catoni estimator for the gradient)

        Parameters
        ----------
        initial_w : d-array, initial value of w

        X1 : d-n array, first half of the x-sample

        X2 : d-n array, second half of the x-sample used to compute the value at risk

        Y1 : n array, first half of the y-sample

        Y2 : n-array, second half of the y-sample used to compute the value-at-risk

        alpha : float, quantile level

        delta : float, a parameter that determines the precision of the gradient estimates

        eta : float, stepsize for each gradient descent

        max_iterations : int, maximum number of iterations for each gradient descent

        max_gradient_descents : int, maximum number of gradient descents

        Returns
        -------
        ws : d-array, estimate of the optimal candidate
        """

    d = len(initial_w)
    n = len(Y1)

    ws = initial_w

    w_history = []

    S = max_gradient_descents
    T = max_iterations

    for s in range(S):
        X1_s = X1[:, s * int(n / S):(s + 1) * int(n / S)]
        Y1_s = Y1[s * int(n / S): (s + 1) * int(n / S)]
        X2_s = X2[:, s * int(n / S): (s + 1) * int(n / S)]
        Y2_s = Y2[s * int(n / S): (s + 1) * int(n / S)]

        w_0 = ws
        valpha = value_at_risk(loss_function(ws, X2_s, Y2_s), alpha)

        wt = w_0

        for t in range(T):
            gradient = catoni(wt, X1_s, Y1_s, delta, alpha, valpha)
            print(gradient)

            wt = [wt[i] - eta * gradient[i] for i in range(d)]

            w_history += [wt]
            print(wt)

        ws = wt

    return ws
