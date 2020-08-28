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


def robust_cvar_estimation(w, X_1, Y_1, X_2, Y_2, alpha, delta):
    """
         Robust cvar estimate (using the Catoni estimator)

         Parameters
         ----------
         w : d-array, candidate

         X_1 : d-n array, sample

         Y_1 : n array, sample

         X_2 : d-n array, sample used to estimate the value-at-risk

         Y_2 : n-array, sample used to estimate the value-at-risk

         alpha : float, quantile level

         delta : float, a parameter that determines the precision of the gradient estimates


         Returns
         -------
         theta : float, estimate of the cvar at w
         """

    n = len(Y_1)
    valpha = value_at_risk(loss_function(w, X_2, Y_2), alpha)
    loss = modified_loss(w, valpha, X_1, Y_1, alpha)

    var = np.var(loss)
    if var == 0:
        var = 10 ** (-5)

    s = sqrt(var * n / log(2 / delta))  # scaling parameter
    theta = 0

    for k in range(5):
        ll = (loss - theta) / s
        ll = np.where(ll >= 0, np.log(1 + ll + ll * ll), -np.log(1 - ll + ll * ll))
        theta = theta + (s / n) * sum(ll)
    return theta


def sgd_grad(w, v, x, y, alpha):
    """
         Computing the gradient for the sgd

         Parameters
         ----------
         w, v : d-array, float, candidates

         x : d-array

         y : float

         alpha : float, quantile level

         Returns
         -------
         d+1-array : gradient of the loss function at w, v
         """

    d = len(w)
    if loss_function(w, x, y) > v:
        return [(1 / alpha) * g for g in gradient_loss(w, x, y)] + [(alpha - 1)/alpha]
    if loss_function(w, x, y) == v:
        t = np.random.uniform(0, 1)
        return [(1 / alpha) * t * g for g in gradient_loss(w, x, y)] + [(alpha - t)/alpha]
    if loss_function(w, x, y) < v:
        return [0 for k in range(d)] + [1]


def sgd(initial_w, initial_v, X, Y, alpha, beta):
    """
         A stochastic gradient descent algorithm

         Parameters
         ----------
         initial_w : d-array, initial value of w

         initial_v : 1-array, initial value of v

         X : d-n array,

         Y : d-n array

         alpha : float, quantile level

         beta : float, stepsize of every sgd
             quantile level

         Returns
         -------
         w_average + v_average : d-array, average of the estimates
         """

    d = len(initial_w)
    wk, vk = initial_w, initial_v
    w_history = []
    v_history = []
    for k in range(len(Y)):
        y = Y[k]
        x = X[:, k]
        grad = sgd_grad(wk, vk, x, y, alpha)
        print(grad)
        wk = [wk[i] - beta * grad[i] for i in range(0, d)]
        print(wk)
        vk = vk - beta * grad[d]
        w_history.append(wk)
        v_history.append(vk)
    w_average = [sum(np.array(w_history)[:, i])/len(Y) for i in range(d)]
    v_average = sum(v_history)/len(Y)
    return w_average + [v_average]


def robustified_sgd(initial_w, initial_v, X1, Y1, X2, Y2, alpha, delta, k, beta):
    """
        A robust stochastic gradient descent (uses the robust esitmates for validation)

        Parameters
        ----------
        initial_w : d-array, initial value of w

        initial_v : 1-array, initial value of v

        X1 : d-n array, first half of the x-sample

        Y1 : n array, first half of the y-sample

        X2 : d-n array, second half of the x-sample used to compute the value at risk

        Y2 : n-array, second half of the y-sample used to compute the value-at-risk

        alpha : float, quantile level

        delta : float, a parameter that determines the precision of the gradient estimates

        k : int, number of parallel sgd we run

        beta : float, stepsize of every sgd
            quantile level

        Returns
        -------
        w : d-array, estimate of the optimal candidate
        """

    n = len(Y2)
    d = len(initial_w)
    cost = 10 ** 5
    w = np.zeros(d)
    for i in range(k):
        x = X1[:, i * int(n / k): (i + 1) * int(n / k)]
        y = Y1[i * int(n / k): (i + 1) * int(n / k)]
        candidate_w = sgd(initial_w, initial_v, x, y, alpha, beta)[0:d]
        cvar = robust_cvar_estimation(candidate_w, X2[:, 0:int(n/2)], Y2[0:int(n/2)], X2[:, int(n/2):n], Y2[int(n/2):n],
                                      alpha, delta)
        if cvar <= cost:
            cost = cvar
            w = candidate_w
    return w
