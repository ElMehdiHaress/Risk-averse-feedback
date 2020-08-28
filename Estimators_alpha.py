"""
Estimators : Empirical, Catoni, Median of means, Trimmed mean
 Random truncation for u=empirical second moment and for u=true second moment

Data distributions:
  - Normal (with mean=0, sd = 1.5, 2.2, 2.4)
  - Log-normal (with log-mean=0, log-sd = 1.25, 1.75, 1.95)
  - Pareto (a=3,xm= 4.1,6,6.5)

The parameters are chosen such that the inter quartile range is the same in each
setting

"""

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
import pandas as pd


# Defining a function that returns the results of one setting (setting = parameters fixed + method fixed) :
def setting_alpha(string_estimator, sigma_normal, sigma_lognormal, x_m, a, max_trials, min_alpha,
            max_alpha, alpha_size, n, u):
    """
         Comparing the evolution of the excess risk of the three distributions (normal, lognormal, pareto) for one
         estimator over many trials with respect to alpha

         Parameters
         ----------
         string_estimator : string, name of the estimator
            empirical, catoni, median_means, trimmed_mean, random_trunc
         sigma_normal : float
             s.d of the normal distribution
         sigma_lognormal : float
            s.d of the lognormal distribution
         x_m, a : float, float
             Pareto parameters
         max_trials : int
             maximum number of trials
         min_alpha : float
             smallest alpha we want to consider
         max_alpha : float
             largest alpha we want to consider
         alpha_size : int
             number of alpha points we want to consider
         n: int
             sample size
         u : string or int
             u = 0 if the estimator isn't random_trunc
             u = 'empirical_2nd_moment' if we want to use the empirical variance for the random truncation estimator
             u = 'true_2nd_moment'  if we want to use the true variance for the random truncation estimator

         Returns
         -------
         3-(samples_number +1) array
         Each line corresponds to the results of one distribution
         The array has the form :
         [['normal', results_normal],['lognormal', results_lognormal],['pareto', results_pareto]
         """

    estimator = dic[string_estimator]
    MeanVariance_normal = []
    MeanVariance_lognormal = []
    MeanVariance_pareto = []
    alpha_line = linspace(min_alpha, max_alpha, alpha_size)

    # Calculating the second moments of each distribution
    second_moment_normal = sigma_normal ** 2
    second_moment_lognormal = exp((sigma_lognormal ** 2) / 2) ** 2 + (exp(sigma_lognormal ** 2) - 1) \
                              * exp(sigma_lognormal ** 2)
    second_moment_pareto = (a * x_m / (a - 1)) ** 2 + (x_m ** 2) * a / ((a - 1) ** 2) * (a - 2)
    # _____________________________________________________

    for alpha in alpha_line:
        Gaussian_estimates = []
        Lognormal_estimates = []
        Pareto_estimates = []
        if u == 0:
            Gaussian_estimates = [estimator(data_mod(np.random.normal(0, sigma_normal, n),
                                                     value_at_risk(np.random.normal(0, sigma_normal, n), alpha)))
                                  for i in range(max_trials)]
            Lognormal_estimates = [estimator(data_mod(np.random.lognormal(0, sigma_lognormal, n),
                                                      value_at_risk(np.random.lognormal(0, sigma_lognormal, n), alpha)))
                                   for i in range(max_trials)]
            Pareto_estimates = [estimator(data_mod((np.random.pareto(a, n) + 1) * x_m,
                                                   value_at_risk((np.random.pareto(a, n) + 1) * x_m, alpha))) for i in
                                range(max_trials)]
        if u == 'true_2nd_moment':
            Gaussian_estimates = [estimator(data_mod(np.random.normal(0, sigma_normal, n),
                                                     value_at_risk(np.random.normal(0, sigma_normal, n), alpha)),
                                            second_moment_normal) for i in range(max_trials)]
            Lognormal_estimates = [estimator(data_mod(np.random.lognormal(0, sigma_lognormal, n),
                                                      value_at_risk(np.random.lognormal(0, sigma_lognormal, n), alpha)),
                                             second_moment_lognormal)
                                   for
                                   i in range(max_trials)]
            Pareto_estimates = [estimator(data_mod((np.random.pareto(a, n) + 1) * x_m,
                                                   value_at_risk((np.random.pareto(a, n) + 1) * x_m, alpha)),
                                          second_moment_pareto) for p in
                                range(max_trials)]
        if u == 'empirical_2nd_moment':
            Gaussian = np.random.normal(0, sigma_normal, n)
            Lognormal = np.random.lognormal(0, sigma_lognormal, n)
            Pareto = (np.random.pareto(a, n) + 1) * x_m

            Gaussian_estimates = [estimator(data_mod(Gaussian,
                                                     value_at_risk(np.random.normal(0, sigma_normal, n), alpha)),
                                            np.var(Gaussian)) for i in range(max_trials)]
            Lognormal_estimates = [estimator(data_mod(Lognormal,
                                                      value_at_risk(np.random.lognormal(0, sigma_lognormal, n), alpha)),
                                             np.var(Lognormal) + np.mean(Lognormal) ** 2)
                                   for i in range(max_trials)]
            Pareto_estimates = [estimator(data_mod(Pareto,
                                                   value_at_risk((np.random.pareto(a, n) + 1) * x_m, alpha)),
                                          np.var(Pareto)) + np.mean(Pareto) ** 2 for i in range(max_trials)]

        MeanVariance_normal.append(
            (abs(np.mean(Gaussian_estimates) - cvar('normal', sigma_normal, sigma_lognormal, x_m, a, alpha)),
             np.var(Gaussian_estimates)))
        MeanVariance_lognormal.append(
            (abs(np.mean(Lognormal_estimates) - cvar('lognormal', sigma_normal, sigma_lognormal, x_m, a, alpha)),
             np.var(Lognormal_estimates)))
        MeanVariance_pareto.append(
            (abs(np.mean(Pareto_estimates) - cvar('pareto', sigma_normal, sigma_lognormal, x_m, a, alpha)),
             np.var(Pareto_estimates)))

    return [[string_estimator + '_Normal'] + MeanVariance_normal] + [[string_estimator + '_Lognormal'] +
                                                                     MeanVariance_lognormal] + [[string_estimator +
                                                                                                 '_Pareto'] +
                                                                                                MeanVariance_pareto]
