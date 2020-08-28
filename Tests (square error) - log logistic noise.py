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

# Defining the noise
noise = fisk.rvs(2.95, size=sample_size)

# Training the model
noise1 = noise[0:int(sample_size / 2)]
noise2 = noise[int(sample_size / 2):sample_size]
Y1 = np.dot(optimal_w, X1) + noise1
Y2 = np.dot(optimal_w, X2) + noise2

robust_sgd = robustified_sgd(initial_w, initial_v, X1, Y1, X2, Y2, alpha, delta, k, beta)
stagewise_rgd = rgd(initial_w, X1, Y1, X2, Y2, alpha, delta, eta, max_iterations, max_gradient_descents)
empirical_minimization = erm(initial_w, X1, Y1, X2, Y2, alpha, eta, max_iterations, max_gradient_descents)
print(robust_sgd, stagewise_rgd, empirical_minimization)

# Computing the prediction error over many trials with respect to the sample size
list_rgd = []
list_sgd = []
list_erm = []
for n in prediction:
    X = [np.random.multivariate_normal(mu, covariance, n).T for i in range(max_trials)]
    noise = [np.random.lognormal(0, 1, n) for i in range(max_trials)]
    Y = np.dot(optimal_w, X) + noise

    list_sgd += [sum([np.mean((np.dot(robust_sgd, X[i]) - Y[i])**2) for i in range(max_trials)]) / max_trials]
    list_rgd += [sum([np.mean((np.dot(stagewise_rgd, X[i]) - Y[i])**2) for i in range(max_trials)]) / max_trials]
    list_erm += [sum([np.mean((np.dot(empirical_minimization, X[i]) - Y[i])**2) for i in range(max_trials)])
                 / max_trials]

plt.plot(prediction, list_rgd, 'red', label='rgd')
plt.plot(prediction, list_sgd, 'blue', label='robust_sgd')
plt.plot(prediction, list_erm, 'green', label='erm')

