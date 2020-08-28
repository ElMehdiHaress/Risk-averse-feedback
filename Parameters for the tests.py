from math import *
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy import integrate
from scipy import special
from numpy import median
from numpy import linspace
from copy import deepcopy
from scipy.stats import fisk

"Defining the parameters for the tests"

alpha = 0.05
delta = 0.002
k = int(log(2 * int(log(1 / delta)) * (1 / delta)))
beta = 0.001
eta = 0.001
max_iterations = 15
max_gradient_descents = 15

sample_size = 2000
optimal_w = [2, 3]
initial_w = [4, 5]
initial_v = 5

mu = [0, 0]
covariance = [[10, 0], [0, 10]]
X = np.random.multivariate_normal(mu, covariance, sample_size).T

X1 = X[:, 0:int(sample_size / 2)]
X2 = X[:, int(sample_size / 2):sample_size]

max_trials = 1000
min_sample = 15
max_sample = 125
samples = 20
prediction = linspace(min_sample, max_sample, samples).astype(int)
