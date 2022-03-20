import numpy as np
import itertools
from .correlations import pearson_coeff
from .normalizations import minmax_normalization, sum_normalization


# entropy weighting
def entropy_weighting(X, types):
    """
    Calculate criteria weights using objective Entropy weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray

    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)
    pij = np.abs(pij)
    m, n = np.shape(pij)

    H = np.zeros((m, n))

    for j, i in itertools.product(range(n), range(m)):
        if pij[i, j]:
            H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))
    d = 1 - h
    w = d / (np.sum(d))
    return w


# CRITIC weighting
def critic_weighting(X, types):
    """
    Calculate criteria weights using objective CRITIC weighting method
    Parameters
    ----------
        X : ndarray
            Decision matrix with performance values of m alternatives and n criteria
        types : ndarray
            
    Returns
    -------
        ndarray
            vector of criteria weights
    """
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    x_norm = minmax_normalization(X, criteria_type)
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    correlations = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        correlations[i, j] = pearson_coeff(x_norm[:, i], x_norm[:, j])

    difference = 1 - correlations
    C = std * np.sum(difference, axis = 0)
    w = C / np.sum(C)
    return w