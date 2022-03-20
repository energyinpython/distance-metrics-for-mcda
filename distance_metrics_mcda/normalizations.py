import numpy as np

# linear normalization
def linear_normalization(X, types):
    """
    Normalize decision matrix using linear normalization method.

    Parameters
    ----------
        X : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    -------
        ndarray
            Normalized decision matrix
    """
    x_norm = np.zeros(np.shape(X))
    x_norm[:, types == 1] = X[:, types == 1] / (np.amax(X[:, types == 1], axis = 0))
    x_norm[:, types == -1] = np.amin(X[:, types == -1], axis = 0) / X[:, types == -1]
    return x_norm


# min-max normalization
def minmax_normalization(X, types):
    """
    Normalize decision matrix using minimum-maximum normalization method.

    Parameters
    ----------
        X : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    -------
        ndarray
            Normalized decision matrix
    """
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    x_norm[:, types == 1] = (X[:, types == 1] - np.amin(X[:, types == 1], axis = 0)
                             ) / (np.amax(X[:, types == 1], axis = 0) - np.amin(X[:, types == 1], axis = 0))

    x_norm[:, types == -1] = (np.amax(X[:, types == -1], axis = 0) - X[:, types == -1]
                           ) / (np.amax(X[:, types == -1], axis = 0) - np.amin(X[:, types == -1], axis = 0))

    return x_norm


# max normalization
def max_normalization(X, types):
    """
    Normalize decision matrix using maximum normalization method.

    Parameters
    ----------
        X : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    -------
        ndarray
            Normalized decision matrix
    """
    maximes = np.amax(X, axis = 0)
    X = X / maximes
    X[:, types == -1] = 1 - X[:, types == -1]
    return X


# sum normalization
def sum_normalization(X, types):
    """
    Normalize decision matrix using sum normalization method.

    Parameters
    ----------
        X : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    -------
        ndarray
            Normalized decision matrix
    """
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    x_norm[:, types == 1] = X[:, types == 1] / np.sum(X[:, types == 1], axis = 0)
    x_norm[:, types == -1] = (1 / X[:, types == -1]) / np.sum((1 / X[:, types == -1]), axis = 0)

    return x_norm


# vector normalization
def vector_normalization(X, types):
    """
    Normalize decision matrix using vector normalization method.

    Parameters
    ----------
        X : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    -------
        ndarray
            Normalized decision matrix
    """
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    x_norm[:, types == 1] = X[:, types == 1] / (np.sum(X[:, types == 1] ** 2, axis = 0))**(0.5)
    x_norm[:, types == -1] = 1 - (X[:, types == -1] / (np.sum(X[:, types == -1] ** 2, axis = 0))**(0.5))

    return x_norm
