import numpy as np
from scipy.stats import norm
from numpy.typing import ArrayLike
from scipy.optimize import root_scalar


def safe_root_scalar(f, bracket, method="bisect", **kwargs):
    a, b = bracket
    fa = f(a)
    fb = f(b)

    if np.sign(fa) != np.sign(fb):
        return root_scalar(f, bracket=[a, b], method=method, **kwargs).root

    if fa == 0:
        return a
    if fb == 0:
        return b

    return a if abs(fa) < abs(fb) else b


def get_threshold_zscore(x: ArrayLike) -> float:
    """
    Converts binary class frequencies into a single normal quantile.
    """
    x_arr = np.asarray(x)
    _, counts = np.unique(x_arr, return_counts=True)

    if len(counts) != 2:
        raise ValueError(f"Input must contain exactly two classes, found {len(counts)}")

    # Calculate the quantile of the first class (sorted order)
    # This represents the point on the CDF where the split occurs
    quantile = counts[0] / counts.sum()

    return float(norm.ppf(quantile))


def get_category_zscores(x: ArrayLike) -> np.ndarray:
    """
    Converts categorical frequencies to a vector of normal quantiles.
    """
    x_arr = np.asarray(x)
    _, counts = np.unique(x_arr, return_counts=True)

    # Cumulative proportions define the thresholds for each category boundary
    quantiles = counts.cumsum() / counts.sum()

    # We return the Z-scores corresponding to these cumulative probabilities
    return norm.ppf(quantiles)


def build_contingency_mat(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Builds a 2D contingency table (frequency matrix) for two variables.
    """
    x_arr, y_arr = np.asarray(x), np.asarray(y)

    # Convert labels to 0-indexed integers
    x_idx = np.unique(x_arr, return_inverse=True)[1]
    y_idx = np.unique(y_arr, return_inverse=True)[1]

    # Initialize the matrix based on the number of unique categories
    n_x, n_y = x_idx.max() + 1, y_idx.max() + 1
    contingency = np.zeros((n_x, n_y), dtype=int)

    # Use vectorized accumulation to populate the table
    np.add.at(contingency, (x_idx, y_idx), 1)

    return contingency
