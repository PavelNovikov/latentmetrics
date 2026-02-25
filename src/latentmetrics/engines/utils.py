import numpy as np
from scipy.stats import norm, kendalltau
from numpy.typing import ArrayLike
from scipy.optimize import root_scalar


def safe_root_scalar(f, bracket: list, method: str = "bisect", **kwargs) -> float:
    a, b = bracket
    fa, fb = f(a), f(b)
    if np.sign(fa) != np.sign(fb):
        return root_scalar(f, bracket=[a, b], method=method, **kwargs).root
    if fa == 0:
        return a
    if fb == 0:
        return b
    return a if abs(fa) < abs(fb) else b


def get_threshold_zscore(x: ArrayLike, proportions: ArrayLike | None = None) -> float:
    """
    Converts binary class frequencies into a single normal quantile.
    If proportions are provided, they are used instead of empirical frequencies.
    """
    x_arr = np.asarray(x)
    _, counts = np.unique(x_arr, return_counts=True)
    if len(counts) != 2:
        raise ValueError(f"Input must contain exactly two classes, found {len(counts)}")

    if proportions is not None:
        props = np.asarray(proportions, dtype=float)
        if len(props) != 2:
            raise ValueError(f"proportions must have length 2, got {len(props)}")
        props = props / props.sum()
        quantile = float(props[0])
    else:
        quantile = counts[0] / counts.sum()

    return float(norm.ppf(quantile))


def get_category_zscores(
    x: ArrayLike, proportions: ArrayLike | None = None
) -> np.ndarray:
    """
    Converts categorical frequencies to a vector of normal quantiles.
    If proportions are provided, they are used instead of empirical frequencies.
    """
    x_arr = np.asarray(x)
    _, counts = np.unique(x_arr, return_counts=True)

    if proportions is not None:
        props = np.asarray(proportions, dtype=float)
        if len(props) != len(counts):
            raise ValueError(
                f"proportions length {len(props)} must match number of categories {len(counts)}"
            )
        props = props / props.sum()
    else:
        props = counts / counts.sum()

    quantiles = np.clip(props.cumsum(), 0.0, 1.0)
    zscores = norm.ppf(quantiles)
    zscores[-1] = np.inf
    return zscores


def build_contingency_mat(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Builds a 2D contingency table (frequency matrix) for two variables.
    """
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    x_idx = np.unique(x_arr, return_inverse=True)[1]
    y_idx = np.unique(y_arr, return_inverse=True)[1]
    n_x, n_y = x_idx.max() + 1, y_idx.max() + 1
    contingency = np.zeros((n_x, n_y), dtype=int)
    # Vectorized accumulation to populate the table
    np.add.at(contingency, (x_idx, y_idx), 1)
    return contingency


def compute_pairwise_concordance(
    x: ArrayLike, y: ArrayLike  # x is now continuous, y is ordinal
) -> tuple[np.ndarray, np.ndarray]:
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    classes = np.unique(y_arr)
    K = len(classes)

    x_by_class = [np.sort(x_arr[y_arr == k]) for k in classes]

    theta = np.zeros((K, K))

    for i in range(K):
        x_i, N_i = x_by_class[i], len(x_by_class[i])
        if N_i == 0:
            continue
        for j in range(K):
            if i == j:
                continue
            x_j, N_j = x_by_class[j], len(x_by_class[j])
            if N_j == 0:
                continue

            less_than = np.searchsorted(x_j, x_i, side="left")
            greater_than = N_j - np.searchsorted(x_j, x_i, side="right")

            theta[i, j] = np.sum(less_than - greater_than) / (N_i * N_j)

    return theta, classes


def compute_tau_a(
    x: ArrayLike,
    y: ArrayLike,  # Ordinal
    weights: ArrayLike | None = None,
) -> float:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    theta, classes = compute_pairwise_concordance(x_arr, y_arr)

    if weights is None:
        _, counts = np.unique(y_arr, return_counts=True)
        w = counts.astype(float)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(classes):
            raise ValueError(
                f"weights length {len(w)} must match number of classes {len(classes)}"
            )

    w_outer = np.outer(w, w)
    W = w.sum()
    numerator = -2.0 * np.triu(w_outer * theta, k=1).sum()
    denom = W * (W - 1)

    return float(numerator / denom)


def compute_tau_a_continuous(x: ArrayLike, y: ArrayLike) -> float:
    """
    Compute the original Kendall's tau (tau-a) by reconstructing it
    from the tie-corrected Kendall's tau-c.
    """

    tau_c = kendalltau(x, y, variant="c").correlation

    num_unique_x = len(np.unique(x))
    num_unique_y = len(np.unique(y))
    min_unique = min(num_unique_x, num_unique_y)

    n = len(x)

    tau_a = tau_c * (min_unique - 1) / min_unique * n / (n - 1)

    return float(tau_a)


def recover_proportions(
    x: ArrayLike,
    y: ArrayLike,
    proportions_y: ArrayLike,
) -> np.ndarray:
    """
    Recover the true population marginal proportions of a discrete x
    after outcome-dependent subsampling on y, using the identity:

        P(x=j) = sum_k P(x=j | y=k) * P(y=k)
    """
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    y_classes = np.unique(y_arr)
    x_classes = np.unique(x_arr)
    props_y = np.asarray(proportions_y, dtype=float)
    props_y = props_y / props_y.sum()

    cond_probs = np.array(
        [[np.mean(x_arr[y_arr == yk] == xj) for xj in x_classes] for yk in y_classes]
    )  # shape (K_y, K_x)

    result = props_y @ cond_probs
    return result / result.sum()
