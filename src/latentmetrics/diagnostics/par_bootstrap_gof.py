"""
Tests for Assessing Agreement with a Gaussian Copula Model
Using the Asymptotic Distribution of the Test Statistic
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Any, Dict, Tuple
from scipy import stats
from scipy.stats import norm, multivariate_normal
from latentmetrics.types import VariableType as vt
from latentmetrics._utils import swap_args
from functools import partial

FitFn = Callable[[ArrayLike, ArrayLike], Any]
GenerateFn = Callable[[Any, int, np.random.Generator], Tuple[ArrayLike, ArrayLike]]
StatFn = Callable[[ArrayLike, ArrayLike, Any], float]


def get_gof_test_fn(x_type, y_type):
    xt, yt = x_type, y_type
    if x_type == vt.BINARY:
        xt = vt.ORDINAL
    if y_type == vt.BINARY:
        yt = vt.ORDINAL
    if (xt, yt) == (vt.CONTINUOUS, vt.CONTINUOUS):
        return partial(run_parametric_bootstrap, stat_fn=gaussian_copula_cvm_stat)
    elif (xt, yt) == (vt.CONTINUOUS, vt.ORDINAL):
        return partial(run_parametric_bootstrap, stat_fn=cont_ord_gaussian_cvm_stat)
    elif (xt, yt) == (vt.ORDINAL, vt.CONTINUOUS):
        return partial(
            run_parametric_bootstrap, stat_fn=swap_args(cont_ord_gaussian_cvm_stat)
        )
    elif (xt, yt) == (vt.ORDINAL, vt.ORDINAL):
        return partial(run_parametric_bootstrap, stat_fn=polychoric_cvm_stat)


def run_parametric_bootstrap(
    x: ArrayLike,
    y: ArrayLike,
    fit_fn: FitFn,
    generate_fn: GenerateFn,
    stat_fn: StatFn,
    n_boot: int = 100,
    seed: int = 42,
) -> Dict[str, float]:

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n = len(x_arr)
    rng = np.random.default_rng(seed)

    params_obs = fit_fn(x_arr, y_arr)
    s_obs = stat_fn(x_arr, y_arr, params_obs)

    boot_stats = np.zeros(n_boot)
    for b in range(n_boot):
        x_synth, y_synth = generate_fn(params_obs, n, rng)

        params_star = fit_fn(x_synth, y_synth)

        boot_stats[b] = stat_fn(x_synth, y_synth, params_star)

    p_value = np.mean(boot_stats >= s_obs)

    return float(p_value)


def empirical_copula_at_points(obs: np.ndarray):
    """
    Calculates the empirical CDF for all points.
    """
    n, _ = obs.shape
    ecdf_values = np.zeros(n)

    for i in range(n):
        mask = np.all(obs <= obs[i], axis=1)
        ecdf_values[i] = np.sum(mask) / n

    return ecdf_values


def gaussian_copula_cvm_stat(x: np.ndarray, y: np.ndarray, rho: float) -> float:
    """
    Cramér-von Mises statistic Sn for the Gaussian Copula

    Reference
    ----------
    Genest, C., Rémillard, B., & Beaudoin, D. (2009).
    Goodness-of-fit tests for copulas: A review and a power study.
    Insurance: Mathematics and economics, 44(2), 199-213.
    """
    n = len(x)

    u = stats.rankdata(x) / (n + 1)
    v = stats.rankdata(y) / (n + 1)
    uv = np.column_stack([u, v])

    c_emp = empirical_copula_at_points(uv)

    z = norm.ppf(uv)
    r = np.clip(rho, -0.999, 0.999)
    c_theo = multivariate_normal.cdf(z, cov=[[1, r], [r, 1]])

    s_n = np.sum((c_emp - c_theo) ** 2)

    return float(s_n)


def cont_ord_gaussian_cvm_stat(
    x_continuous: np.ndarray, y_ordinal: np.ndarray, rho: float, bins: int = 10
) -> float:
    """
    Test statistic for Mixed (Continuous-Ordinal) data.

    The function discretizes the continuous variable into bins
    based on quantiles and then computes the polychoric statistic.
    """
    quantiles = np.linspace(0, 1, bins + 1)
    thresholds = np.quantile(x_continuous, quantiles[1:-1])
    x_binned = np.digitize(x_continuous, thresholds)
    return polychoric_cvm_stat(x_binned, y_ordinal, rho)


def polychoric_cvm_stat(x: ArrayLike, y: ArrayLike, rho: float) -> float:
    """
    Test statistic for the Ordinal-Ordinal case

    Reference
    ---------
    Foldnes, N., & Grønneberg, S. (2020).
    Pernicious polychorics: The impact and detection of underlying non-normality.
    Structural Equation Modeling: A Multidisciplinary Journal, 27(4), 525-543.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n = len(x_arr)

    obs_counts = stats.contingency.crosstab(x_arr, y_arr)[1]
    pi_obs = obs_counts / n

    def get_thresholds(data):
        _, counts = np.unique(data, return_counts=True)
        return norm.ppf(np.cumsum(counts)[:-1] / len(data))

    tau_x = np.concatenate([[-np.inf], get_thresholds(x_arr), [np.inf]])
    tau_y = np.concatenate([[-np.inf], get_thresholds(y_arr), [np.inf]])

    # Create a grid of all threshold intersections
    tx, ty = np.meshgrid(tau_x, tau_y, indexing="ij")

    points = np.stack([tx.flatten(), ty.flatten()], axis=1)
    cov = [[1, rho], [rho, 1]]

    cdf_grid = multivariate_normal.cdf(points, cov=cov).reshape(tx.shape)

    # Prob = F(x2, y2) - F(x1, y2) - F(x2, y1) + F(x1, y1)
    pi_theo = (
        cdf_grid[1:, 1:] - cdf_grid[:-1, 1:] - cdf_grid[1:, :-1] + cdf_grid[:-1, :-1]
    )

    pi_theo = np.maximum(pi_theo, 1e-10)
    return np.sum((pi_obs - pi_theo) ** 2)
