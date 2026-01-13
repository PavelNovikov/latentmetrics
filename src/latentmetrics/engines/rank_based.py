"""
Latent correlation estimation using rank-based statistics under
Gaussian copula models.

The methods are based on:

Dey, D., & Zipunnikov, V. (2022).
"Semiparametric Gaussian Copula Regression Modeling for Mixed Data Types (SGCRM)."
arXiv preprint arXiv:2205.06868.
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import root_scalar
from scipy.stats import kendalltau, multivariate_normal, norm

from .utils import get_category_zscores, get_threshold_zscore


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def get_tau_a(x: ArrayLike, y: ArrayLike) -> float:
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


# ---------------------------------------------------------------------
# Continuous–continuous
# ---------------------------------------------------------------------


def latent_rank_cc(
    x: ArrayLike,
    y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = 42,
) -> float:
    """
    Estimate the latent correlation between two continuous variables
    from the observed Kendall's tau using Greiner's formula.

    Reference
    ----------
    Newson, R. (2002).
    Parameters behind “nonparametric” statistics: Kendall's tau,
    Somers’ D and median differences.
    *The Stata Journal*, 2(1), 45–64.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    tau_observed = get_tau_a(x, y)

    # Greiner's relationship between Kendall's tau and latent correlation
    latent_rho = np.sin((np.pi / 2) * tau_observed)

    return float(latent_rho)


# ---------------------------------------------------------------------
# Continuous–ordinal
# ---------------------------------------------------------------------


def latent_rank_co(
    continuous_x: ArrayLike,
    ordinal_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = 42,
) -> float:
    """
    Estimate the correlation parameter of a Gaussian copula underlying one
    continuous and one observed ordinal variable, based on Kendall’s tau.

    Reference
    ----------
    Dey, D., & Zipunnikov, V. (2022).
    *Semiparametric Gaussian Copula Regression Modeling for Mixed Data Types (SGCRM)*.
    arXiv preprint arXiv:2205.06868.
    """

    x = np.asarray(continuous_x)
    y = np.asarray(ordinal_y)

    tau_observed = get_tau_a(x, y)
    zscores_y = get_category_zscores(y)

    rng = np.random.default_rng(seed)

    def bridge_function(rho: float) -> float:
        """
        Map latent correlation rho to the expected Kendall's tau
        for a continuous–ordinal variable pair.
        """

        # Each row corresponds to (lower, upper) bounds of an ordinal category
        deltas = np.column_stack(
            (zscores_y[:-1], zscores_y[1:], np.zeros(len(zscores_y) - 1))
        )

        mean = np.zeros(3)
        cov = np.eye(3)
        cov[0, 2] = cov[2, 0] = rho / np.sqrt(2)
        cov[1, 2] = cov[2, 1] = -rho / np.sqrt(2)

        return float(
            np.sum(
                4 * multivariate_normal.cdf(deltas, mean=mean, cov=cov, rng=rng)
                - 2 * norm.cdf(deltas[:, 0]) * norm.cdf(deltas[:, 1])
            )
        )

    def objective(rho: float) -> float:
        return bridge_function(rho) - tau_observed

    solution = root_scalar(
        objective,
        bracket=[-1.0 + eps, 1.0 - eps],
        method="bisect",
    )

    return float(solution.root)


# ---------------------------------------------------------------------
# Continuous–binary
# ---------------------------------------------------------------------


def latent_rank_cb(
    continuous_x: ArrayLike,
    binary_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = 42,
) -> float:
    """
    Estimate the correlation parameter of a Gaussian copula underlying one
    continuous and one observed binary variable, based on Kendall’s tau.

    Reference
    ----------
    Fan, J., Liu, H., Ning, Y., & Zou, H. (2017).
    *High dimensional semiparametric latent graphical model for mixed data*.
    JRSS-B, 79(2), 405–421.
    """

    x = np.asarray(continuous_x)
    y = np.asarray(binary_y)

    tau_observed = get_tau_a(x, y)
    zscore_y = get_threshold_zscore(y)

    rng = np.random.default_rng(seed)

    def bridge_function(rho: float) -> float:
        """
        Expected Kendall's tau under a latent Gaussian copula
        for continuous–binary data.
        """

        mean = np.zeros(2)
        cov = np.eye(2)
        cov[0, 1] = cov[1, 0] = rho / np.sqrt(2)

        joint = multivariate_normal.cdf([zscore_y, 0.0], mean=mean, cov=cov, rng=rng)
        marginal = norm.cdf(zscore_y)

        return 4 * joint - 2 * marginal

    def objective(rho: float) -> float:
        return bridge_function(rho) - tau_observed

    solution = root_scalar(
        objective,
        bracket=[-1.0 + eps, 1.0 - eps],
        method="bisect",
    )

    return float(solution.root)


# ---------------------------------------------------------------------
# Ordinal–ordinal
# ---------------------------------------------------------------------


def latent_rank_oo(
    ordinal_x: ArrayLike,
    ordinal_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = 42,
) -> float:
    """
    Estimate the correlation parameter of a Gaussian copula underlying two
    observed ordinal variables, based on Kendall’s tau.

    Reference
    ----------
    Dey, D., & Zipunnikov, V. (2022).
    *Semiparametric Gaussian Copula Regression Modeling for Mixed Data Types (SGCRM)*.
    arXiv preprint arXiv:2205.06868.
    """

    x = np.asarray(ordinal_x)
    y = np.asarray(ordinal_y)

    # Observed Kendall's tau from data
    tau_observed = get_tau_a(x, y)

    # Latent Gaussian cut-points for ordinal categories
    # Include -inf to simplify indexing of adjacent intervals
    zscores_x = np.concatenate(([-np.inf], get_category_zscores(x)))
    zscores_y = np.concatenate(([-np.inf], get_category_zscores(y)))

    rng = np.random.default_rng(seed)

    def bridge_function(rho: float) -> float:
        """
        Compute the expected Kendall's tau as a function of the
        latent correlation rho.

        The computation proceeds by:
        1. Evaluating the bivariate Gaussian CDF at all interior
           combinations of adjacent cut-points.
        2. Aggregating contributions corresponding to concordant
           and discordant category pairs.
        3. Subtracting marginal correction terms.
        """

        mean = np.zeros(2)
        cov = [[1.0, rho], [rho, 1.0]]

        # Interior cut-points (exclude -inf and +inf)
        Ax, Ay = np.meshgrid(
            zscores_x[1:-1],
            zscores_y[1:-1],
            indexing="ij",
        )
        points = np.column_stack([Ax.ravel(), Ay.ravel()])
        cdf_ab = multivariate_normal.cdf(points, mean=mean, cov=cov, rng=rng).reshape(
            Ax.shape
        )

        # Upper-right corners of rectangles
        Ax_n, Ay_n = np.meshgrid(
            zscores_x[2:],
            zscores_y[2:],
            indexing="ij",
        )
        points_n = np.column_stack([Ax_n.ravel(), Ay_n.ravel()])
        cdf_next = multivariate_normal.cdf(
            points_n, mean=mean, cov=cov, rng=rng
        ).reshape(Ax_n.shape)

        # Lower-right corners of rectangles
        Ax_s, Ay_p = np.meshgrid(
            zscores_x[2:],
            zscores_y[:-2],
            indexing="ij",
        )
        points_p = np.column_stack([Ax_s.ravel(), Ay_p.ravel()])
        cdf_prev = multivariate_normal.cdf(
            points_p, mean=mean, cov=cov, rng=rng
        ).reshape(Ax_s.shape)

        # Core summation term from the bridge function
        s = np.sum(cdf_ab * (cdf_next - cdf_prev))

        # Marginal term
        norm_cdfs_x = norm.cdf(zscores_x[1:-1])
        corr_points = np.column_stack(
            [zscores_x[2:], np.full(len(zscores_x) - 2, zscores_y[-2])]
        )
        corr_cdf = multivariate_normal.cdf(corr_points, mean=mean, cov=cov, rng=rng)

        s -= np.sum(norm_cdfs_x * corr_cdf)

        return 2 * s

    def objective(rho: float) -> float:
        return bridge_function(rho) - tau_observed

    solution = root_scalar(
        objective,
        bracket=[-1.0 + eps, 1.0 - eps],
        method="bisect",
    )

    return float(solution.root)


# ---------------------------------------------------------------------
# Binary–binary
# ---------------------------------------------------------------------


def latent_rank_bb(
    binary_x: ArrayLike,
    binary_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = 42,
) -> float:
    """
    Estimate the correlation parameter of a Gaussian copula underlying two
    observed binary variables, based on Kendall’s tau.

    Reference
    ----------
    Fan, J., Liu, H., Ning, Y., & Zou, H. (2017).
    *High dimensional semiparametric latent graphical model for mixed data*.
    JRSS-B, 79(2), 405–421.
    """

    x = np.asarray(binary_x)
    y = np.asarray(binary_y)

    tau_observed = get_tau_a(x, y)
    zscore_x = get_threshold_zscore(x)
    zscore_y = get_threshold_zscore(y)

    rng = np.random.default_rng(seed)

    def bridge_function(rho: float) -> float:
        """Expected Kendall's tau under a latent Gaussian copula."""

        mean = np.zeros(2)
        cov = [[1.0, rho], [rho, 1.0]]

        joint = multivariate_normal.cdf(
            [zscore_x, zscore_y], mean=mean, cov=cov, rng=rng
        )
        marginal = norm.cdf(zscore_x) * norm.cdf(zscore_y)

        return 2 * (joint - marginal)

    def objective(rho: float) -> float:
        return bridge_function(rho) - tau_observed

    solution = root_scalar(
        objective,
        bracket=[-1.0 + eps, 1.0 - eps],
        method="bisect",
    )

    return float(solution.root)
