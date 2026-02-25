from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from scipy.stats import multivariate_normal, norm

from .utils import (
    get_category_zscores,
    get_threshold_zscore,
    recover_proportions,
    safe_root_scalar,
    compute_tau_a,
    compute_tau_a_continuous,
)


# --- Continuous–continuous -------------------------------------------


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
    Parameters behind "nonparametric" statistics: Kendall's tau,
    Somers' D and median differences.
    *The Stata Journal*, 2(1), 45–64.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    tau_observed = compute_tau_a_continuous(x, y)

    latent_rho = np.sin((np.pi / 2) * tau_observed)

    return float(latent_rho)


# --- Continuous–ordinal -----------------------------------------------


def latent_rank_co(
    continuous_x: ArrayLike,
    ordinal_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = 42,
    proportions: Optional[ArrayLike] = None,
) -> float:
    """
    Estimate the correlation parameter of a Gaussian copula underlying one
    continuous and one observed ordinal variable, based on Kendall's tau.

    Reference
    ----------
    Quan, X., Booth, J. G., & Wells, M. T. (2018).
    Rank-based approach for estimating correlations in mixed ordinal data.
    arXiv preprint arXiv:1809.06255.
    """

    x = np.asarray(continuous_x)
    y = np.asarray(ordinal_y)

    weights = np.asarray(proportions) * len(y) if proportions is not None else None
    tau_observed = compute_tau_a(x, y, weights=weights)
    zscores_y = get_category_zscores(y, proportions=proportions)

    rng = np.random.default_rng(seed)

    def bridge_function(rho: float) -> float:
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

    return safe_root_scalar(
        objective,
        bracket=[-1.0 + eps, 1.0 - eps],
        method="brentq",
    )


# --- Continuous–binary -----------------------------------------------


def latent_rank_cb(
    continuous_x: ArrayLike,
    binary_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = 42,
    proportions: Optional[ArrayLike] = None,
) -> float:
    """
    Estimate the correlation parameter of a Gaussian copula underlying one
    continuous and one observed binary variable, based on Kendall's tau.

    Reference
    ----------
    Fan, J., Liu, H., Ning, Y., & Zou, H. (2017).
    *High dimensional semiparametric latent graphical model for mixed data*.
    JRSS-B, 79(2), 405–421.
    """

    x = np.asarray(continuous_x)
    y = np.asarray(binary_y)

    weights = np.asarray(proportions) * len(y) if proportions is not None else None
    tau_observed = compute_tau_a(x, y, weights=weights)
    zscore_y = get_threshold_zscore(y, proportions=proportions)

    rng = np.random.default_rng(seed)

    def bridge_function(rho: float) -> float:
        mean = np.zeros(2)
        cov = np.eye(2)
        cov[0, 1] = cov[1, 0] = rho / np.sqrt(2)

        joint = multivariate_normal.cdf([zscore_y, 0.0], mean=mean, cov=cov, rng=rng)
        marginal = norm.cdf(zscore_y)

        return 4 * joint - 2 * marginal

    def objective(rho: float) -> float:
        return bridge_function(rho) - tau_observed

    return safe_root_scalar(
        objective,
        bracket=[-1.0 + eps, 1.0 - eps],
        method="brentq",
    )


# --- Ordinal–ordinal -------------------------------------------------


def latent_rank_oo(
    ordinal_x: ArrayLike,
    ordinal_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = 42,
    proportions: Optional[ArrayLike] = None,
) -> float:
    """
    Estimate the correlation parameter of a Gaussian copula underlying two
    observed ordinal variables, based on Kendall's tau.

    When proportions are provided, the true population proportions of x are
    recovered automatically via recover_proportions(), since subsampling on y
    also distorts the marginal distribution of x.

    Reference
    ----------
    Dey, D., & Zipunnikov, V. (2022).
    *Semiparametric Gaussian Copula Regression Modeling for Mixed Data Types (SGCRM)*.
    arXiv preprint arXiv:2205.06868.
    """

    x = np.asarray(ordinal_x)
    y = np.asarray(ordinal_y)

    weights = np.asarray(proportions) * len(y) if proportions is not None else None
    tau_observed = compute_tau_a(x, y, weights=weights)

    proportions_x = (
        recover_proportions(x, y, proportions) if proportions is not None else None
    )
    zscores_x = np.concatenate(
        ([-np.inf], get_category_zscores(x, proportions=proportions_x))
    )
    zscores_y = np.concatenate(
        ([-np.inf], get_category_zscores(y, proportions=proportions))
    )

    rng = np.random.default_rng(seed)

    def bridge_function(rho: float) -> float:
        mean = np.zeros(2)
        cov = [[1.0, rho], [rho, 1.0]]

        Ax, Ay = np.meshgrid(
            zscores_x[1:-1],
            zscores_y[1:-1],
            indexing="ij",
        )
        points = np.column_stack([Ax.ravel(), Ay.ravel()])
        cdf_ab = multivariate_normal.cdf(points, mean=mean, cov=cov, rng=rng).reshape(
            Ax.shape
        )

        Ax_n, Ay_n = np.meshgrid(
            zscores_x[2:],
            zscores_y[2:],
            indexing="ij",
        )
        points_n = np.column_stack([Ax_n.ravel(), Ay_n.ravel()])
        cdf_next = multivariate_normal.cdf(
            points_n, mean=mean, cov=cov, rng=rng
        ).reshape(Ax_n.shape)

        Ax_s, Ay_p = np.meshgrid(
            zscores_x[2:],
            zscores_y[:-2],
            indexing="ij",
        )
        points_p = np.column_stack([Ax_s.ravel(), Ay_p.ravel()])
        cdf_prev = multivariate_normal.cdf(
            points_p, mean=mean, cov=cov, rng=rng
        ).reshape(Ax_s.shape)

        s = np.sum(cdf_ab * (cdf_next - cdf_prev))

        norm_cdfs_x = norm.cdf(zscores_x[1:-1])
        corr_points = np.column_stack(
            [zscores_x[2:], np.full(len(zscores_x) - 2, zscores_y[-2])]
        )
        corr_cdf = multivariate_normal.cdf(corr_points, mean=mean, cov=cov, rng=rng)

        s -= np.sum(norm_cdfs_x * corr_cdf)

        return 2 * s

    def objective(rho: float) -> float:
        return bridge_function(rho) - tau_observed

    print("zscores_x", zscores_x)

    print("zscores_y", zscores_y)

    print("tau observed:", tau_observed)
    return safe_root_scalar(
        objective,
        bracket=[-1.0 + eps, 1.0 - eps],
        method="brentq",
    )


# --- Binary–binary ---------------------------------------------------


def latent_rank_bb(
    binary_x: ArrayLike,
    binary_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = 42,
    proportions: Optional[ArrayLike] = None,
) -> float:
    """
    Estimate the correlation parameter of a Gaussian copula underlying two
    observed binary variables, based on Kendall's tau.

    When proportions are provided, the true population proportion of x is
    recovered automatically via recover_proportions(), since subsampling on y
    also distorts the marginal distribution of x.

    Reference
    ----------
    Fan, J., Liu, H., Ning, Y., & Zou, H. (2017).
    *High dimensional semiparametric latent graphical model for mixed data*.
    JRSS-B, 79(2), 405–421.
    """

    x = np.asarray(binary_x)
    y = np.asarray(binary_y)

    weights = np.asarray(proportions) * len(y) if proportions is not None else None
    tau_observed = compute_tau_a(x, y, weights=weights)

    proportions_x = (
        recover_proportions(x, y, proportions) if proportions is not None else None
    )
    zscore_x = get_threshold_zscore(x, proportions=proportions_x)
    zscore_y = get_threshold_zscore(y, proportions=proportions)

    rng = np.random.default_rng(seed)

    def bridge_function(rho: float) -> float:
        mean = np.zeros(2)
        cov = [[1.0, rho], [rho, 1.0]]

        joint = multivariate_normal.cdf(
            [zscore_x, zscore_y], mean=mean, cov=cov, rng=rng
        )
        marginal = norm.cdf(zscore_x) * norm.cdf(zscore_y)

        return 2 * (joint - marginal)

    def objective(rho: float) -> float:
        return bridge_function(rho) - tau_observed

    return safe_root_scalar(
        objective,
        bracket=[-1.0 + eps, 1.0 - eps],
        method="brentq",
    )
