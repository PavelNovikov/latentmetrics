"""
Latent correlation estimation using value-based statistics under
bivariate Gaussian assumptions.

The methods include biserial, polyserial, tetrachoric,
and polychoric correlations.
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, root_scalar
from scipy.stats import multivariate_normal, norm, pearsonr

from .utils import build_contingency_mat


# ---------------------------------------------------------------------
# Continuous–continuous
# ---------------------------------------------------------------------


def pearson_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Standard Pearson product–moment correlation."""

    return float(pearsonr(x, y).statistic)


# ---------------------------------------------------------------------
# Binary–binary
# ---------------------------------------------------------------------


def tetrachoric_correlation(
    binary_x: ArrayLike,
    binary_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = None,
) -> float:
    """
    Compute the tetrachoric correlation between two binary variables.

    The tetrachoric correlation estimates the correlation of two latent
    normally distributed variables that have been dichotomized.

    Reference
    ----------
    Brown, M. B. (1977).
    *The tetrachoric correlation and its asymptotic standard error*.
    JRSS-C, 26(3), 343–351.
    """

    x = np.asarray(binary_x)
    y = np.asarray(binary_y)

    # Identify the "positive" category for each variable
    x_pos = x.max()
    y_pos = y.max()

    # Marginal probabilities
    p_x = (x == x_pos).mean()
    p_y = (y == y_pos).mean()

    # Joint probability of both being positive
    p_xy = ((x == x_pos) & (y == y_pos)).mean()

    rng = np.random.default_rng(seed)

    def objective(rho: float) -> float:
        """
        Difference between the implied bivariate normal probability
        and the observed joint probability.
        """

        thresholds = [norm.ppf(p_x), norm.ppf(p_y)]

        implied = multivariate_normal.cdf(
            thresholds,
            cov=[[1.0, rho], [rho, 1.0]],
            rng=rng,
        )

        return float(implied - p_xy)

    solution = root_scalar(
        objective,
        bracket=[-1.0 + eps, 1.0 - eps],
        method="bisect",
    )

    return float(solution.root)


# ---------------------------------------------------------------------
# Continuous–binary
# ---------------------------------------------------------------------


def biserial_correlation(
    continuous: ArrayLike,
    binary: ArrayLike,
) -> float:
    """
    Compute the biserial correlation between a continuous variable and
    a binary variable.

    The biserial correlation assumes the binary variable reflects an
    underlying dichotomized normal distribution.

    Reference
    ----------
    Tate, R. F. (1950).
    *The biserial and point correlation coefficients*.
    North Carolina State University, Department of Statistics.
    """

    x = np.asarray(continuous)
    y = np.asarray(binary)

    pos = y.max()

    mean_pos = x[y == pos].mean()
    mean_neg = x[y != pos].mean()

    p_pos = (y == pos).mean()
    p_neg = 1.0 - p_pos

    std_x = np.std(x, ddof=0)

    # Height of standard normal density at the dichotomization threshold
    z_thresh = norm.ppf(p_pos)
    pdf_z = norm.pdf(z_thresh)

    rho = (mean_pos - mean_neg) / std_x * (p_pos * p_neg / pdf_z)

    return float(rho)


# ---------------------------------------------------------------------
# Ordinal–ordinal
# ---------------------------------------------------------------------


def polychoric_correlation(
    ordinal_x: ArrayLike,
    ordinal_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = None,
) -> float:
    """
    Compute the polychoric correlation between two ordinal variables
    using the two-step maximum likelihood method.

    The two-step method:
    1. Estimates marginal thresholds from observed proportions.
    2. Maximizes the likelihood with respect to the latent correlation.

    Reference
    ----------
    Olsson, U. (1979).
    *Maximum likelihood estimation of the polychoric correlation coefficient*.
    Psychometrika, 44(4), 443–460.
    """

    x = np.asarray(ordinal_x)
    y = np.asarray(ordinal_y)

    # Contingency table n_ij
    contingency = build_contingency_mat(x, y)
    s, r = contingency.shape
    N = contingency.sum()

    # Marginal cumulative proportions
    a_cum = np.cumsum(contingency.sum(axis=1) / N)[:-1]
    b_cum = np.cumsum(contingency.sum(axis=0) / N)[:-1]

    # Latent thresholds: a_0 = -inf, a_s = +inf
    a = np.concatenate(([-np.inf], norm.ppf(a_cum), [np.inf]))
    b = np.concatenate(([-np.inf], norm.ppf(b_cum), [np.inf]))

    rng = np.random.default_rng(seed)

    def Phi2(h: float, k: float, rho: float) -> float:
        """Bivariate standard normal CDF with correlation rho."""

        if np.isneginf(h) or np.isneginf(k):
            return 0.0
        if np.isposinf(h) and np.isposinf(k):
            return 1.0
        if np.isposinf(h):
            return float(norm.cdf(k))
        if np.isposinf(k):
            return float(norm.cdf(h))

        return float(
            multivariate_normal.cdf(
                [h, k],
                mean=[0.0, 0.0],
                cov=[[1.0, rho], [rho, 1.0]],
                rng=rng,
            )
        )

    def negative_log_likelihood(params: np.ndarray) -> float:
        """
        Negative log-likelihood for the polychoric model.
        """

        rho = params[0]
        ll = 0.0

        for i in range(s):
            for j in range(r):
                n_ij = contingency[i, j]
                if n_ij == 0:
                    continue

                p_ij = (
                    Phi2(a[i + 1], b[j + 1], rho)
                    - Phi2(a[i], b[j + 1], rho)
                    - Phi2(a[i + 1], b[j], rho)
                    + Phi2(a[i], b[j], rho)
                )

                ll += n_ij * np.log(max(p_ij, 1e-12))

        return -ll

    result = minimize(
        negative_log_likelihood,
        x0=[0.0],
        bounds=[(-1.0 + eps, 1.0 - eps)],
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(f"Polychoric optimization failed: {result.message}")

    return float(result.x[0])


# ---------------------------------------------------------------------
# Continuous–ordinal
# ---------------------------------------------------------------------


def polyserial_correlation(
    continuous_x: ArrayLike,
    ordinal_y: ArrayLike,
    eps: float = 1e-8,
    seed: Optional[int] = None,
) -> float:
    """
    Compute the polyserial correlation between a continuous variable and
    an ordinal variable using the two-step maximum likelihood method.

    Reference
    ----------
    Olsson, U., Drasgow, F., & Dorans, N. J. (1982).
    *The polyserial correlation coefficient*.
    Psychometrika, 47(3), 337–347.
    """

    X = np.asarray(continuous_x)
    Y = np.asarray(ordinal_y)

    # Standardize continuous variable
    Z = (X - X.mean()) / X.std(ddof=0)

    # Encode ordinal categories as indices
    levels = np.unique(Y)
    y_idx = np.searchsorted(levels, Y)
    r = len(levels)

    # Estimate thresholds from marginal proportions
    counts = np.bincount(y_idx, minlength=r)
    cum_props = np.cumsum(counts / len(Y))[:-1]

    tau = np.concatenate(([-np.inf], norm.ppf(cum_props), [np.inf]))

    rng = np.random.default_rng(seed)

    def negative_log_likelihood(rho: float) -> float:
        """
        Conditional negative log-likelihood of Y given X.
        """

        denom = np.sqrt(1.0 - rho**2)

        tau_upper = (tau[y_idx + 1] - rho * Z) / denom
        tau_lower = (tau[y_idx] - rho * Z) / denom

        p = norm.cdf(tau_upper) - norm.cdf(tau_lower)

        return -np.sum(np.log(np.clip(p, 1e-12, None)))

    # Initial guess based on Pearson correlation
    rho0 = np.clip(np.corrcoef(Z, y_idx)[0, 1], -0.9, 0.9)

    result = minimize(
        fun=lambda p: negative_log_likelihood(p[0]),
        x0=[rho0],
        bounds=[(-1.0 + eps, 1.0 - eps)],
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(f"Polyserial optimization failed: {result.message}")

    return float(result.x[0])
