import numpy as np
import pytest
from scipy.optimize import brentq


# =====================================================
# Gaussian copula (normal marginals)
# =====================================================


@pytest.fixture
def gaussian_dist():
    def _gen(rho, n=10_000, seed=0):
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal(
            mean=[0, 0],
            cov=[[1, rho], [rho, 1]],
            size=n,
        )
        return z[:, 0], z[:, 1]

    return _gen


# =====================================================
# Gaussian copula + lognormal marginals
# =====================================================


@pytest.fixture
def lognormal_gaussian_copula():
    def _gen(rho, n=10_000, seed=0):
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal(
            mean=[0, 0],
            cov=[[1, rho], [rho, 1]],
            size=n,
        )
        return np.exp(z[:, 0]), np.exp(z[:, 1])

    return _gen


# =====================================================
# Gumbel–Hougaard copula (Uniform marginals)
# =====================================================


def gumbel_copula_conditional_dist(v, u, theta):
    ln_u = -np.log(u)
    ln_v = -np.log(v)

    term_sum = ln_u**theta + ln_v**theta
    exponent = term_sum ** (1 / theta)

    return np.exp(-exponent) * (term_sum ** (1 / theta - 1)) * (ln_u ** (theta - 1)) / u


@pytest.fixture
def gumbel_copula():
    def _gen(rho, n=8_000, seed=42):
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 1, n)

        # Kendall's tau for Gumbel copula
        tau = 2 / np.pi * np.arcsin(rho)
        theta = 1 / (1 - tau)

        w = rng.uniform(0, 1, n)
        v = []

        for ui, wi in zip(u, w):
            f = lambda x: gumbel_copula_conditional_dist(x, ui, theta) - wi
            try:
                v.append(brentq(f, 1e-12, 1 - 1e-12))
            except ValueError:
                v.append(np.nan)

        u = np.asarray(u)
        v = np.asarray(v)

        mask = np.isfinite(u) & np.isfinite(v)
        return u[mask], v[mask]

    return _gen
