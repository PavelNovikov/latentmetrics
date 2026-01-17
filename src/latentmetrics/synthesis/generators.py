"""
Submodule for generating synthetic bivariate samples.
"""

import numpy as np
import scipy.stats as st


class Synthesis:

    @staticmethod
    def _get_rng(seed):
        return np.random.default_rng(seed)

    @staticmethod
    def normal(rho, n=128, seed=42):
        """
        Generates samples from a Bivariate Standard Normal distribution.
        """
        rng = Synthesis._get_rng(seed)
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        data = rng.multivariate_normal(mean, cov, size=n)
        return data[:, 0], data[:, 1]

    @staticmethod
    def lognormal(rho, n=128, s=1.0, seed=42):
        """
        Generates samples from a Bivariate Lognormal distribution via a Gaussian Copula.
        """
        z_x, z_y = Synthesis.normal(rho, n, seed)
        u_x = st.norm.cdf(z_x)
        u_y = st.norm.cdf(z_y)
        x = st.lognorm.ppf(u_x, s=s, scale=np.exp(0.0))
        y = st.lognorm.ppf(u_y, s=s, scale=np.exp(0.0))
        return x, y

    @staticmethod
    def clayton(n=128, tau=None, theta=None, seed=42):
        """
        Generating clayton copula with Marshall-Olkin algorithm

        Reference
        ----------
        Hofert, M. (2008). Sampling archimedean copulas.
        Computational Statistics & Data Analysis, 52(12), 5163–5174.

        """
        if (tau is None) == (theta is None):
            raise ValueError("Exactly one of 'tau' or 'theta' must be provided.")
        if theta is None:
            theta = (2 * tau) / (1 - tau)

        rng = Synthesis._get_rng(seed)

        # 1. Sample V ~ F. For Clayton: Gamma(shape=1/theta, scale=1)
        v = rng.gamma(1.0 / theta, 1.0, n)

        # 2. Sample i.i.d. Xi ~ U[0, 1]
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)

        # 3. Return Ui = psi(-log(Xi) / V)
        u1 = (1.0 + (-np.log(x1) / v)) ** (-1.0 / theta)
        u2 = (1.0 + (-np.log(x2) / v)) ** (-1.0 / theta)

        return u1, u2

    @staticmethod
    def gumbel(n=128, tau=None, theta=None, seed=42):
        """
        Generating Gumbel copula with Marshall-Olkin algorithm

        Reference
        ----------
        Hofert, M. (2008). Sampling archimedean copulas.
        Computational Statistics & Data Analysis, 52(12), 5163–5174.
        """
        if (tau is None) == (theta is None):
            raise ValueError("Exactly one of 'tau' or 'theta' must be provided.")
        if theta is None:
            theta = 1.0 / (1.0 - tau)

        rng = Synthesis._get_rng(seed)

        alpha = 1.0 / theta

        # Scale parameter from Hofert (Table 1)
        scale = (np.cos(np.pi * alpha / 2.0)) ** (1.0 / alpha)

        # 1. Sample V ~ positive alpha-stable
        v = st.levy_stable.rvs(
            alpha=alpha, beta=1.0, loc=0.0, scale=scale, size=n, random_state=seed
        )

        # Numerical safety
        v = np.maximum(v, np.finfo(float).tiny)

        # 2. Sample i.i.d. Xi ~ U[0, 1]
        x1 = rng.uniform(0.0, 1.0, n)
        x2 = rng.uniform(0.0, 1.0, n)

        # 3. Return Ui = psi(-log(Xi) / V)
        u1 = np.exp(-((-np.log(x1) / v) ** alpha))
        u2 = np.exp(-((-np.log(x2) / v) ** alpha))

        return u1, u2

    @staticmethod
    def discretize(data, probabilities, labels=None):
        """
        Discretizes continuous data into categories based on a probability vector.
        """
        probabilities = np.array(probabilities)

        # Validation
        if not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("The probability vector must sum to 1.")

        # 1. Calculate the cumulative probabilities (the cut points in CDF space)
        # We drop the first 0 and the last 1 for np.quantile boundaries
        cum_probs = np.cumsum(probabilities)[:-1]

        # 2. Find the values in the data that correspond to these quantiles
        bins = np.quantile(data, cum_probs)

        # 3. Digitization: assigns data points to bins
        indices = np.digitize(data, bins)

        if labels is not None:
            if len(labels) != len(probabilities):
                raise ValueError("Labels length must match probabilities length.")
            return np.array([labels[i] for i in indices])

        return indices
