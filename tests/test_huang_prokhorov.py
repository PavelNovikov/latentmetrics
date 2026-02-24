import numpy as np
import pytest
import sympy as sp
from scipy.stats import norm, multivariate_normal
from dataclasses import dataclass
from typing import Callable
from latentmetrics.diagnostics.asymptotic_gof import (
    calculate_score,
    calculate_hessian,
    calculate_grad_D,
    calculate_M,
    calculate_W,
    calculate_correction_factor,
    calculate_mixed_derivative,
    huang_prokhorov_test,
    calculate_grad_D_dq1,
)


# --- Fixtures and Utilities -----------------------------------------------------


@dataclass
class DistributionPoint:
    """Container for test point parameters"""

    u: float
    v: float
    rho: float
    name: str

    @property
    def q1(self):
        return norm.ppf(self.u)

    @property
    def q2(self):
        return norm.ppf(self.v)


# Shared parametrize list used across TestCoreFormulas, TestWComponent, and
# TestMComponent to exercise a representative spread of (u, v, rho) combinations.
STANDARD_TEST_POINTS = [
    DistributionPoint(u=0.5, v=0.7, rho=0.5,  name="Standard"),
    DistributionPoint(u=0.5, v=0.5, rho=0.5,  name="Symmetric"),
    DistributionPoint(u=0.3, v=0.8, rho=0.3,  name="Low rho"),
    DistributionPoint(u=0.6, v=0.4, rho=0.8,  name="High rho"),
    DistributionPoint(u=0.5, v=0.7, rho=0.0,  name="Zero rho"),
    DistributionPoint(u=0.5, v=0.7, rho=-0.5, name="Negative rho"),
]


@dataclass
class GaussianCopulaSymbolic:
    """Sympy symbolic representation of the Gaussian log-copula density.

    Builds the expression log c(q1, q2; r) = -1/2 * log(1-r^2)
        - (r^2*(q1^2+q2^2) - 2*r*q1*q2) / (2*(1-r^2))

    in quantile (q1, q2) coordinates, then lazily derives the score S = d log c/dr,
    the hessian H = d^2 log c/dr^2, and the information indicator D = H + S^2.
    All are sympy expressions ready to be substituted with .subs().
    """

    r:     sp.Symbol
    q1_s:  sp.Symbol
    q2_s:  sp.Symbol
    u:     sp.Symbol
    v:     sp.Symbol
    log_c: sp.Expr

    @classmethod
    def build(cls) -> "GaussianCopulaSymbolic":
        r    = sp.Symbol("r",  real=True)
        q1_s = sp.Symbol("q1", real=True)
        q2_s = sp.Symbol("q2", real=True)
        u    = sp.Symbol("u",  positive=True)
        v    = sp.Symbol("v",  positive=True)
        R2    = 1 - r**2
        log_c = -sp.Rational(1, 2) * sp.log(R2) - (
            r**2 * (q1_s**2 + q2_s**2) - 2 * r * q1_s * q2_s
        ) / (2 * R2)
        return cls(r=r, q1_s=q1_s, q2_s=q2_s, u=u, v=v, log_c=log_c)

    def score(self) -> sp.Expr:
        """d log c / dr"""
        return sp.diff(self.log_c, self.r)

    def hessian(self) -> sp.Expr:
        """d^2 log c / dr^2"""
        return sp.diff(self.log_c, self.r, 2)

    def information_indicator(self) -> sp.Expr:
        """H + S^2, the information matrix indicator D_t(theta)"""
        return self.hessian() + self.score()**2

    def mixed_derivative_wrt_u(self) -> sp.Expr:
        """d^2 log c / (dr du), with q1 = Phi^{-1}(u) and q2 = Phi^{-1}(v).

        Substitutes the inverse-CDF expressions into log_c so that sympy can
        differentiate through them with respect to u, then takes d/dr d/du.
        """
        q1_of_u = sp.sqrt(2) * sp.erfinv(2 * self.u - 1)
        q2_of_v = sp.sqrt(2) * sp.erfinv(2 * self.v - 1)
        log_c_prob = self.log_c.subs([(self.q1_s, q1_of_u), (self.q2_s, q2_of_v)])
        return sp.diff(log_c_prob, self.r, self.u)

    def subs(self, expr: sp.Expr, rho: float, q1: float, q2: float) -> float:
        """Evaluate a symbolic expression at a concrete (rho, q1, q2) point."""
        return float(expr.subs([(self.r, rho), (self.q1_s, q1), (self.q2_s, q2)]))


SYMBOLIC_COPULA = GaussianCopulaSymbolic.build()


@pytest.fixture
def standard_test_point():
    return DistributionPoint(u=0.5, v=0.7, rho=0.5, name="Standard case")


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.sort(np.random.uniform(0.1, 0.9, 50))


def numerical_derivative(func: Callable, x: float, h: float = 1e-6) -> float:
    """Central difference approximation of derivative"""
    return (func(x + h) - func(x - h)) / (2 * h)


def get_log_copula_density(u, v, rho):
    """Compute log copula density using multivariate normal."""
    q = [norm.ppf(u), norm.ppf(v)]
    cov = [[1, rho], [rho, 1]]
    log_f_xy = multivariate_normal.logpdf(q, mean=[0, 0], cov=cov)
    log_marginals = norm.logpdf(q[0]) + norm.logpdf(q[1])
    return log_f_xy - log_marginals


def get_numerical_bracket(u, v, rho, h=1e-4):
    """Numerical computation of H + S^2 using finite differences."""
    l_p1 = get_log_copula_density(u, v, rho + h)
    l_m1 = get_log_copula_density(u, v, rho - h)
    l_0 = get_log_copula_density(u, v, rho)

    score = (l_p1 - l_m1) / (2 * h)
    hessian = (l_p1 - 2 * l_0 + l_m1) / (h**2)
    return hessian + score**2


# --- Testing Core Formulas -----------------------------------------------------


class TestCoreFormulas:
    """Tests for the fundamental copula derivative formulas."""

    @pytest.mark.parametrize("test_point", STANDARD_TEST_POINTS)
    def test_score_against_symbolic(self, test_point):
        """Verify score matches symbolic derivation from log copula density."""
        q1, q2, rho = test_point.q1, test_point.q2, test_point.rho
        rtol = 1e-10

        expected = SYMBOLIC_COPULA.subs(SYMBOLIC_COPULA.score(), rho, q1, q2)
        actual = calculate_score(q1, q2, rho)

        assert np.isclose(
            actual, expected, rtol=rtol
        ), f"Score mismatch at {test_point.name}: expected {expected}, got {actual}"

    @pytest.mark.parametrize("test_point", STANDARD_TEST_POINTS)
    def test_hessian_against_symbolic(self, test_point):
        """Verify hessian matches symbolic derivation."""
        q1, q2, rho = test_point.q1, test_point.q2, test_point.rho
        rtol = 1e-10

        expected = SYMBOLIC_COPULA.subs(SYMBOLIC_COPULA.hessian(), rho, q1, q2)
        actual = calculate_hessian(q1, q2, rho)

        assert np.isclose(
            actual, expected, rtol=rtol
        ), f"Hessian mismatch at {test_point.name}: expected {expected}, got {actual}"

    @pytest.mark.parametrize("rho", [0.2, 0.5, 0.8, -0.3])
    def test_gradient_numerical_consistency(self, rho):
        """Test if calculate_grad_D matches finite difference of H + S^2."""
        finite_diff_step = 1e-5
        u, v = 0.3, 0.6
        rtol = 1e-4
        q1, q2 = norm.ppf(u), norm.ppf(v)

        def get_dt(r):
            s = calculate_score(q1, q2, r)
            hess = calculate_hessian(q1, q2, r)
            return hess + s**2

        numerical_grad = (get_dt(rho + finite_diff_step) - get_dt(rho - finite_diff_step)) / (2 * finite_diff_step)
        calculated_grad = calculate_grad_D(q1, q2, rho)

        print(
            f"\n[Rho {rho}] Calculated: {calculated_grad:.6f}, Numerical: {numerical_grad:.6f}"
        )
        assert np.isclose(
            calculated_grad, numerical_grad, rtol=rtol
        ), f"Gradient mismatch: calculated={calculated_grad}, numerical={numerical_grad}"

    @pytest.mark.parametrize("test_point", STANDARD_TEST_POINTS)
    def test_gradient_against_symbolic(self, test_point):
        """Verify gradient matches full symbolic derivation of d(H+S^2)/drho."""
        q1, q2, rho = test_point.q1, test_point.q2, test_point.rho
        rtol = 1e-5

        grad_D_sym = sp.diff(SYMBOLIC_COPULA.information_indicator(), SYMBOLIC_COPULA.r)
        expected = SYMBOLIC_COPULA.subs(grad_D_sym, rho, q1, q2)
        actual = calculate_grad_D(q1, q2, rho)

        assert np.isclose(
            actual, expected, rtol=rtol
        ), f"Gradient mismatch at {test_point.name}: expected {expected}, got {actual}"

    def test_chain_rule_consistency(self, standard_test_point):
        """Verify d(S^2)/drho = 2*S*H (since dS/drho = H)."""
        tp = standard_test_point
        q1, q2, rho = tp.q1, tp.q2, tp.rho
        finite_diff_step = 1e-6
        rtol = 1e-4

        # Numerical derivative of S^2
        score_squared_func = lambda r: calculate_score(q1, q2, r) ** 2
        d_score_sq_numerical = numerical_derivative(score_squared_func, rho, finite_diff_step)

        # Chain rule: 2 * S * H
        S = calculate_score(q1, q2, rho)
        H = calculate_hessian(q1, q2, rho)
        d_score_sq_chain = 2 * S * H

        assert np.isclose(
            d_score_sq_numerical, d_score_sq_chain, rtol=rtol
        ), f"Chain rule failed: numerical={d_score_sq_numerical}, chain_rule={d_score_sq_chain}"

    def test_grad_D_expectation_stability(self):
        """E[grad_D] should converge as sample size grows (law of large numbers check)."""

        def sample_grad_mean(rho, n, seed):
            rng = np.random.default_rng(seed)
            z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
            return np.mean(calculate_grad_D(z[:, 0], z[:, 1], rho))

        rho = 0.5
        n_small = 200
        n_large = 5000
        n_seeds = 20
        convergence_slack = 3.0

        seeds = range(n_seeds)
        small_n_estimates = [sample_grad_mean(rho, n=n_small, seed=s) for s in seeds]
        large_n_estimates = [sample_grad_mean(rho, n=n_large, seed=s) for s in seeds]

        std_small = np.std(small_n_estimates)
        std_large = np.std(large_n_estimates)

        print(f"\nE[grad_D] std at n={n_small}: {std_small:.4f}")
        print(f"E[grad_D] std at n={n_large}: {std_large:.4f}")

        assert not np.any(
            np.isnan(small_n_estimates) | np.isnan(large_n_estimates)
        ), "NaN detected in grad_D estimates"

        assert (
            std_large < std_small
        ), f"E[grad_D] does not converge: std(n={n_small})={std_small:.4f}, std(n={n_large})={std_large:.4f}"

        expected_ratio = np.sqrt(n_large / n_small)
        ratio = std_small / std_large
        print(
            f"Std ratio: {ratio:.2f} (expected ~{expected_ratio:.2f}, min={expected_ratio / convergence_slack:.2f})"
        )

        assert (
            ratio > expected_ratio / convergence_slack
        ), f"Convergence rate suspiciously slow: ratio={ratio:.2f}, expected ~{expected_ratio:.2f}"


# --- W Component Tests ---------------------------------------------------------


class TestWComponent:
    """Tests for the W adjustment component (score estimation correction)."""

    @pytest.mark.parametrize("test_point", STANDARD_TEST_POINTS)
    def test_mixed_derivative_against_symbolic(self, test_point):
        """
        Verify the mixed derivative d^2 log c / (drho du) matches symbolic derivation.
        """
        q1, q2, rho = test_point.q1, test_point.q2, test_point.rho
        rtol = 1e-5

        mixed_sym = SYMBOLIC_COPULA.mixed_derivative_wrt_u()
        expected = float(mixed_sym.subs([
            (SYMBOLIC_COPULA.r, rho),
            (SYMBOLIC_COPULA.u, test_point.u),
            (SYMBOLIC_COPULA.v, test_point.v),
        ]))
        actual = calculate_mixed_derivative(q1, q2, rho)

        print(f"\n[{test_point.name}] Expected: {expected:.6f}, Actual: {actual:.6f}")
        assert np.isclose(
            actual, expected, rtol=rtol
        ), f"Mixed derivative mismatch at {test_point.name}: expected {expected}, got {actual}"

    @pytest.mark.parametrize("rho", [0.3, 0.5, -0.5])
    def test_w_even_symmetry(self, rho):
        """W should be even symmetric: W(F) = W(1-F).

        This follows from the symmetry of the Gaussian copula -- the copula
        density and mixed derivative are symmetric under the transformation
        (u, v) -> (1-u, 1-v), which maps F -> 1-F while preserving the
        integral value.
        """
        test_pairs = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
        rtol = 1e-3
        for f_lo, f_hi in test_pairs:
            w_lo = calculate_W(np.array([f_lo]), rho)[0]
            w_hi = calculate_W(np.array([f_hi]), rho)[0]
            print(f"\n[Rho {rho}] W({f_lo})={w_lo:.6f}, W({f_hi})={w_hi:.6f}")
            assert np.isclose(
                w_lo, w_hi, rtol=rtol
            ), f"Even symmetry violated for rho={rho}: W({f_lo})={w_lo:.6f}, W({f_hi})={w_hi:.6f}"

    @pytest.mark.parametrize("rho_val", [0.0, 0.5, -0.5])
    def test_w_finite(self, rho_val):
        """W should be finite and non-NaN across typical input values."""
        n_points = 20
        data = np.linspace(0.05, 0.95, n_points)
        w = calculate_W(data, rho_val)

        print(f"\n[Rho {rho_val}] W mean: {np.mean(w):.4f}, std: {np.std(w):.4f}")
        assert not np.any(np.isnan(w)), f"NaN in W for rho={rho_val}"
        assert not np.any(np.isinf(w)), f"Inf in W for rho={rho_val}"


# --- M Component Tests ---------------------------------------------------------


class TestMComponent:
    """Tests for the M adjustment component in the asymptotic variance."""

    @pytest.mark.parametrize("test_point", STANDARD_TEST_POINTS)
    def test_grad_D_dq1_against_symbolic(self, test_point):
        """
        Verify d(H+S^2)/dq1 matches symbolic derivation via sympy.
        """
        q1, q2, rho = test_point.q1, test_point.q2, test_point.rho
        rtol = 1e-5

        d_dt_dq1_sym = sp.diff(SYMBOLIC_COPULA.information_indicator(), SYMBOLIC_COPULA.q1_s)
        expected = SYMBOLIC_COPULA.subs(d_dt_dq1_sym, rho, q1, q2)
        actual = calculate_grad_D_dq1(q1, q2, rho)

        print(f"\n[{test_point.name}] Expected: {expected:.6f}, Actual: {actual:.6f}")
        assert np.isclose(
            actual, expected, rtol=rtol
        ), f"d(H+S^2)/dq1 mismatch at {test_point.name}: expected {expected}, got {actual}"

    @pytest.mark.parametrize("rho", [0.2, 0.5, -0.4])
    def test_m_zero_mean_property(self, rho):
        """
        The M component should have near-zero mean (influence function property).
        """
        n_sweep = 500
        grid_res = 400
        mean_zero_atol = 5e-3

        sweep = np.linspace(0.01, 0.99, n_sweep)
        m_values = calculate_M(sweep, rho, grid_res=grid_res)

        mean_val = np.mean(m_values)
        std_val = np.std(m_values)

        print(f"\n[Rho {rho}] M Mean: {mean_val:.2e}, Std: {std_val:.4f}")
        assert np.abs(mean_val) < mean_zero_atol, f"Mean of M too far from zero: {mean_val}"

    def test_m_symmetry_property(self):
        """For Gaussian copula, M(v) should equal M(1-v) (symmetry)."""
        rho = 0.5
        grid_res = 300
        symmetry_atol = 1e-5
        v_points = np.array([0.1, 0.9])

        m_values = calculate_M(v_points, rho, grid_res=grid_res)

        diff = np.abs(m_values[0] - m_values[1])
        print(
            f"\n[Symmetry] M(0.1): {m_values[0]:.6f}, M(0.9): {m_values[1]:.6f}, Diff: {diff:.2e}"
        )

        assert np.isclose(
            m_values[0], m_values[1], atol=symmetry_atol
        ), f"Symmetry failed: M(0.1)={m_values[0]}, M(0.9)={m_values[1]}"

    def test_m_grid_convergence(self):
        """Increasing grid resolution should stabilize M values."""
        rho = 0.5
        grid_res_low = 200
        grid_res_high = 400
        convergence_atol = 1e-3
        v_test = np.array([0.25])

        m_low = calculate_M(v_test, rho, grid_res=grid_res_low)[0]
        m_high = calculate_M(v_test, rho, grid_res=grid_res_high)[0]

        diff = np.abs(m_low - m_high)
        print(
            f"\n[Convergence] Res {grid_res_low}: {m_low:.6f}, Res {grid_res_high}: {m_high:.6f}, Delta: {diff:.2e}"
        )

        assert diff < convergence_atol, f"Grid convergence failed: delta={diff}"

    @pytest.mark.parametrize("rho", [0.2, 0.5, 0.8])
    def test_m_parameter_sensitivity(self, sample_data, rho):
        """M should be finite and non-NaN across a range of correlation values."""
        m_values = calculate_M(sample_data, rho=rho)

        print(
            f"\n[Rho {rho}] M mean: {np.mean(m_values):.2e}, std: {np.std(m_values):.4f}"
        )

        assert not np.any(np.isnan(m_values)), f"NaN detected in M for rho={rho}"
        assert not np.any(np.isinf(m_values)), f"Inf detected in M for rho={rho}"

    @pytest.mark.parametrize("rho_val", [0.0, 0.99, -0.99])
    def test_m_edge_cases(self, sample_data, rho_val):
        """M should handle independence and extreme correlation without NaN/Inf."""
        try:
            m_values = calculate_M(sample_data, rho_val)
            assert not np.any(np.isnan(m_values)), f"NaN detected for rho={rho_val}"
            assert not np.any(np.isinf(m_values)), f"Inf detected for rho={rho_val}"
        except Exception as e:
            pytest.fail(f"M calculation crashed on rho={rho_val}: {e}")


# --- Variance Correction Factor Test -------------------------------------------


class TestCorrectionFactor:
    """Tests for the asymptotic variance (correction factor) calculation."""

    def test_variance_positive(self):
        """Asymptotic variance must be strictly positive."""
        rho = 0.5
        n_grid = 20
        u = np.linspace(0.05, 0.95, n_grid)
        v = np.linspace(0.05, 0.95, n_grid)
        uu, vv = np.meshgrid(u, v)

        v_rho = calculate_correction_factor(uu.flatten(), vv.flatten(), rho)

        print(f"\nAsymptotic Variance V_theta: {v_rho:.6f}")
        assert v_rho > 0, f"Variance must be positive, got {v_rho}"

    @pytest.mark.parametrize("rho", [0.3, 0.7, -0.4])
    def test_psi_properties(self, rho):
        """The correction factor should be well-behaved across rho values."""
        n_grid = 20
        u = np.linspace(0.05, 0.95, n_grid)
        v = np.linspace(0.05, 0.95, n_grid)
        uu, vv = np.meshgrid(u, v)

        v_rho = calculate_correction_factor(uu.flatten(), vv.flatten(), rho)

        assert v_rho > 0, f"Variance not positive for rho={rho}"
        assert not np.isnan(v_rho), f"Variance is NaN for rho={rho}"
        assert not np.isinf(v_rho), f"Variance is Inf for rho={rho}"

    def test_variance_at_independence(self):
        """At rho=0, the variance should still be well-defined."""
        n_samples = 500
        np.random.seed(42)
        u = np.random.uniform(0.1, 0.9, n_samples)
        v = np.random.uniform(0.1, 0.9, n_samples)

        v_rho = calculate_correction_factor(u, v, 0.0)

        print(f"\nVariance at Independence: {v_rho:.6f}")
        assert v_rho > 0, "Variance at independence should be positive"

    def test_empirical_variance_match(self):
        """
        V_rho should match the empirical variance of sqrt(n) * D_bar.
        """
        rho = 0.5
        n = 250
        n_sims = 200
        # Allow 5% relative tolerance for Monte Carlo variance
        variance_rtol = 0.05
        d_bars = []

        np.random.seed(42)
        for _ in range(n_sims):
            z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
            u, v = norm.cdf(z[:, 0]), norm.cdf(z[:, 1])
            q1, q2 = norm.ppf(u), norm.ppf(v)

            s = calculate_score(q1, q2, rho)
            h = calculate_hessian(q1, q2, rho)
            d_bars.append(np.mean(h + s**2))

        empirical_V = np.var(np.array(d_bars)) * n

        # Calculate V_rho on last sample
        sample_Vp = calculate_correction_factor(u, v, rho)

        print(f"\nEmpirical Var: {empirical_V:.6f}, Calculated Vp: {sample_Vp:.6f}")

        assert np.isclose(
            sample_Vp, empirical_V, rtol=variance_rtol
        ), f"Variance mismatch: empirical={empirical_V}, calculated={sample_Vp}"


# --- End-to-End Test -----------------------------------------------------------


class TestEndToEnd:
    """Integration tests for the complete Huang-Prokhorov test."""

    def test_gaussian_copula_not_rejected(self):
        """Data from Gaussian copula should not be rejected."""
        np.random.seed(42)
        n = 500
        rho = 0.5
        min_p_value = 0.01

        z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        x, y = z[:, 0], z[:, 1]

        p_value = huang_prokhorov_test(x, y)

        print(f"\nGaussian Copula Test: p-value = {p_value:.4f}")
        assert p_value > min_p_value, f"Gaussian copula incorrectly rejected: p={p_value}"

    def test_size_under_null(self):
        """
        Under the null, rejection rate should be close to nominal size.
        """
        from scipy.stats import binom

        np.random.seed(123)
        n = 1000
        rho = 0.5
        n_sims = 50
        alpha = 0.05
        # Probability of the meta-test itself raising a false alarm
        meta_alpha = 0.01
        max_rejections = binom.ppf(1 - meta_alpha, n_sims, alpha)

        rejections = 0
        for _ in range(n_sims):
            z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
            p_value = huang_prokhorov_test(z[:, 0], z[:, 1])
            if p_value < alpha:
                rejections += 1

        print(
            f"\nEmpirical Size: {rejections / n_sims:.3f} (nominal: {alpha}, "
            f"max rejections allowed: {int(max_rejections)}/{n_sims})"
        )

        assert (
            rejections <= max_rejections
        ), f"Too many rejections under true null: {rejections}/{n_sims}, max allowed: {int(max_rejections)}"


    def test_numerical_stability_at_tails(self):
        """Test should handle extreme quantile values gracefully."""
        np.random.seed(42)
        n = 100
        rho = 0.5
        clip_limit = 4.0

        z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        z = np.clip(z, -clip_limit, clip_limit)

        try:
            p_value = huang_prokhorov_test(z[:, 0], z[:, 1])
            assert not np.isnan(p_value), "P-value is NaN"
            assert 0 <= p_value <= 1, f"Invalid p-value: {p_value}"
        except Exception as e:
            pytest.fail(f"Test crashed with extreme values: {e}")

