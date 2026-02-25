import numpy as np
import pytest
from scipy.stats import kendalltau as scipy_kendalltau

from latentmetrics.engines.utils import compute_pairwise_concordance, compute_tau_a


# --- Helpers -----------------------------------------------------------------


def make_perfectly_ordered(n_per_class: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Continuous x strictly increases with ordinal y classes — theta[i,j] = -1 for i<j."""
    y = np.repeat([0, 1, 2], n_per_class)
    x = np.concatenate(
        [
            np.linspace(0, 1, n_per_class),
            np.linspace(1.5, 2.5, n_per_class),
            np.linspace(3, 4, n_per_class),
        ]
    )
    return x, y


def make_perfectly_reversed(n_per_class: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Continuous x strictly decreases with ordinal y classes — theta[i,j] = +1 for i<j."""
    y = np.repeat([0, 1, 2], n_per_class)
    x = np.concatenate(
        [
            np.linspace(3, 4, n_per_class),
            np.linspace(1.5, 2.5, n_per_class),
            np.linspace(0, 1, n_per_class),
        ]
    )
    return x, y


def _tau_a_from_tau_c(x: np.ndarray, y: np.ndarray) -> float:
    """
    Reconstructs tau_a from scipy's tau_c using the exact relationship:
        tau_a = tau_c * n*(K-1) / ((n-1)*K)
    where K = number of unique classes in y (the ordinal variable).
    """
    n = len(y)
    K = len(np.unique(y))
    tc = scipy_kendalltau(x, y, variant="c").statistic
    return float(tc * n * (K - 1) / ((n - 1) * K))


# --- compute_pairwise_concordance: shape and classes -------------------------


def test_output_shapes_binary():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0, 0, 1, 1, 1])
    theta, classes = compute_pairwise_concordance(x, y)
    assert theta.shape == (2, 2)
    assert len(classes) == 2


def test_output_shapes_multiclass():
    x = np.random.default_rng(0).standard_normal(40)
    y = np.repeat([0, 1, 2, 3], 10)
    theta, classes = compute_pairwise_concordance(x, y)
    assert theta.shape == (4, 4)
    np.testing.assert_array_equal(classes, [0, 1, 2, 3])


def test_diagonal_is_zero():
    x, y = make_perfectly_ordered()
    theta, _ = compute_pairwise_concordance(x, y)
    np.testing.assert_array_equal(np.diag(theta), 0.0)


def test_non_contiguous_class_labels():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([10, 10, 20, 20, 30, 30])
    theta, classes = compute_pairwise_concordance(x, y)
    np.testing.assert_array_equal(classes, [10, 20, 30])
    assert theta[0, 2] == pytest.approx(-1.0)


# --- compute_pairwise_concordance: boundary values ---------------------------


def test_perfect_concordance():
    """x increases with y class => class i is entirely below class j => theta[i,j] = -1."""
    x, y = make_perfectly_ordered()
    theta, _ = compute_pairwise_concordance(x, y)
    assert theta[0, 1] == pytest.approx(-1.0)
    assert theta[0, 2] == pytest.approx(-1.0)
    assert theta[1, 2] == pytest.approx(-1.0)


def test_perfect_discordance():
    """x decreases with y class => class i is entirely above class j => theta[i,j] = +1."""
    x, y = make_perfectly_reversed()
    theta, _ = compute_pairwise_concordance(x, y)
    assert theta[0, 1] == pytest.approx(1.0)
    assert theta[0, 2] == pytest.approx(1.0)
    assert theta[1, 2] == pytest.approx(1.0)


def test_antisymmetry():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(90)
    y = np.repeat([0, 1, 2], 30)
    theta, _ = compute_pairwise_concordance(x, y)
    np.testing.assert_allclose(theta, -theta.T, atol=1e-12)


def test_values_in_range():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(100)
    y = np.repeat([0, 1, 2, 3], 25)
    theta, _ = compute_pairwise_concordance(x, y)
    assert np.all(theta >= -1.0 - 1e-12)
    assert np.all(theta <= 1.0 + 1e-12)


# --- compute_pairwise_concordance: tie handling ------------------------------


def test_all_ties_gives_zero():
    x = np.ones(6)
    y = np.array([0, 0, 1, 1, 2, 2])
    theta, _ = compute_pairwise_concordance(x, y)
    off_diag = theta[~np.eye(3, dtype=bool)]
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)


def test_partial_ties():
    """Class 0 x-values are strictly below class 1 despite internal ties => theta[0,1] = -1."""
    x = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    theta, _ = compute_pairwise_concordance(x, y)
    assert theta[0, 1] == pytest.approx(-1.0)


# --- compute_pairwise_concordance: edge cases --------------------------------


def test_single_sample_per_class():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0, 1, 2])
    theta, _ = compute_pairwise_concordance(x, y)
    assert theta.shape == (3, 3)
    assert theta[0, 1] == pytest.approx(-1.0)
    assert theta[1, 0] == pytest.approx(1.0)


# --- compute_tau_a: correctness ----------------------------------------------


def test_tau_a_matches_tau_c_small():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([0, 0, 1, 1, 2, 2])
    assert compute_tau_a(x, y) == pytest.approx(_tau_a_from_tau_c(x, y), abs=1e-6)


def test_tau_a_matches_tau_c_random():
    rng = np.random.default_rng(123)
    x = rng.standard_normal(80)
    y = np.repeat([0, 1, 2, 3], 20)
    assert compute_tau_a(x, y) == pytest.approx(_tau_a_from_tau_c(x, y), abs=1e-6)


def test_tau_a_perfectly_ordered():
    """One sample per class, strictly increasing x => tau_a = 1.0."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0, 1, 2, 3, 4])
    assert compute_tau_a(x, y) == pytest.approx(1.0)


def test_tau_a_perfectly_reversed():
    """One sample per class, strictly decreasing x => tau_a = -1.0."""
    x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    y = np.array([0, 1, 2, 3, 4])
    assert compute_tau_a(x, y) == pytest.approx(-1.0)


# --- compute_tau_a: weights --------------------------------------------------


def test_tau_a_uniform_weights_in_range():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(90)
    y = np.repeat([0, 1, 2], 30)
    assert -1.0 <= compute_tau_a(x, y, weights=[1.0, 1.0, 1.0]) <= 1.0


def test_tau_a_weights_equivalent_to_replication():
    """weights=(1,2,3) should match physically replicating classes 1x, 2x, 3x."""
    x_base = np.array([1.0, 2.0, 3.0])
    y_base = np.array([0, 1, 2])
    x_rep = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    y_rep = np.array([0, 1, 1, 2, 2, 2])
    assert compute_tau_a(x_base, y_base, weights=[1.0, 2.0, 3.0]) == pytest.approx(
        compute_tau_a(x_rep, y_rep), abs=1e-6
    )


def test_tau_a_weights_equivalent_to_replication_random_x():
    """Same as above but with random x and more classes."""
    rng = np.random.default_rng(0)
    K, multipliers = 5, [1, 2, 3, 4, 5]
    y_base = np.arange(K)
    x_base = rng.standard_normal(K)
    y_rep = np.repeat(y_base, multipliers)
    x_rep = np.repeat(x_base, multipliers)
    assert compute_tau_a(
        x_base, y_base, weights=np.array(multipliers, dtype=float)
    ) == pytest.approx(compute_tau_a(x_rep, y_rep), abs=1e-6)


def test_tau_a_custom_weights_change_value():
    """Uniform weights on skewed class sizes should differ from empirical weights."""
    rng = np.random.default_rng(42)
    y = np.repeat([0, 1, 2], [10, 20, 70])
    x = rng.standard_normal(100)
    tau_empirical = compute_tau_a(x, y)
    tau_uniform = compute_tau_a(x, y, weights=[1.0, 1.0, 1.0])
    assert tau_empirical != pytest.approx(tau_uniform, abs=1e-3)


def test_tau_a_wrong_weights_length():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="weights length"):
        compute_tau_a(x, y, weights=[1.0, 1.0, 1.0])
