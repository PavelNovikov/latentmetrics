import pytest
import numpy as np
import scipy.stats as st
from latentmetrics.synthesis import Synthesis
from latentmetrics import gauss_rho_to_tau

# --- Distribution Generation Tests ---

@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("normal", {"rho": 0.8}),
        ("lognormal", {"rho": 0.8, "s": 1.0}),
        ("clayton", {"tau": 0.8}),
        ("gumbel", {"tau": 0.8}),
    ],
)
def test_synthesis_shapes_and_types(method, kwargs):
    """Verifies that all methods return two numpy arrays of correct length."""
    n = 50
    func = getattr(Synthesis, method)
    x, y = func(n=n, **kwargs)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == n
    assert len(y) == n


def test_normal_pearson_rho():
    """Tests the Pearson correlation for the Normal distribution."""
    target_rho = 0.7
    n = 10000
    x, y = Synthesis.normal(rho=target_rho, n=n, seed=42)

    sample_rho, _ = st.pearsonr(x, y)
    assert np.isclose(sample_rho, target_rho, atol=0.03)


@pytest.mark.parametrize("target_tau", [0.3, 0.6])
def test_clayton_kendall_tau(target_tau):
    """Tests Kendall's Tau for Clayton Copula."""
    n = 10000
    u1, u2 = Synthesis.clayton(n=n, tau=target_tau, seed=42)

    sample_tau, _ = st.kendalltau(u1, u2)
    assert np.isclose(sample_tau, target_tau, atol=0.03)


@pytest.mark.parametrize("target_tau", [0.3, 0.6])
def test_gumbel_kendall_tau(target_tau):
    """Tests Kendall's Tau for Gumbel Copula."""
    n = 10000
    u1, u2 = Synthesis.gumbel(n=n, tau=target_tau, seed=42)

    sample_tau, _ = st.kendalltau(u1, u2)
    assert np.isclose(sample_tau, target_tau, atol=0.03)


def test_lognormal_kendall_tau():
    """Tests Kendall's Tau for Lognormal using rho-to-tau conversion."""
    latent_rho = 0.5
    expected_tau = gauss_rho_to_tau(latent_rho)
    n = 10000

    x, y = Synthesis.lognormal(rho=latent_rho, n=n, seed=42)

    sample_tau, _ = st.kendalltau(x, y)
    assert np.isclose(sample_tau, expected_tau, atol=0.03)


# --- Discretization Tests ---


def test_discretize_boundaries():
    """Verifies that discretization respects the provided probability splits."""
    data = np.linspace(0, 99, 100)
    probs = [0.2, 0.3, 0.5]

    indices = Synthesis.discretize(data, probs)

    unique, counts = np.unique(indices, return_counts=True)

    # Assert counts match the probabilities [20%, 30%, 50%] exactly
    assert len(unique) == 3
    assert np.array_equal(counts, [20, 30, 50])


def test_discretize_labels():
    """Verifies that string labels are correctly applied."""
    # Data split at the median
    data = np.array([1, 2, 9, 10])
    probs = [0.5, 0.5]
    labels = ["Low", "High"]

    result = Synthesis.discretize(data, probs, labels=labels)

    assert np.array_equal(result, ["Low", "Low", "High", "High"])


def test_discretize_invalid_sum():
    """Raises error if probabilities do not sum to 1."""
    with pytest.raises(ValueError, match="must sum to 1"):
        Synthesis.discretize(np.random.rand(10), [0.1, 0.1])


def test_discretize_label_mismatch():
    """Raises error if labels length doesn't match probabilities."""
    with pytest.raises(ValueError, match="Labels length must match"):
        Synthesis.discretize(np.random.rand(10), [0.5, 0.5], labels=["A"])
