import numpy as np
import pytest
from latentmetrics import make_corr_fn, VariableType, EstimateMethod

# =====================================================
# Utilities
# =====================


def discretize(data, bins, balanced=False):
    if bins < 2:
        return np.zeros_like(data, dtype=int)

    if balanced:
        # Equal split (e.g., 0.5 for binary)
        q = np.linspace(0, 1, bins + 1)[1:-1]
    else:
        # Original 90/10 split
        remaining_mass = 0.1
        steps = bins - 1
        q = [0.9 + (i * remaining_mass / steps) for i in range(steps)]

    thresholds = np.quantile(data, q)
    return np.searchsorted(thresholds, data)


def make_observed(x_latent, y_latent, x_type, y_type, balanced=False):
    def convert(z, vtype):
        if vtype == VariableType.BINARY:
            return discretize(z, 2, balanced=balanced)
        if vtype == VariableType.ORDINAL:
            return discretize(z, 4, balanced=balanced)
        return z

    return convert(x_latent, x_type), convert(y_latent, y_type)


def run_correlation_test(
    request,
    dist_name,
    x_type,
    y_type,
    method,
    rho_true,
    expected_pass,
    balanced=False,
    atol_pass=0.03,
    atol_fail=0.05,
):
    """Helper to run the core logic of the test."""
    generator = request.getfixturevalue(dist_name)
    x_latent, y_latent = generator(rho_true)

    # Pass the balanced flag here
    x, y = make_observed(x_latent, y_latent, x_type, y_type, balanced=balanced)

    calc_fn = make_corr_fn(x_type, y_type, method)
    res = calc_fn(x, y)
    r1 = res.estimate

    # Determinism Check
    r2 = calc_fn(x, y).estimate
    assert r2 == pytest.approx(r1, abs=1e-15)

    # Reconstruction Check
    if expected_pass:
        assert r1 == pytest.approx(rho_true, abs=atol_pass)
    else:
        assert abs(r1 - rho_true) > atol_fail


# =====================================================
# Test Suites
# =====================================================


@pytest.mark.parametrize("method", [EstimateMethod.VALUE, EstimateMethod.RANK])
@pytest.mark.parametrize(
    "x_type", [VariableType.CONTINUOUS, VariableType.ORDINAL, VariableType.BINARY]
)
@pytest.mark.parametrize(
    "y_type", [VariableType.CONTINUOUS, VariableType.ORDINAL, VariableType.BINARY]
)
def test_gaussian_full_suite(request, method, x_type, y_type):
    run_correlation_test(
        request, "gaussian_dist", x_type, y_type, method, 0.5, expected_pass=True
    )


@pytest.mark.parametrize("method", [EstimateMethod.VALUE, EstimateMethod.RANK])
@pytest.mark.parametrize(
    "types",
    [
        (VariableType.BINARY, VariableType.CONTINUOUS),
        (VariableType.BINARY, VariableType.BINARY),
    ],
)
def test_gumbel_reconstruction_fails(request, method, types):
    x_type, y_type = types
    run_correlation_test(
        request, "gumbel_copula", x_type, y_type, method, 0.5, expected_pass=False
    )


@pytest.mark.parametrize(
    "types",
    [
        (VariableType.BINARY, VariableType.CONTINUOUS),
        (VariableType.CONTINUOUS, VariableType.CONTINUOUS),
    ],
    ids=["BINARY-CONTINUOUS", "CONTINUOUS-CONTINUOUS"],
)
@pytest.mark.parametrize(
    "method, should_pass",
    [(EstimateMethod.RANK, True), (EstimateMethod.VALUE, False)],
    ids=["RANK", "VALUE"],
)
def test_lognormal_rank_pass_value_fail(request, types, method, should_pass):
    x_type, y_type = types
    # Set balanced=True specifically for this suite
    run_correlation_test(
        request,
        "lognormal_gaussian_copula",
        x_type,
        y_type,
        method,
        0.5,
        expected_pass=should_pass,
        balanced=True,
    )
