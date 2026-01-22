import numpy as np
import pytest
from latentmetrics import make_corr_fn, VariableType, EstimateMethod
from latentmetrics.diagnostics import get_par_bootstrap_gof_fn, get_asymptotic_gof_fn
from latentmetrics.synthesis import Synthesis

N_SAMPLES = 2000
SEED = 42
PROBS = [0.3, 0.4, 0.3]


class DataGeneratorFactory:

    @staticmethod
    def get_rank_fitter(type_x: VariableType, type_y: VariableType):
        base_fn = make_corr_fn(type_x, type_y, EstimateMethod.RANK)
        return lambda x, y: base_fn(x, y).estimate

    @staticmethod
    def get_generator(probs_x=None, probs_y=None, dist="normal"):
        def generate_fn(rho_or_tau: float, n: int, g: np.random.Generator):
            if dist == "normal":
                x_cont, y_cont = Synthesis.normal(rho_or_tau, n, seed=g)
            elif dist == "lognormal":
                x_cont, y_cont = Synthesis.lognormal(rho_or_tau, n, seed=g)
            elif dist == "clayton":
                x_cont, y_cont = Synthesis.clayton(n, rho_or_tau)
            elif dist == "gumbel":
                x_cont, y_cont = Synthesis.gumbel(n, rho_or_tau)

            x_out = Synthesis.discretize(x_cont, probs_x) if probs_x else x_cont
            y_out = Synthesis.discretize(y_cont, probs_y) if probs_y else y_cont
            return x_out, y_out

        return generate_fn


@pytest.mark.parametrize(
    "type_x, type_y, probs_x, probs_y",
    [
        (VariableType.CONTINUOUS, VariableType.CONTINUOUS, None, None),
        (VariableType.ORDINAL, VariableType.CONTINUOUS, PROBS, None),
        (VariableType.ORDINAL, VariableType.ORDINAL, PROBS, PROBS),
    ],
)
@pytest.mark.parametrize(
    "dist, should_pass",
    [
        ("lognormal", True),
        ("clayton", False),
        ("gumbel", False),
    ],
)
def test_parametric_bootstrap_gof(type_x, type_y, probs_x, probs_y, dist, should_pass):
    fit_fn = DataGeneratorFactory.get_rank_fitter(type_x, type_y)
    gen_null = DataGeneratorFactory.get_generator(
        probs_x=probs_x, probs_y=probs_y, dist="normal"
    )
    gen_alt = DataGeneratorFactory.get_generator(
        probs_x=probs_x, probs_y=probs_y, dist=dist
    )

    rng = np.random.default_rng(SEED)
    x, y = gen_alt(0.6, N_SAMPLES, rng)

    gof_test = get_par_bootstrap_gof_fn(type_x, type_y)
    p_val = gof_test(x, y, fit_fn, gen_null, n_boot=100, seed=SEED)

    if should_pass:
        assert p_val > 0.05
    else:
        assert p_val < 0.05


@pytest.mark.parametrize(
    "type_x, type_y, probs_x, probs_y",
    [
        (VariableType.CONTINUOUS, VariableType.CONTINUOUS, None, None),
        (VariableType.ORDINAL, VariableType.CONTINUOUS, PROBS, None),
        (VariableType.ORDINAL, VariableType.ORDINAL, PROBS, PROBS),
    ],
)
@pytest.mark.parametrize(
    "dist, should_pass",
    [
        ("lognormal", True),
        ("clayton", False),
        ("gumbel", False),
    ],
)
def test_fast_gof_suite(type_x, type_y, probs_x, probs_y, dist, should_pass):
    gen = DataGeneratorFactory.get_generator(
        probs_x=probs_x, probs_y=probs_y, dist=dist
    )
    rng = np.random.default_rng(SEED)

    x, y = gen(0.5, N_SAMPLES, rng)
    gof_test = get_asymptotic_gof_fn(type_x, type_y)
    res = gof_test(x, y)

    if should_pass:
        assert res > 0.05
    else:
        assert res < 0.05
