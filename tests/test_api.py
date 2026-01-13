import pytest
from latentmetrics import make_corr_fn, VariableType, EstimateMethod, CorrResult


def test_factory_returns_callable():
    metric = make_corr_fn(
        VariableType.CONTINUOUS, VariableType.CONTINUOUS, EstimateMethod.RANK
    )
    assert callable(metric)


def test_result_structure():
    metric = make_corr_fn(
        VariableType.CONTINUOUS, VariableType.BINARY, EstimateMethod.VALUE
    )
    res = metric([1, 2, 3], [0, 1, 0])
    assert isinstance(res, CorrResult)
    assert isinstance(res.estimate, float)
