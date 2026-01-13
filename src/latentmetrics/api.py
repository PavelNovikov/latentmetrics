import numpy as np
from typing import Callable
from numpy.typing import ArrayLike
from .types import VariableType, EstimateMethod, CorrResult
from .registry import CORR_REGISTRY


def make_corr_fn(
    x_type: VariableType,
    y_type: VariableType,
    method: EstimateMethod = EstimateMethod.VALUE,
) -> Callable[[ArrayLike, ArrayLike], CorrResult]:

    impl = CORR_REGISTRY.get((method, x_type, y_type))

    if not impl:
        raise NotImplementedError(
            f"No implementation for {method} with {x_type} and {y_type}"
        )

    def metric(x: ArrayLike, y: ArrayLike) -> CorrResult:
        x_arr, y_arr = np.asarray(x), np.asarray(y)

        val = impl(x_arr, y_arr)

        return CorrResult(
            estimate=float(val),
            method=method,
            x_type=x_type,
            y_type=y_type,
            resolved_name=getattr(impl, "__name__", "lambda_wrapper"),
        )

    return metric
