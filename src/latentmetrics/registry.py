from .types import VariableType, EstimateMethod
from .engines import rank_based as rb
from .engines import value_based as vb
import functools


def swap_args(func):
    """Returns a new function that swaps the first two arguments."""

    @functools.wraps(func)
    def wrapper(a, b, *args, **kwargs):
        return func(b, a, *args, **kwargs)

    return wrapper


CORR_REGISTRY = {
    # Rank-based mappings
    (
        EstimateMethod.RANK,
        VariableType.CONTINUOUS,
        VariableType.CONTINUOUS,
    ): rb.latent_rank_cc,
    (
        EstimateMethod.RANK,
        VariableType.CONTINUOUS,
        VariableType.ORDINAL,
    ): rb.latent_rank_co,
    (
        EstimateMethod.RANK,
        VariableType.ORDINAL,
        VariableType.CONTINUOUS,
    ): swap_args(rb.latent_rank_co),
    (
        EstimateMethod.RANK,
        VariableType.ORDINAL,
        VariableType.ORDINAL,
    ): rb.latent_rank_oo,
    (
        EstimateMethod.RANK,
        VariableType.BINARY,
        VariableType.ORDINAL,
    ): rb.latent_rank_oo,
    (
        EstimateMethod.RANK,
        VariableType.ORDINAL,
        VariableType.BINARY,
    ): rb.latent_rank_oo,
    (
        EstimateMethod.RANK,
        VariableType.CONTINUOUS,
        VariableType.BINARY,
    ): rb.latent_rank_cb,
    (
        EstimateMethod.RANK,
        VariableType.BINARY,
        VariableType.CONTINUOUS,
    ): swap_args(rb.latent_rank_cb),
    (EstimateMethod.RANK, VariableType.BINARY, VariableType.BINARY): rb.latent_rank_bb,
    # Value-based mappings
    (
        EstimateMethod.VALUE,
        VariableType.CONTINUOUS,
        VariableType.CONTINUOUS,
    ): vb.pearson_correlation,
    (
        EstimateMethod.VALUE,
        VariableType.CONTINUOUS,
        VariableType.ORDINAL,
    ): vb.polyserial_correlation,
    (
        EstimateMethod.VALUE,
        VariableType.ORDINAL,
        VariableType.CONTINUOUS,
    ): swap_args(vb.polyserial_correlation),
    (
        EstimateMethod.VALUE,
        VariableType.ORDINAL,
        VariableType.ORDINAL,
    ): vb.polychoric_correlation,
    (
        EstimateMethod.VALUE,
        VariableType.BINARY,
        VariableType.ORDINAL,
    ): vb.polychoric_correlation,
    (
        EstimateMethod.VALUE,
        VariableType.ORDINAL,
        VariableType.BINARY,
    ): vb.polychoric_correlation,
    (
        EstimateMethod.VALUE,
        VariableType.CONTINUOUS,
        VariableType.BINARY,
    ): vb.biserial_correlation,
    (
        EstimateMethod.VALUE,
        VariableType.BINARY,
        VariableType.CONTINUOUS,
    ): swap_args(vb.biserial_correlation),
    (
        EstimateMethod.VALUE,
        VariableType.BINARY,
        VariableType.BINARY,
    ): vb.tetrachoric_correlation,
}
