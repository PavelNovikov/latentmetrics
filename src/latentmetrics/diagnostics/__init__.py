"""
Diagnostic tools and visualizations for latent correlation models.
"""

import importlib.util
from typing import Any, Callable
from .par_bootstrap_gof import get_gof_test_fn as get_par_bootstrap_gof_fn
from .asymptotic_gof import get_gof_test_fn as get_asymptotic_gof_fn


def _has_matplotlib() -> bool:
    """Check if matplotlib is installed in the current environment."""
    return importlib.util.find_spec("matplotlib") is not None


if _has_matplotlib():
    from .tail import plot_tail_concentration_function, plot_k_plot
    from .histograms import plot_latent_density

    __all__ = [
        "plot_tail_concentration_function",
        "plot_k_plot",
        "plot_latent_density",
    ]
else:

    def _create_missing_dependency_error(func_name: str) -> Callable[..., Any]:
        """Create a placeholder function that raises ImportError when called."""

        def missing_dependency_func(*args: Any, **kwargs: Any) -> None:
            raise ImportError(
                f"The '{func_name}' function requires 'matplotlib'. "
                "Please install the optional diagnostics dependencies using:\n\n"
                "  pip install 'latentmetrics[diagnostics]'"
            )

        return missing_dependency_func

    plot_tail_concentration_function = _create_missing_dependency_error(
        "plot_tail_concentration_function"
    )
    plot_k_plot = _create_missing_dependency_error("plot_k_plot")
    plot_latent_density = _create_missing_dependency_error(
        "plot_latent_correlation_structure"
    )

    __all__ = [
        "plot_tail_concentration_function",
        "plot_k_plot",
        "plot_latent_density",
        "get_par_bootstrap_gof_fn",
        "get_asymptotic_gof_fn",
    ]
