from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.stats import norm, multivariate_normal


# ---------------------------------------------------------------------
# Latent Space Visualization Functions
# ---------------------------------------------------------------------


def get_latent_edges(data: ArrayLike, bins_cont: int = 40) -> np.ndarray:
    """
    Calculate bin edges for the latent Gaussian space.
    """
    data = np.asarray(data)
    unique_vals = np.sort(np.unique(data))
    n = data.size

    if len(unique_vals) <= bins_cont:
        # Calculate thresholds based on cumulative probabilities
        counts = np.array([np.sum(data <= v) for v in unique_vals])
        probs = counts / n

        # Map to Z-space, excluding the final 1.0 to avoid +inf
        thresholds = norm.ppf(probs[:-1])

        # Dynamic boundaries with buffer
        z_min = min(-4.5, np.min(thresholds) - 0.5) if thresholds.size > 0 else -4.5
        z_max = max(4.5, np.max(thresholds) + 0.5) if thresholds.size > 0 else 4.5

        z_edges = np.concatenate([[z_min], thresholds, [z_max]])
    else:
        # Standard range for continuous data
        z_edges = np.linspace(-4.5, 4.5, bins_cont + 1)

    return np.unique(np.sort(z_edges))


def plot_latent_density(
    x: ArrayLike,
    y: ArrayLike,
    title: str,
    rho: Optional[float] = None,
    decimal_points: int = 2,
    ax_empirical: Optional[plt.Axes] = None,
    ax_theoretical: Optional[plt.Axes] = None,
) -> None:
    """
    Plot the empirical latent density by mapping observations to Z-space.
    Optionally compares against a theoretical Gaussian density.
    """
    # 1. Validation Logic
    if ax_theoretical is not None:
        if rho is None or ax_empirical is None:
            raise ValueError(
                "Providing 'ax_theoretical' requires both 'rho' and 'ax_empirical' to be provided."
            )

    if rho is not None and ax_empirical is not None and ax_theoretical is None:
        raise ValueError(
            "When 'rho' and 'ax_empirical' are both provided, 'ax_theoretical' must also be provided."
        )

    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size

    # 2. Map Marginals to Latent Space (Z-space)
    u_x = np.argsort(np.argsort(x)) / (n + 1)
    u_y = np.argsort(np.argsort(y)) / (n + 1)
    z_x, z_y = norm.ppf(u_x), norm.ppf(u_y)

    # 3. Get Edges
    x_edges = get_latent_edges(x)
    y_edges = get_latent_edges(y)

    # 4. Compute Empirical 2D Histogram
    hist_empirical, _, _ = np.histogram2d(
        z_x, z_y, bins=[x_edges, y_edges], density=True
    )
    X, Y = np.meshgrid(x_edges, y_edges)

    # 5. Internal Figure Setup
    if ax_empirical is None:
        num_plots = 2 if rho is not None else 1
        fig, axes = plt.subplots(
            1, num_plots, figsize=(7 * num_plots, 5), squeeze=False
        )
        ax_empirical = axes[0, 0]
        if rho is not None:
            ax_theoretical = axes[0, 1]

    # 6. Plot Empirical Subplot
    mesh0 = ax_empirical.pcolormesh(
        X, Y, hist_empirical.T, cmap="viridis", shading="flat"
    )
    plt.colorbar(mesh0, ax=ax_empirical, label="Density")
    ax_empirical.set_title(title)

    # 7. Compute and Plot Theoretical Subplot
    if rho is not None:
        rv = multivariate_normal([0, 0], [[1, rho], [rho, 1]])
        hist_theoretical = np.zeros((len(x_edges) - 1, len(y_edges) - 1))

        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                p_tr = rv.cdf(np.array([x_edges[i + 1], y_edges[j + 1]]))
                p_tl = rv.cdf(np.array([x_edges[i], y_edges[j + 1]]))
                p_br = rv.cdf(np.array([x_edges[i + 1], y_edges[j]]))
                p_bl = rv.cdf(np.array([x_edges[i], y_edges[j]]))

                prob = p_tr - p_tl - p_br + p_bl
                area = (x_edges[i + 1] - x_edges[i]) * (y_edges[j + 1] - y_edges[j])
                hist_theoretical[i, j] = prob / area if area > 0 else 0

        mesh1 = ax_theoretical.pcolormesh(
            X,
            Y,
            hist_theoretical.T,
            cmap="viridis",
            shading="flat",
            vmax=np.max(hist_empirical),
        )
        plt.colorbar(mesh1, ax=ax_theoretical, label="Density")
        # Apply decimal points formatting here
        ax_theoretical.set_title(
            rf"Theoretical Gaussian ($\rho={rho:.{decimal_points}f}$)"
        )

    # 8. Aesthetics
    active_axes = [ax_empirical]
    if ax_theoretical is not None:
        active_axes.append(ax_theoretical)

    for ax in active_axes:
        ax.set_xlabel(r"Latent $Z_x$")
        ax.set_ylabel(r"Latent $Z_y$")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.grid(False)
