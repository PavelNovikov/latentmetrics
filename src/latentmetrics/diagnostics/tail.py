from typing import Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.stats import multivariate_normal, norm
from scipy.optimize import root_scalar

# ---------------------------------------------------------------------
# Tail Concentration and Dependence Functions
# ---------------------------------------------------------------------


def plot_tail_concentration_function(
    x: ArrayLike,
    y: ArrayLike,
    resolution: int = 200,
    rho: Optional[float] = None,
    decimal_points: int = 2,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the empirical tail concentration function q_C(t) based on the
    copula diagonal.

    Reference
    ----------
    Durante, F., Fernandez-Sanchez, J., & Pappada, R. (2015). 
    Copulas, diagonals, and tail dependence. 
    Fuzzy Sets and Systems, 264, 22-41.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs x and y must be one-dimensional.")

    n = x.size
    # Transform to pseudo-observations (empirical copula domain)
    u = np.argsort(np.argsort(x)) / (n - 1)
    v = np.argsort(np.argsort(y)) / (n - 1)

    t = np.linspace(1e-3, 1 - 1e-3, resolution)

    # Compute empirical diagonal section delta(t) = C(t, t)
    delta_empirical = np.array([np.mean((u <= ti) & (v <= ti)) for ti in t])

    # Compute tail concentration function q(t)
    q_empirical = np.where(
        t <= 0.5, delta_empirical / t, (1 - 2 * t + delta_empirical) / (1 - t)
    )

    ax.plot(
        t,
        q_empirical,
        label=r"Empirical tail concentration function $q_C(t)$",
        color="steelblue",
        lw=2,
    )

    if rho is not None:
        mvn = multivariate_normal(cov=[[1.0, rho], [rho, 1.0]])
        z = norm.ppf(t)
        delta_gaussian = mvn.cdf(np.column_stack([z, z]))

        q_gaussian = np.where(
            t <= 0.5, delta_gaussian / t, (1 - 2 * t + delta_gaussian) / (1 - t)
        )
        ax.plot(
            t, 
            q_gaussian, 
            "k--", 
            alpha=0.8, 
            label=rf"Gaussian copula ($\rho={rho:.{decimal_points}f}$)"
        )

    ax.set_title("Tail concentration function")
    ax.axvline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$q_C(t)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True)
    
    return ax


def plot_k_plot(
    x: ArrayLike,
    y: ArrayLike,
    rho: Optional[float] = None,
    decimal_points: int = 2,
    mode: Literal["standard", "survival"] = "survival",
    n_sim: int = 5000,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Generate a Kendall Plot (K-Plot) to detect dependence structure.

    Reference
    ----------
    Genest, C., & Favre, A. C. (2007). 
    Everything you always wanted to know about copula modeling but were afraid to ask. 
    Journal of hydrologic engineering, 12(4), 347-368.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

   
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    
    # 1. Compute Empirical Joint Values (H_i)
    if mode == "standard":
        h = np.array([np.sum((x <= x[i]) & (y <= y[i])) / n for i in range(n)])
    else:
        h = np.array([np.sum((x > x[i]) & (y > y[i])) / n for i in range(n)])
    h_ordered = np.sort(h)
    
    # 2. Independence Reference Functions
    def k0_dist(w):
        return w - w * np.log(w) if w > 1e-12 else 0.0
    
    def get_w_expected(p):
        if p <= 1e-10: return 0.0
        if p >= 1 - 1e-10: return 1.0
        return root_scalar(lambda w: k0_dist(w) - p, bracket=[1e-12, 1.0]).root
    
    w_expected = np.array([get_w_expected(i / (n + 1)) for i in range(1, n + 1)])
    
    # Reference: Independence (Diagonal)
    ax.plot([0, 1], [0, 1], color="green", ls="--", label="Independence")
    
    # Reference: Perfect Positive Dependence
    t_ref = np.linspace(1e-9, 1.0, 200)
    ax.plot(
        t_ref,
        [k0_dist(t) for t in t_ref],
        color="blue",
        ls=":",
        lw=1.5,
        label="Perfect Positive Dependence",
    )
    
    # 3. Gaussian Reference Curve
    if rho is not None:
        sim_data = multivariate_normal.rvs(cov=[[1, rho], [rho, 1]], size=n_sim)
        xs, ys = sim_data[:, 0], sim_data[:, 1]
        if mode == "standard":
            w_sim = np.array([np.sum((xs <= xs[i]) & (ys <= ys[i])) / n_sim for i in range(n_sim)])
        else:
            w_sim = np.array([np.sum((xs > xs[i]) & (ys > ys[i])) / n_sim for i in range(n_sim)])
        gauss_h = np.sort(w_sim)
        gauss_w = np.array([get_w_expected(p) for p in np.linspace(1/(n_sim+1), n_sim/(n_sim+1), n_sim)])
        ax.plot(
            gauss_w, 
            gauss_h, 
            color="gray", 
            lw=2, 
            label=rf"Gaussian ($\rho={rho:.{decimal_points}f}$)"
        )
    
    # 4. Actual Data Points
    ax.scatter(
        w_expected,
        h_ordered,
        color="red",
        s=12,
        alpha=0.4,
        label=f"Observed Data ({mode})",
    )
    
    ax.set_title("K-plot")
    ax.set_xlabel(r"Expected $W_{i:n}$ (Independence)")
    ax.set_ylabel(r"Ordered Empirical $H_{(i)}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    
    # Legend in lower right corner
    ax.legend(loc='lower right', frameon=True, fontsize=8)
    
    
    return ax