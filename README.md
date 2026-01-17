[![Tests](https://github.com/PavelNovikov/latentmetrics/actions/workflows/tests.yml/badge.svg)](https://github.com/PavelNovikov/latentmetrics/actions/workflows/tests.yml)

# Latent Correlation Estimation Package

This package provides **minimalistic implementations of latent correlation estimators** between pairs of continuous variables when one or both variables are **discretized**. 

The main goal is to offer **both value-based and rank-based correlation estimates in one place**, with a **simple and easy-to-understand implementation**.

> Note: Binary data is a special case of ordinal data. For clarity, we implemented binary correlations separately as a simpler example.

## Supported Estimators

### 1. Value-Based Correlations
- **Tetrachoric**
- **Polychoric**
- **Biserial**
- **Polyserial**

These estimators assume a **bivariate normal underlying distribution** and estimate the correlation parameter ($\rho$) by **maximizing the likelihood** of the observed discretized data.

### 2. Rank-Based Correlations
- Suitable for **binary, ordinal, and mixed (ordinal-continuous) data**, and also supports **continuous data** through **Greiner's formula**.
- Based on the assumption that data arises from an **arbitrary monotonic transformation** of underlying bivariate Gaussian variables (Gaussian copula model).
- The correlation parameter ($\rho$) is estimated by **matching the observed Kendall's tau** to its expected value as a function of $\rho$. This makes these estimators invariant to marginal transformations.

---

## Synthetic Data Scenarios

The package includes utilities for generating synthetic data to test estimator robustness across different distributional assumptions:

1.  **Normal Distribution**: Satisfies the underlying assumptions for both value-based and rank-based methods.
2.  **Lognormal Distribution**: Violates the assumptions of value-based methods due to non-normality, but holds for rank-based methods.
3.  **Gumbel and Clayton Copulas**: Violate the assumptions of both families of estimators due to the presence of tail dependence, which deviates from the Gaussian copula model.

---

## Diagnostics & Model Fit

To verify if the data aligns with the theoretical assumptions of the Gaussian copula model, the package provides tools to visualize and compare the observed data against theoretical expectations:

1.  **Tail Concentration Function**: (For continuous data) Helps detect if dependence is concentrated in the tails, which would contradict a Gaussian assumption.
2.  **K-plot (Kendall plot)**: (For continuous data) A diagnostic tool to assess the dependency structure independently of marginal distributions.
3.  **Latent Space Comparison**: For both continuous and discrete data, the package allows for a direct visual check of the Gaussian copula assumption. This is done by comparing 2D histograms or empirical categorical densities against the theoretical densities expected under the estimated latent correlation. The comparison is performed in the latent space by mapping observed data via rank transformation.

---


## Installation

```bash
pip install latentmetrics[all]
```

## Usage

```python
import matplotlib.pyplot as plt
from latentmetrics import make_corr_fn, VariableType, EstimateMethod, gauss_rho_to_tau
from latentmetrics.synthesis import Synthesis
from latentmetrics.diagnostics import (
    plot_tail_concentration_function,
    plot_k_plot,
    plot_latent_density,
)

N_SAMPLES = 1000
RHO_TRUE = 0.4

# --- Data Synthesis ---
# x, y = Synthesis.gumbel(tau=gauss_rho_to_tau(RHO_TRUE), n=N_SAMPLES)
# x, y = Synthesis.clayton(tau=gauss_rho_to_tau(RHO_TRUE), n=N_SAMPLES)
x, y = Synthesis.lognormal(rho=RHO_TRUE, n=N_SAMPLES)

x_obs = Synthesis.discretize(x, [0.25, 0.25, 0.25, 0.25])
y_obs = Synthesis.discretize(y, [0.25, 0.25, 0.25, 0.25])

# --- Correlation Estimation ---
corr_fn = make_corr_fn(VariableType.ORDINAL, VariableType.ORDINAL, method=EstimateMethod.RANK)
rho_est = corr_fn(x_obs, y_obs).estimate

# --- Visualization ---
fig, axs = plt.subplots(3, 2, figsize=(16, 18))

# 1. Latent Density (Observed/Discretized)
plot_latent_density(
    x_obs, y_obs, "Latent Space (Discretized)", rho=rho_est,
    ax_theoretical=axs[0, 0], ax_empirical=axs[0, 1]
)

# 2. Latent Density (Original/Continuous)
plot_latent_density(
    x, y, "Latent Space (Continuous)", rho=rho_est,
    ax_theoretical=axs[1, 0], ax_empirical=axs[1, 1]
)

# 3. Tail Concentration and K-Plot
plot_tail_concentration_function(x, y, rho=rho_est, ax=axs[2, 0])
plot_k_plot(x, y, rho=rho_est, mode="standard", ax=axs[2, 1])

# --- Formatting ---
# hspace adds vertical space, wspace adds horizontal space
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()
```
![Visual diagnostics](./diagnostics.svg)

## Literature & References

### Value-Based Correlations
- **Polychoric correlation** Olsson, U. (1979). *Maximum likelihood estimation of the polychoric correlation coefficient*. Psychometrika, 44(4), 443–460.  

- **Polyserial correlation** Olsson, U., Drasgow, F., & Dorans, N. J. (1982). *The polyserial correlation coefficient*. Psychometrika, 47(3), 337–347.  

### Rank-Based & Semiparametric Modeling
- Dey, D., & Zipunnikov, V. (2022). *Semiparametric Gaussian Copula Regression Modeling for Mixed Data Types (SGCRM)*. arXiv preprint arXiv:2205.06868.  

### Copula Theory & Background
- Durante, F., Fernández-Sánchez, J., & Pappadà, R. (2015). *Copulas, diagonals, and tail dependence*. Fuzzy Sets and Systems, 264, 22–41.

- Genest, C., & Favre, A. C. (2007). *Everything you always wanted to know about copula modeling but were afraid to ask*. Journal of Hydrologic Engineering, 12(4), 347–368.



- Hofert, M., Kojadinovic, I., Mächler, M., & Yan, J. (2018). *Elements of copula modeling with R*. Springer.

### Simulation & Data Synthesis
- Hofert, M. (2008). *Sampling Archimedean copulas*. Computational Statistics & Data Analysis, 52(12), 5163–5174.

## Related Packages for Latent Correlations

If you are looking for alternative implementations or specialized features, you may find these packages useful:

### R Packages
- [polycor](https://cran.r-project.org/web/packages/polycor/index.html) – The established standard for Polychoric and Polyserial correlations in R. 
- [latentcor](https://cran.r-project.org/web/packages/latentcor/vignettes/latentcor.html) – Efficient implementations of rank-based correlations.

### Python Packages
- [latentcor](https://pypi.org/project/latentcor/) – High-performance Python implementation of rank-based correlation by the authors of the R package with the same name.
- [semopy](https://pypi.org/project/semopy/) – Structural Equation Modeling (SEM) package; includes polychoric and polyserial correlations.
