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

These estimators assume a **bivariate normal underlying distribution** and estimate the correlation parameter (`rho`) by **maximizing the likelihood** of the observed discretized data.

### 2. Rank-Based Correlations
- Suitable for **binary, ordinal, and mixed data**, and also supports **continuous data** through **Greiner's formula**
- Based on the assumption that data arises from an **arbitrary monotonic transformation** of underlying bivariate Gaussian variables (Gaussian copula model)
- The correlation parameter (`rho`) is estimated by **matching the observed Kendall's tau** (rank correlation, invariant to monotonic transformations) to its expected value as a function of `rho`
- Can also handle continuous data (via **Greiner's formula**)

## Usage

```python
import numpy as np
from latentmetrics import make_corr_fn, VariableType, EstimateMethod

# Example latent data
x_latent = np.random.normal(size=100)
y_latent = 0.5 * x_latent + np.sqrt(1 - 0.5**2) * np.random.normal(size=100)

# Discretize to ordinal/binary
x_obs = np.digitize(x_latent, np.quantile(x_latent, [0.25, 0.5, 0.75]))
y_obs = np.digitize(y_latent, np.quantile(y_latent, [0.5]))

# Create correlation function and compute
corr_fn = make_corr_fn(VariableType.ORDINAL, VariableType.BINARY, method=EstimateMethod.VALUE)
result = corr_fn(x_obs, y_obs)

print("Estimated correlation:", result.estimate)
```

## Literature & References

### Value-Based Correlations
- **Polychoric correlation**  
  Olsson, U. (1979). *Maximum likelihood estimation of the polychoric correlation coefficient*. Psychometrika, 44(4), 443–460.  

- **Polyserial correlation**  
  Olsson, U., Drasgow, F., & Dorans, N. J. (1982). *The polyserial correlation coefficient*. Psychometrika, 47(3), 337–347.  

### Rank-Based Correlations
- Dey, D., & Zipunnikov, V. (2022). "Semiparametric Gaussian Copula Regression Modeling for Mixed Data Types (SGCRM)." arXiv preprint arXiv:2205.06868.  

### Copula Background
- Hofert, M., Kojadinovic, I., Mächler, M., & Yan, J. (2018). *Elements of copula modeling with R*. Springer.

## Available Packages

### R Packages
- [polycor](https://cran.r-project.org/web/packages/polycor/index.html) – Polychoric and Polyserial correlations  
- [latentcor](https://cran.r-project.org/web/packages/latentcor/vignettes/latentcor.html) – Efficient implementations of rank-based correlations

### Python Packages
- [latentcor](https://pypi.org/project/latentcor/) – Efficient rank-based correlations from the authors of the R package  
- [semopy](https://pypi.org/project/semopy/) – Structural Equation Modeling (SEM) package; includes polychoric and polyserial correlations
