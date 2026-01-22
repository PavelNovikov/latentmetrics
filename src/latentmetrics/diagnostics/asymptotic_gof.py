"""
Tests for Assessing Agreement with a Gaussian Copula Model Using the Asymptotic Distribution of the Test Statistic
"""

import numpy as np
from scipy import stats
from scipy.stats import norm, chi2, multivariate_normal
from latentmetrics.types import VariableType as vt
from latentmetrics._utils import swap_args
from latentmetrics.engines.value_based import polychoric_correlation


def get_gof_test_fn(x_type, y_type):
    xt, yt = x_type, y_type
    if x_type == vt.BINARY:
        xt = vt.ORDINAL
    if y_type == vt.BINARY:
        yt = vt.ORDINAL
    if (xt, yt) == (vt.CONTINUOUS, vt.CONTINUOUS):
        return huang_prokhorov_test
    elif (xt, yt) == (vt.CONTINUOUS, vt.ORDINAL):
        return m2_mixed
    elif (xt, yt) == (vt.ORDINAL, vt.CONTINUOUS):
        return swap_args(m2_mixed)
    elif (xt, yt) == (vt.ORDINAL, vt.ORDINAL):
        return m2_test


def m2_mixed(x_cont, y_ord, bins=10):
    """
    Assesses underlying normality for mixed continuous/ordinal data by binning
    the continuous variable.

    """
    x_cont = np.asarray(x_cont)
    y_ord = np.asarray(y_ord)

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(x_cont, quantiles))

    x_binned = np.digitize(x_cont, bin_edges[1:-1])

    return m2_test(x_binned, y_ord)


def m2_test(x, y):
    """
    Assesses underlying normality for bivariate ordinal data.

    Reference
    ---------
    Maydeu-Olivares, A., García-Forero, C., Gallardo-Pujol, D., & Renom, J. (2009).
    Testing categorized bivariate normality with two-stage polychoric correlation estimates.
    Methodology, 5(4), 131-136.
    """
    N = len(x)
    categories_x = np.sort(np.unique(x))
    categories_y = np.sort(np.unique(y))
    I, J = len(categories_x), len(categories_y)

    observed_counts = np.zeros((I, J))
    for i, val_x in enumerate(categories_x):
        for j, val_y in enumerate(categories_y):
            observed_counts[i, j] = np.sum((x == val_x) & (y == val_y))
    p = (observed_counts / N).flatten()

    def get_thresholds(data):
        props = np.cumsum(np.bincount(data) / len(data))[:-1]
        return norm.ppf(props)

    alpha = get_thresholds(x.astype(int))
    beta = get_thresholds(y.astype(int))

    # The test assumes that to-step polychoric estimator is used.
    rho_hat = polychoric_correlation(x, y)

    kappa = np.concatenate([alpha, beta, [rho_hat]])
    num_params = len(kappa)

    def get_probs(k):
        th_x = np.concatenate(([-np.inf], k[: I - 1], [np.inf]))
        th_y = np.concatenate(([-np.inf], k[I - 1 : -1], [np.inf]))
        rho = k[-1]
        cov = [[1.0, rho], [rho, 1.0]]
        dist = multivariate_normal(mean=[0, 0], cov=cov, allow_singular=True)

        pi_vec = []
        for i in range(I):
            for j in range(J):
                val = (
                    dist.cdf([th_x[i + 1], th_y[j + 1]])
                    - dist.cdf([th_x[i], th_y[j + 1]])
                    - dist.cdf([th_x[i + 1], th_y[j]])
                    + dist.cdf([th_x[i], th_y[j]])
                )
                pi_vec.append(max(val, 1e-10))
        return np.array(pi_vec)

    pi_hat = get_probs(kappa)

    eps = 1e-5
    Delta = np.zeros((I * J, num_params))
    for k in range(num_params):
        k_plus = kappa.copy()
        k_plus[k] += eps
        Delta[:, k] = (get_probs(k_plus) - pi_hat) / eps

    D_inv = np.diag(1.0 / pi_hat)
    # The weight matrix W = D^-1 - D^-1 * Delta * (Delta' * D^-1 * Delta)^-1 * Delta' * D^-1
    Dt_Dinv = Delta.T @ D_inv
    middle_inv = np.linalg.inv(Dt_Dinv @ Delta)
    W = D_inv - (D_inv @ Delta @ middle_inv @ Dt_Dinv)

    diff = p - pi_hat
    mn_stat = N * (diff.T @ W @ diff)

    # df = (number of cells) - (number of parameters) - 1
    df = (I * J) - num_params - 1
    p_value = 1 - chi2.cdf(mn_stat, df)
    return p_value


def calculate_score(q1, q2, rho):
    return (
        -1
        * (q1**2 * rho - q1 * q2 * rho**2 - q1 * q2 + q2**2 * rho + rho**3 - rho)
        / ((rho - 1) ** 2 * (rho + 1) ** 2)
    )


def calculate_hessian(q1, q2, rho):
    return (
        3 * q1**2 * rho**2
        + q1**2
        - 2 * q1 * q2 * rho**3
        - 6 * q1 * q2 * rho
        + 3 * q2**2 * rho**2
        + q2**2
        + rho**4
        - 1
    ) / ((rho - 1) ** 3 * (rho + 1) ** 3)


def calculate_W(F_data, rho, grid_res=100):
    n = len(F_data)

    ticks = np.linspace(0.001, 0.999, grid_res)
    u_grid, v_grid = np.meshgrid(ticks, ticks)

    q1 = norm.ppf(u_grid)
    q2 = norm.ppf(v_grid)
    denom = (rho**2 - 1) ** 2

    nabla_k1 = (
        np.sqrt(2 * np.pi)
        * np.exp(0.5 * q1**2)
        * (rho**2 * q2 - 2 * rho * q1 + q2)
        / denom
    )

    density = (1 / np.sqrt(1 - rho**2)) * np.exp(
        -(rho**2 * (q1**2 + q2**2) - 2 * rho * q1 * q2) / (2 * (1 - rho**2))
    )

    kernel_weight = nabla_k1 * density * (1.0 / grid_res**2)

    W = np.zeros(n)

    for t in range(n):
        indicator_part = (F_data[t] <= u_grid).astype(float) - u_grid
        W[t] = np.sum(indicator_part * kernel_weight)

    return W


def calculate_grad_D(q1, q2, rho):
    return -(
        6 * q1**4 * rho**3
        + 2 * q1**4 * rho
        - 10 * q1**3 * q2 * rho**4
        - 20 * q1**3 * q2 * rho**2
        - 2 * q1**3 * q2
        + 4 * q1**2 * q2**2 * rho**5
        + 28 * q1**2 * q2**2 * rho**3
        + 16 * q1**2 * q2**2 * rho
        + 20 * q1**2 * rho**5
        - 4 * q1**2 * rho**3
        - 16 * q1**2 * rho
        - 10 * q1 * q2**3 * rho**4
        - 20 * q1 * q2**3 * rho**2
        - 2 * q1 * q2**3
        - 12 * q1 * q2 * rho**6
        - 40 * q1 * q2 * rho**4
        + 44 * q1 * q2 * rho**2
        + 8 * q1 * q2
        + 6 * q2**4 * rho**3
        + 2 * q2**4 * rho
        + 20 * q2**2 * rho**5
        - 4 * q2**2 * rho**3
        - 16 * q2**2 * rho
        + 4 * rho**7
        - 12 * rho**3
        + 8 * rho
    ) / ((1.0 * rho - 1.0) ** 5 * (1.0 * rho + 1.0) ** 5)


def calculate_M(F_data, rho, grid_res=100):
    n = len(F_data)

    ticks = np.linspace(0.001, 0.999, grid_res)
    u_grid, v_grid = np.meshgrid(ticks, ticks)

    q1 = norm.ppf(u_grid)
    q2 = norm.ppf(v_grid)

    density = (1 / np.sqrt(1 - rho**2)) * np.exp(
        -(rho**2 * (q1**2 + q2**2) - 2 * rho * q1 * q2) / (2 * (1 - rho**2))
    )

    denom_m = (rho - 1) ** 4 * (rho + 1) ** 4

    km = (
        4 * q1**3 * rho**2
        - 6 * q1**2 * q2 * rho**3
        - 6 * q1**2 * q2 * rho
        + 2 * q1 * q2**2 * rho**4
        + 8 * q1 * q2**2 * rho**2
        + 2 * q1 * q2**2
        + q1 * rho**4
        - 8 * q1 * rho**2
        - 2 * q1
        - 2 * q2**3 * rho**3
        - 2 * q2**3 * rho
        - 4 * q2 * rho**5
        - 4 * q2 * rho**3
        + 8 * q2 * rho
    )

    m_weight = (
        (np.sqrt(2 * np.pi) * km * np.exp(0.5 * q1**2) / denom_m)
        * density
        / (grid_res**2)
    )

    M = np.zeros(n)

    for t in range(n):
        indic = (F_data[t] <= u_grid).astype(float) - u_grid

        M[t] = np.sum(indic * m_weight)

    return M


def calculate_correction_factor(u, v, rho):
    q1 = norm.ppf(u)
    q2 = norm.ppf(v)

    score_t = calculate_score(q1, q2, rho)
    hessian_t = calculate_hessian(q1, q2, rho)
    dt_vec = hessian_t + score_t**2

    W1 = calculate_W(u, rho)
    W2 = calculate_W(v, rho)
    M1 = calculate_M(u, rho)
    M2 = calculate_M(v, rho)

    B = np.mean(hessian_t)

    grad_D = np.mean(calculate_grad_D(q1, q2, rho))

    adj_factor = grad_D / B

    term_a = dt_vec + M1 + M2
    term_b = adj_factor * (score_t + W1 + W2)

    psi = term_a + term_b

    v_rho = np.mean(psi**2)

    return v_rho


def huang_prokhorov_test(x, y):
    """
    Reference
    ---------
    Huang, W., & Prokhorov, A. (2014).
    A goodness-of-fit test for copulas.
    Econometric Reviews, 33(7), 751-771.
    """
    T = len(x)
    u = stats.rankdata(x) / (T + 1)
    v = stats.rankdata(y) / (T + 1)
    q1, q2 = norm.ppf(u), norm.ppf(v)
    rho = np.corrcoef(q1, q2)[0, 1]
    s = calculate_score(q1, q2, rho)
    h = calculate_hessian(q1, q2, rho)
    dt = h + s**2
    D_bar = np.mean(dt)
    Vp = calculate_correction_factor(u, v, rho)
    stat = T * (D_bar**2) / Vp
    p_val = 1 - chi2.cdf(stat, df=1)
    return p_val
