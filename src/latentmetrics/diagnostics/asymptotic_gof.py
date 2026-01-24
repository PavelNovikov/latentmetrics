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

    num = (
        3 * q1**2 * rho**2
        + q1**2
        - 2 * q1 * q2 * rho**3
        - 6 * q1 * q2 * rho
        + 3 * q2**2 * rho**2
        + q2**2
        + rho**4
        - 1
    )
    denom = (rho**2 - 1) ** 3
    return num / denom


def calculate_grad_hessian(q1, q2, rho):

    num = (
        -12 * q1**2 * rho**3
        - 12 * q1**2 * rho
        + 6 * q1 * q2 * rho**4
        + 36 * q1 * q2 * rho**2
        + 6 * q1 * q2
        - 12 * q2**2 * rho**3
        - 12 * q2**2 * rho
        - 2 * rho**5
        - 4 * rho**3
        + 6 * rho
    )
    denom = (rho**2 - 1) ** 4
    return num / denom


def calculate_grad_D(q1, q2, rho):

    S = calculate_score(q1, q2, rho)
    H = calculate_hessian(q1, q2, rho)
    H_prime = calculate_grad_hessian(q1, q2, rho)

    return H_prime + 2 * S * H


def calculate_W(data, rho, margin=2, grid_res=400):

    data = np.atleast_1d(data)
    n = len(data)

    h = 1.0 / grid_res
    ticks = np.linspace(h / 2, 1 - h / 2, grid_res)
    u_grid, v_grid = np.meshgrid(ticks, ticks)
    q1 = norm.ppf(u_grid)
    q2 = norm.ppf(v_grid)

    R2 = 1 - rho**2

    density = (1 / np.sqrt(R2)) * np.exp(
        -(rho**2 * (q1**2 + q2**2) - 2 * rho * q1 * q2) / (2 * R2)
    )

    hessian = calculate_hessian(q1, q2, rho)

    weight = hessian * density * (h**2)
    weight -= np.mean(weight)

    W = np.zeros(n)
    for t in range(n):
        indic = (data[t] <= v_grid).astype(float) - v_grid
        W[t] = np.sum(indic * weight)

    return W


def calculate_M(F_data, rho, margin=2, grid_limit=5, grid_res=400):

    F_data = np.atleast_1d(F_data)
    nodes = np.linspace(-grid_limit, grid_limit, grid_res)
    dq = nodes[1] - nodes[0]
    q1_g, q2_g = np.meshgrid(nodes, nodes)

    R2 = 1 - rho**2
    denom = (rho**2 - 1) ** 4

    if margin == 2:
        cond_density = (1 / np.sqrt(2 * np.pi * R2)) * np.exp(
            -((q1_g - rho * q2_g) ** 2) / (2 * R2)
        )
        target_grid = norm.cdf(q2_g)

        km = (
            -2 * q1_g**3 * rho**3
            - 2 * q1_g**3 * rho
            + 2 * q1_g**2 * q2_g * rho**4
            + 8 * q1_g**2 * q2_g * rho**2
            + 2 * q1_g**2 * q2_g
            - 6 * q1_g * q2_g**2 * rho**3
            - 6 * q1_g * q2_g**2 * rho
            - 4 * q1_g * rho**5
            - 4 * q1_g * rho**3
            + 8 * q1_g * rho
            + 4 * q2_g**3 * rho**2
            + 10 * q2_g * rho**4
            - 8 * q2_g * rho**2
            - 2 * q2_g
        )
    else:
        cond_density = (1 / np.sqrt(2 * np.pi * R2)) * np.exp(
            -((q2_g - rho * q1_g) ** 2) / (2 * R2)
        )
        target_grid = norm.cdf(q1_g)

        km = (
            -2 * q2_g**3 * rho**3
            - 2 * q2_g**3 * rho
            + 2 * q2_g**2 * q1_g * rho**4
            + 8 * q2_g**2 * q1_g * rho**2
            + 2 * q2_g**2 * q1_g
            - 6 * q2_g * q1_g**2 * rho**3
            - 6 * q2_g * q1_g**2 * rho
            - 4 * q2_g * rho**5
            - 4 * q2_g * rho**3
            + 8 * q2_g * rho
            + 4 * q1_g**3 * rho**2
            + 10 * q1_g * rho**4
            - 8 * q1_g * rho**2
            - 2 * q1_g
        )

    weight_q = (km / denom) * cond_density * (dq**2)
    weight_q -= np.mean(weight_q)

    M = np.zeros(len(F_data))
    for t in range(len(F_data)):
        indic = (F_data[t] <= target_grid).astype(float) - target_grid
        M[t] = np.sum(indic * weight_q)

    return M


def calculate_psi(u, v, rho):
    q1, q2 = norm.ppf(u), norm.ppf(v)

    score_t = calculate_score(q1, q2, rho)
    hessian_t = calculate_hessian(q1, q2, rho)

    M1 = calculate_M(u, rho, margin=1)
    M2 = calculate_M(v, rho, margin=2)
    term_a = (hessian_t + score_t**2) + M1 + M2

    B = -np.mean(hessian_t)

    grad_D_vec = calculate_grad_D(q1, q2, rho)
    adj_factor = np.mean(grad_D_vec) / B

    W1 = calculate_W(u, rho, margin=1)
    W2 = calculate_W(v, rho, margin=2)
    term_b = adj_factor * (score_t + W1 + W2)

    return term_a + term_b


def calculate_correction_factor(u, v, rho):
    psi = calculate_psi(u, v, rho)
    return np.mean(psi**2)


def huang_prokhorov_test(x, y):
    """
    Huang-Prokhorov goodness-of-fit test for Gaussian copula.

    Reference
    ---------
    Huang, W., & Prokhorov, A. (2014).
    A goodness-of-fit test for copulas.
    Econometric Reviews, 33(7), 751-771.
    """
    x = np.asarray(x)
    y = np.asarray(y)
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
