import numpy as np


def gauss_rho_to_tau(rho):
    """
    Convert Pearson's rho to Kendall's tau under the bivariate normal assumption.
    """
    return (2 / np.pi) * np.arcsin(rho)


def gauss_tau_to_rho(tau):
    """
    Convert Kendall's tau to Pearson's rho under the bivariate normal assumption.
    """
    return np.sin((np.pi / 2) * tau)
