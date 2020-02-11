from sklearn.neighbors.kde import KernelDensity
from scipy.optimize import minimize
import numpy as np
import pandas as pd


def fit_KDE(obs: np.array, b_width: float = .25, kernel: str = 'gaussian', x: np.array = None) -> pd.Series:
    """

    :param obs:
    :param b_width:
    :param kernel:
    :param x:
    :return:
    """
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=b_width).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    log_prob = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())
    return pdf


def mp_PDF(var: float, q: float, pts: int) -> pd.Series:
    """

    :param var:
    :param q:
    :param pts:
    :return:
    """
    # Marcenko-Pastur pdf
    # q=T/N
    e_min, e_max = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
    e_val = np.linspace(e_min, e_max, pts)
    pdf = q / (2 * np.pi * var * e_val) * ((e_max - e_val) * (e_val - e_min)) ** .5
    pdf = pd.Series(pdf, index=e_val)
    return pdf


def err_PDFs(var: float, e_val: pd.Series, q: float, b_width: float, pts: int = 1000) -> float:
    """

    :param var:
    :param e_val:
    :param q:
    :param b_width:
    :param pts:
    :return:
    """
    # Fit error
    pdf0 = mp_PDF(var, q, pts)  # theoretical pdf
    pdf1 = fit_KDE(e_val, b_width, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse


def find_max_eval(e_val: np.array, q: float, b_width: float) -> (float, float):
    """

    :param e_val:
    :param q:
    :param b_width:
    :return:
    """
    # Find max random e_val by fitting Marcenko's dist to the empirical one
    out = minimize(
        lambda *x: err_PDFs(*x),
        .5,
        args=(e_val, q, b_width),
        bounds=((1E-5, 1 - 1E-5),)
    )
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    e_max = var * (1 + (1. / q) ** .5) ** 2
    return e_max, var


def corr_to_cov(corr: np.array, std: np.array) -> np.array:
    """

    :param corr:
    :param std:
    :return:
    """
    cov = corr * np.outer(std, std)
    return cov


def cov_to_corr(cov: np.array) -> np.array:
    """

    :param cov:
    :return:
    """
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


def get_PCA(matrix: np.array) -> (np.array, np.array):
    """

    :param matrix:
    :return:
    """
    # Get e_val,e_vec from a Hermitian matrix
    e_val, e_vec = np.linalg.eigh(matrix)
    indices = e_val.argsort()[::-1]  # arguments for sorting e_val desc
    e_val, e_vec = e_val[indices], e_vec[:, indices]
    e_val = np.diagflat(e_val)
    return e_val, e_vec


def denoised_corr(e_val: np.array, e_vec: np.array, n_facts: int) -> np.array:
    """

    :param e_val:
    :param e_vec:
    :param n_facts:
    :return:
    """
    # Remove noise from corr by fixing random eigenvalues
    e_val_ = np.diag(e_val).copy()
    e_val_[n_facts:] = e_val_[n_facts:].sum() / float(e_val_.shape[0] - n_facts)
    e_val_ = np.diag(e_val_)
    corr1 = np.dot(e_vec, e_val_).dot(e_vec.T)
    corr1 = cov_to_corr(corr1)
    return corr1


def de_noise_cov(cov0: np.array, q: float, b_width: float) -> np.array:
    """

    :param cov0:
    :param q:
    :param b_width:
    :return:
    """
    corr0 = cov_to_corr(cov0)
    e_val_0, e_vec_0 = get_PCA(corr0)
    e_max_0, var0 = find_max_eval(np.diag(e_val_0), q, b_width)
    n_facts_0 = e_val_0.shape[0] - np.diag(e_val_0)[::-1].searchsorted(e_max_0)
    corr1 = denoised_corr(e_val_0, e_vec_0, n_facts_0)
    cov1 = corr_to_cov(corr1, np.diag(cov0) ** .5)
    return cov1
