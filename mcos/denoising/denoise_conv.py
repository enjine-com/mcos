# Code modified from section 4.2 of the "A Robust Estimator of the Efficient Frontier" paper
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
import numpy as np
import pandas as pd


def _fit_KDE(obs: np.array, bandwidth: float = .25, kernel: str = 'gaussian', x: np.array = None) -> pd.Series:
    """
    Fit kernel to a series of observations, and derive the prob of observations.
    x is the array of values on which the fit KDE will be evaluated
    :param obs: the series of observations
    :param bandwidth: bandwidth hyper-parameter for KernelDensity
    :param kernel: kernel hyper-parameter for KernelDensity
    :param x: array of values _fit_KDE will be evaluated against
    :return: a Marcenko-Pastur empirical probability density function
    """
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    log_prob = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())
    return pdf


def _mp_PDF(var: float, q: float, pts: int) -> pd.Series:
    """
    Creates a theoretical Marcenko-Pastur probability density function
    :param var: variance 𝜎^2
    :param q: q=T/N where T=sample length and N=number of variables
    :param pts: number of points in the distribution
    :return: a Marcenko-Pastur theoretical probability density function
    """
    e_min, e_max = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
    e_val = np.linspace(e_min, e_max, pts).flatten()
    pdf = q / (2 * np.pi * var * e_val) * ((e_max - e_val) * (e_val - e_min)) ** .5
    pdf = pdf.flatten()
    pdf = pd.Series(pdf, index=e_val)
    return pdf


def _err_PDFs(var: float, e_val: pd.Series, q: float, bandwidth: float, pts: int = 1000) -> float:
    """
    Calculates a theoretical Marcenko-Pastur probability density function and
    an empirical Marcenko-Pastur probability density function,
    and finds the error between the two by squaring the difference of the two
    :param var: variance 𝜎^2
    :param e_val: array of eigenvalues
    :param q: q=T/N where T=sample length and N=number of variables
    :param bandwidth: bandwidth hyper-parameter for KernelDensity
    :param pts: number of points in the distribution
    :return: the error of the probability distribution functions obtained by squaring the difference
    of the theoretical and empirical Marcenko-Pastur probability density functions
    """
    # Fit error
    pdf0 = _mp_PDF(var, q, pts)  # theoretical probability density function
    pdf1 = _fit_KDE(e_val, bandwidth, x=pdf0.index.values)  # empirical probability density function
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse


def _find_max_eval(e_val: np.array, q: float, bandwidth: float) -> (float, float):
    """
    Uses a Kernel Density Estimate (KDE) algorithm to fit the
    Marcenko-Pastur distribution to the empirical distribution of eigenvalues.
    This has the effect of separating noise-related eigenvalues from signal-related eigenvalues.
    :param e_val: array of eigenvalues
    :param q: q=T/N where T=sample length and N=number of variables
    :param bandwidth: bandwidth hyper-parameter for KernelDensity
    :return: max random eigenvalue, variance
    """
    # Find max random e_val by fitting Marcenko's dist to the empirical one
    out = minimize(
        lambda *x: _err_PDFs(*x),
        .5,
        args=(e_val, q, bandwidth),
        bounds=((1E-5, 1 - 1E-5),)
    )
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    e_max = var * (1 + (1. / q) ** .5) ** 2
    return e_max, var


def _corr_to_cov(corr: np.array, std: np.array) -> np.array:
    """
    Recovers the covariance matrix from the de-noise correlation matrix
    :param corr: de-noised correlation matrix
    :param std: standard deviation of the correlation matrix
    :return: a recovered covariance matrix
    """
    cov = corr * np.outer(std, std)
    return cov


def _cov_to_corr(cov: np.array) -> np.array:
    """
    Derive the correlation matrix from a covariance matrix
    :param cov: covariance matrix
    :return: correlation matrix
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


def _get_PCA(matrix: np.array) -> (np.array, np.array):
    """
    Gets eigenvalues and eigenvectors from a Hermitian matrix
    :param matrix: correlation matrix
    :return: array of eigenvalues and eigenvectors
    """
    e_val, e_vec = np.linalg.eigh(matrix)
    indices = e_val.argsort()[::-1]  # arguments for sorting e_val desc
    e_val, e_vec = e_val[indices], e_vec[:, indices]
    e_val = np.diagflat(e_val)
    return e_val, e_vec


def _denoised_corr(e_val: np.array, e_vec: np.array, n_facts: int) -> np.array:
    """
    Shrinks the eigenvalues associated with noise, and returns a de-noised correlation matrix
    :param e_val: array of eigenvalues
    :param e_vec: array of eigenvectors
    :param n_facts: number of elements in e_val that is then diagonalized to replace with the mean of e_val
    :return: de-noised correlation matrix
    """
    # Remove noise from corr by fixing random eigenvalues
    e_val_ = np.diag(e_val).copy()
    e_val_[n_facts:] = e_val_[n_facts:].sum() / float(e_val_.shape[0] - n_facts)
    e_val_ = np.diag(e_val_)
    corr1 = np.dot(e_vec, e_val_).dot(e_vec.T)
    corr1 = _cov_to_corr(corr1)
    return corr1


def de_noise_cov(cov0: np.array, q: float, bandwidth: float) -> np.array:
    """
    Computes the correlation matrix associated with a given covariance matrix,
    and derives the eigenvalues and eigenvectors for that correlation matrix.
    Then shrinks the eigenvalues associated with noise, resulting in a de-noised correlation matrix
    which is then used to recover the covariance matrix. In summary, this step shrinks only the eigenvalues
    associated with noise, leaving the eigenvalues associated with signal unchanged.
    :param cov0: the covariance matrix we want to de-noise
    :param q: q=T/N where T=sample length and N=number of variables
    :param bandwidth: bandwidth hyper-parameter for KernelDensity
    :return: de-noised covariance matrix
    """
    corr0 = _cov_to_corr(cov0)
    e_val_0, e_vec_0 = _get_PCA(corr0)
    e_max_0, var0 = _find_max_eval(np.diag(e_val_0), q, bandwidth)
    n_facts_0 = e_val_0.shape[0] - np.diag(e_val_0)[::-1].searchsorted(e_max_0)
    corr1 = _denoised_corr(e_val_0, e_vec_0, n_facts_0)
    cov1 = _corr_to_cov(corr1, np.diag(cov0) ** .5)
    return cov1
