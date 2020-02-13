from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
import numpy as np
import pandas as pd


def de_noise_covariance_matrix(covariance_matrix: np.array, n_observations: int, bandwidth: float = .25) -> np.array:
    """
    Computes the correlation matrix associated with a given covariance matrix,
    and derives the eigenvalues and eigenvectors for that correlation matrix.
    Then shrinks the eigenvalues associated with noise, resulting in a de-noised correlation matrix
    which is then used to recover the covariance matrix.

    In summary, this step shrinks only the eigenvalues
    associated with noise, leaving the eigenvalues associated with signal unchanged.

    For more info see section 4.2 of "A Robust Estimator of the Efficient Frontier",
    this function and the functions it calls are all modified from this section

    :param covariance_matrix: the covariance matrix we want to de-noise
    :param n_observations: the number of observations used to create the covariance matrix
    :param bandwidth: bandwidth hyper-parameter for KernelDensity
    :return: de-noised covariance matrix
    """
    #  q=T/N where T=sample length and N=number of variables
    q = n_observations / covariance_matrix.shape[1]

    # get correlation matrix based on covariance matrix
    correlation_matrix = _cov_to_corr(covariance_matrix)

    # Get eigenvalues and eigenvectors in the correlation matrix
    eigenvalues, eigenvectors = _get_PCA(correlation_matrix)

    # Find max random eigenvalue
    max_eigenvalue = _find_max_eigenvalue(np.diag(eigenvalues), q, bandwidth)

    # de-noise the correlation matrix
    n_facts = eigenvalues.shape[0] - np.diag(eigenvalues)[::-1].searchsorted(max_eigenvalue)
    correlation_matrix = _de_noised_corr(eigenvalues, eigenvectors, n_facts)

    # recover covariance matrix from correlation matrix
    de_noised_covariance_matrix = _corr_to_cov(correlation_matrix, np.diag(covariance_matrix) ** .5)
    return de_noised_covariance_matrix


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
    :return: array of eigenvalues and array of eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    indices = eigenvalues.argsort()[::-1]  # arguments for sorting eigenvalues desc
    eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:, indices]
    eigenvalues = np.diagflat(eigenvalues)
    return eigenvalues, eigenvectors


def _find_max_eigenvalue(eigenvalues: np.array, q: float, bandwidth: float) -> float:
    """
    Uses a Kernel Density Estimate (KDE) algorithm to fit the
    Marcenko-Pastur distribution to the empirical distribution of eigenvalues.
    This has the effect of separating noise-related eigenvalues from signal-related eigenvalues.
    :param eigenvalues: array of eigenvalues
    :param q: q=T/N where T=sample length and N=number of variables
    :param bandwidth: bandwidth hyper-parameter for KernelDensity
    :return: max random eigenvalue, variance
    """
    # Find max random eigenvalues by fitting Marcenko's dist to the empirical one
    out = minimize(
        lambda *x: _err_PDFs(*x),
        .5,
        args=(eigenvalues, q, bandwidth),
        bounds=((1E-5, 1 - 1E-5),)
    )
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    max_eigenvalue = var * (1 + (1. / q) ** .5) ** 2
    return max_eigenvalue


def _err_PDFs(var: float, eigenvalues: pd.Series, q: float, bandwidth: float, pts: int = 1000) -> float:
    """
    Calculates a theoretical Marcenko-Pastur probability density function and
    an empirical Marcenko-Pastur probability density function,
    and finds the error between the two by squaring the difference of the two
    :param var: variance 𝜎^2
    :param eigenvalues: array of eigenvalues
    :param q: q=T/N where T=sample length and N=number of variables
    :param bandwidth: bandwidth hyper-parameter for KernelDensity
    :param pts: number of points in the distribution
    :return: the error of the probability distribution functions obtained by squaring the difference
    of the theoretical and empirical Marcenko-Pastur probability density functions
    """
    # Fit error
    theoretical_pdf = _mp_PDF(var, q, pts)  # theoretical probability density function
    empirical_pdf = _fit_KDE(eigenvalues, bandwidth, x=theoretical_pdf.index.values)  # empirical probability density function
    sse = np.sum((empirical_pdf - theoretical_pdf) ** 2)
    return sse


def _mp_PDF(var: float, q: float, pts: int) -> pd.Series:
    """
    Creates a theoretical Marcenko-Pastur probability density function
    :param var: variance 𝜎^2
    :param q: q=T/N where T=sample length and N=number of variables
    :param pts: number of points in the distribution
    :return: a Marcenko-Pastur theoretical probability density function
    """
    min_eigenvalue, max_eigenvalue = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
    eigenvalues = np.linspace(min_eigenvalue, max_eigenvalue, pts).flatten()
    pdf = q / (2 * np.pi * var * eigenvalues) * ((max_eigenvalue - eigenvalues) * (eigenvalues - min_eigenvalue)) ** .5
    pdf = pdf.flatten()
    pdf = pd.Series(pdf, index=eigenvalues)
    return pdf


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


def _de_noised_corr(eigenvalues: np.array, eigenvectors: np.array, n_facts: int) -> np.array:
    """
    Shrinks the eigenvalues associated with noise, and returns a de-noised correlation matrix
    :param eigenvalues: array of eigenvalues
    :param eigenvectors: array of eigenvectors
    :param n_facts: number of elements in diagonalized eigenvalues to replace with the mean of eigenvalues
    :return: de-noised correlation matrix
    """
    # Remove noise from corr by fixing random eigenvalues
    eigenvalues_ = np.diag(eigenvalues).copy()
    eigenvalues_[n_facts:] = eigenvalues_[n_facts:].sum() / float(eigenvalues_.shape[0] - n_facts)
    eigenvalues_ = np.diag(eigenvalues_)
    corr = np.dot(eigenvectors, eigenvalues_).dot(eigenvectors.T)
    corr = _cov_to_corr(corr)
    return corr


def _corr_to_cov(corr: np.array, std: np.array) -> np.array:
    """
    Recovers the covariance matrix from the de-noise correlation matrix
    :param corr: de-noised correlation matrix
    :param std: standard deviation of the correlation matrix
    :return: a recovered covariance matrix
    """
    cov = corr * np.outer(std, std)
    return cov
