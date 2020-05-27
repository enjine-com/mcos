from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
import pandas as pd


def cov_to_corr(cov: np.array) -> np.array:
    """
    Derive the correlation matrix from a covariance matrix
    :param cov: covariance matrix
    :return: correlation matrix
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


def corr_to_cov(corr: np.array, std: np.array) -> np.array:
    """
    Recovers the covariance matrix from the de-noise correlation matrix
    :param corr: de-noised correlation matrix
    :param std: standard deviation of the correlation matrix
    :return: a recovered covariance matrix
    """
    cov = corr * np.outer(std, std)
    return cov


def reorder_matrix(m: np.array, sort_index: np.array) -> np.array:
    m = m[sort_index, :]
    m = m[:, sort_index]
    return m


class AbstractCovarianceTransformer(ABC):
    """
    Abstract class for transforming a covariance matrix
    """

    @abstractmethod
    def transform(self, cov: np.array, n_observations: int) -> np.array:
        """
        Transforms a covariance matrix
        :param cov: covariance matrix
        :param n_observations: number of observations used to create the covariance matrix
        :return: transformed covariance matrix
        """

        pass


class DeNoiserCovarianceTransformer(AbstractCovarianceTransformer):
    def __init__(self, bandwidth: float = .25):
        """
        :param bandwidth: bandwidth hyper-parameter for KernelDensity
        """
        self.bandwidth = bandwidth

    def transform(self, cov: np.array, n_observations: int) -> np.array:
        """
        Computes the correlation matrix associated with a given covariance matrix,
        and derives the eigenvalues and eigenvectors for that correlation matrix.
        Then shrinks the eigenvalues associated with noise, resulting in a de-noised correlation matrix
        which is then used to recover the covariance matrix.

        In summary, this step shrinks only the eigenvalues
        associated with noise, leaving the eigenvalues associated with signal unchanged.

        For more info see section 4.2 of "A Robust Estimator of the Efficient Frontier",
        this function and the functions it calls are all modified from this section

        :param cov: the covariance matrix we want to de-noise
        :param n_observations: the number of observations used to create the covariance matrix
        :return: de-noised covariance matrix
        """
        #  q=T/N where T=sample length and N=number of variables
        q = n_observations / cov.shape[1]

        # get correlation matrix based on covariance matrix
        correlation_matrix = cov_to_corr(cov)

        # Get eigenvalues and eigenvectors in the correlation matrix
        eigenvalues, eigenvectors = self._get_PCA(correlation_matrix)

        # Find max random eigenvalue
        max_eigenvalue = self._find_max_eigenvalue(np.diag(eigenvalues), q)

        # de-noise the correlation matrix
        n_facts = eigenvalues.shape[0] - np.diag(eigenvalues)[::-1].searchsorted(max_eigenvalue)
        correlation_matrix = self._de_noised_corr(eigenvalues, eigenvectors, n_facts)

        # recover covariance matrix from correlation matrix
        de_noised_covariance_matrix = corr_to_cov(correlation_matrix, np.diag(cov) ** .5)
        return de_noised_covariance_matrix

    def _get_PCA(self, matrix: np.array) -> (np.array, np.array):
        """
        Gets eigenvalues and eigenvectors from a Hermitian matrix
        :param matrix: a Hermitian matrix
        :return: array of eigenvalues and array of eigenvectors
        """
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        indices = eigenvalues.argsort()[::-1]  # arguments for sorting eigenvalues desc
        eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:, indices]
        eigenvalues = np.diagflat(eigenvalues)
        return eigenvalues, eigenvectors

    def _find_max_eigenvalue(self, eigenvalues: np.array, q: float) -> float:
        """
        Uses a Kernel Density Estimate (KDE) algorithm to fit the
        Marcenko-Pastur distribution to the empirical distribution of eigenvalues.
        This has the effect of separating noise-related eigenvalues from signal-related eigenvalues.
        :param eigenvalues: array of eigenvalues
        :param q: q=T/N where T=sample length and N=number of variables
        :return: max random eigenvalue, variance
        """
        # Find max random eigenvalues by fitting Marcenko's dist to the empirical one
        out = minimize(
            lambda *x: self._err_PDFs(*x),
            .5,
            args=(eigenvalues, q),
            bounds=((1E-5, 1 - 1E-5),)
        )
        if out['success']:
            var = out['x'][0]
        else:
            var = 1
        max_eigenvalue = var * (1 + (1. / q) ** .5) ** 2
        return max_eigenvalue

    def _err_PDFs(self, var: float, eigenvalues: pd.Series, q: float, pts: int = 1000) -> float:
        """
        Calculates a theoretical Marcenko-Pastur probability density function and
        an empirical Marcenko-Pastur probability density function,
        and finds the error between the two by squaring the difference of the two
        :param var: variance 𝜎^2
        :param eigenvalues: array of eigenvalues
        :param q: q=T/N where T=sample length and N=number of variables
        :param pts: number of points in the distribution
        :return: the error of the probability distribution functions obtained by squaring the difference
        of the theoretical and empirical Marcenko-Pastur probability density functions
        """
        # Fit error
        theoretical_pdf = self._mp_PDF(var, q, pts)  # theoretical probability density function
        empirical_pdf = self._fit_KDE(eigenvalues,
                                      x=theoretical_pdf.index.values)  # empirical probability density function
        sse = np.sum((empirical_pdf - theoretical_pdf) ** 2)
        return sse

    def _mp_PDF(self, var: float, q: float, pts: int) -> pd.Series:
        """
        Creates a theoretical Marcenko-Pastur probability density function
        :param var: variance 𝜎^2
        :param q: q=T/N where T=sample length and N=number of variables
        :param pts: number of points in the distribution
        :return: a theoretical Marcenko-Pastur probability density function
        """
        min_eigenvalue, max_eigenvalue = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
        eigenvalues = np.linspace(min_eigenvalue, max_eigenvalue, pts).flatten()
        pdf = q / (2 * np.pi * var * eigenvalues) * \
              ((max_eigenvalue - eigenvalues) * (eigenvalues - min_eigenvalue)) ** .5
        pdf = pdf.flatten()
        pdf = pd.Series(pdf, index=eigenvalues)
        return pdf

    def _fit_KDE(
            self,
            obs: np.array,
            kernel: str = 'gaussian',
            x: np.array = None
    ) -> pd.Series:
        """
        Fit kernel to a series of observations, and derive the prob of observations.
        x is the array of values on which the fit KDE will be evaluated
        :param obs: the series of observations
        :param kernel: kernel hyper-parameter for KernelDensity
        :param x: array of values _fit_KDE will be evaluated against
        :return: an empirical Marcenko-Pastur probability density function
        """
        if len(obs.shape) == 1:
            obs = obs.reshape(-1, 1)
        kde = KernelDensity(kernel=kernel, bandwidth=self.bandwidth).fit(obs)
        if x is None:
            x = np.unique(obs).reshape(-1, 1)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        log_prob = kde.score_samples(x)  # log(density)
        pdf = pd.Series(np.exp(log_prob), index=x.flatten())
        return pdf

    def _de_noised_corr(self, eigenvalues: np.array, eigenvectors: np.array, n_facts: int) -> np.array:
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
        corr = cov_to_corr(corr)
        return corr


class DetoneCovarianceTransformer(AbstractCovarianceTransformer):
    def __init__(self, n_remove: int):
        """
        Removes the largest eigenvalue/eigenvector pairs from the covariance matrix. Since the largest eigenvalues are
        typically associated with the market component, removing such eigenvalues has the effect of removing the
        market's influence on the correlations between securities. See chapter 2.6 of "Machine Learning for Asset
        Managers".
        :param n_remove: The number of the largest eigenvalues to remove
        """
        self.n_remove = n_remove

    def transform(self, cov: np.array, n_observations: int) -> np.array:
        if self.n_remove == 0:
            return cov

        corr = cov_to_corr(cov)

        w, v = linalg.eig(corr)

        # sort from highest eigenvalues to lowest
        sort_index = np.argsort(-np.abs(w))  # get sort_index in descending absolute order - i.e. from most significant
        w = w[sort_index]
        v = v[:, sort_index]

        # remove largest eigenvalue component
        v_market = v[:, 0:self.n_remove]  # largest eigenvectors
        w_market = w[0:self.n_remove]

        market_comp = np.matmul(
            np.matmul(v_market, w_market).reshape((v.shape[0], self.n_remove,)),
            np.transpose(v_market)
        )

        c2 = corr - market_comp

        # normalize the correlation matrix so the diagonals are 1
        norm_matrix = np.diag(c2.diagonal() ** -0.5)
        c2 = np.matmul(np.matmul(norm_matrix, c2), np.transpose(norm_matrix))

        return corr_to_cov(c2, np.diag(cov) ** .5)
