import numpy as np
from abc import abstractmethod, ABC
from sklearn.covariance import LedoitWolf
from pypfopt.risk_models import sample_cov
import pandas as pd


class AbstractObservationSimulator(ABC):

    @abstractmethod
    def simulate(self) -> (np.array, np.array):
        """
        Draws empirical means and covariances. See section 4.1 of the "A Robust Estimator of the Efficient Frontier"
        paper.
        :param n_observations:
        @return: Tuple of expected return vector and covariance matrix
        """
        pass


class MuCovLedoitWolfObservationSimulator(AbstractObservationSimulator):

    def __init__(self, mu: np.array, cov: np.array, n_observations: int):
        self.mu = mu
        self.cov = cov
        self.n_observations = n_observations

    def simulate(self) -> (np.array, np.array):
        x = np.random.multivariate_normal(self.mu.flatten(), self.cov, size=self.n_observations)
        return x.mean(axis=0).reshape(-1, 1), LedoitWolf().fit(x).covariance_


class MuCovObservationSimulator(AbstractObservationSimulator):

    def __init__(self, mu: np.array, cov: np.array, n_observations: int):
        self.mu = mu
        self.cov = cov
        self.n_observations = n_observations

    def simulate(self) -> (np.array, np.array):
        x = np.random.multivariate_normal(self.mu.flatten(), self.cov, size=self.n_observations)
        return x.mean(axis=0).reshape(-1, 1), np.cov(x, rowvar=False)


class MuCovJackknifeObservationSimulator(AbstractObservationSimulator):

    def __init__(self, mu: np.array, cov: np.array, n_observations: int, jackknife_samples):
        self.mu = mu
        self.cov = cov
        self.n_observations = n_observations
        self.jackknife_samples = jackknife_samples

    def simulate(self) -> (np.array, np.array):

        x = {}
        for q in range(self.jackknife_samples):
            x[q] = np.random.multivariate_normal(self.mu.flatten(), self.cov, size=self.n_observations)

        x_prime = None
        for count in range(x.__len__()):
            x_total = None
            for y in range(x.__len__()):
                if count != y:
                    if x_total is None:
                        x_total = x[y]
                    else:
                        x_total = x_total + x[y]
            x_total = x_total / (x.__len__() - 1)
            if x_prime is None:
                x_prime = x_total
            else:
                x_prime = x_prime + x_total
        x_prime = x_prime / x.__len__()

        return x_prime.mean(axis=0).reshape(-1, 1), np.cov(x_prime, rowvar=False)

