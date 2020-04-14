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
        samples = pd.DataFrame()
        for q in range(self.jackknife_samples):
            x = np.random.multivariate_normal(self.mu.flatten(), self.cov, size=self.n_observations)
            x_prime = pd.DataFrame()
            mu_prime = []
            for count in range(np.size(x, 1)):
                y = x[:,count]
                y_prime = []
                for z in y:
                    y_prime.append((sum(y) - z) / (self.n_observations - 1))
                x_prime = x_prime.append([y_prime])
                mu_prime.append(np.asarray(y_prime).mean())
            samples = samples.append([mu_prime])

        return np.asarray(mu_prime).reshape(-1, 1), sample_cov(x_prime)

