from sklearn.covariance import LedoitWolf

from mcos.observation_simulator.base import ObservationSimulator
import numpy as np


class MuCovLedoitWolfObservationSimulator(ObservationSimulator):
    def __init__(self, mu: np.array, cov: np.array, n_observations: int):
        self.mu = mu
        self.cov = cov
        self.n_observations = n_observations

    def simulate(self) -> (np.array, np.array):
        x = np.random.multivariate_normal(self.mu.flatten(), self.cov, size=self.n_observations)
        return x.mean(axis=0).reshape(-1, 1), LedoitWolf().fit(x).covariance_
