from mcos.observation_simulator.base import ObservationSimulator
import numpy as np


class MuCovObservationSimulator(ObservationSimulator):
    def __init__(self, mu: np.array, cov: np.array):
        self.mu = mu
        self.cov = cov

    def simulate(self, n_observations: int) -> (np.array, np.array):
        x = np.random.multivariate_normal(self.mu.flatten(), self.cov, size=n_observations)
        return x.mean(axis=0).reshape(-1, 1), np.cov(x, rowvar=False)