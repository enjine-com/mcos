import numpy as np
from abc import abstractmethod, ABC
from sklearn.covariance import LedoitWolf
import pandas as pd
from pypfopt import risk_models
from pypfopt import expected_returns


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


def convert_price_history(df: pd.DataFrame):
    """
    converts a price history dataframe into expected returns and covariance
     :param df: Dataframe of price histories indexed by data
     @return
    """
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    cov = risk_models.sample_cov(df)
    return mu, cov
