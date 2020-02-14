from abc import ABC, abstractmethod
import numpy as np
from mcos.de_noise import de_noise_covariance_matrix


class AbstractCovarianceTransformer(ABC):
    """
    Abstract class for transforming a covariance matrix
    """

    def __init__(self, cov: np.array, n_observations: int):
        self.cov = cov
        self.n_observations = n_observations

    @abstractmethod
    def transform(self) -> np.array:
        """
        Transforms a covariance matrix
        @return transformed covariance matrix
        """
        pass


class CovarianceMatrixDeNoiser(AbstractCovarianceTransformer):
    def __init__(self, cov: np.array, n_observations: int):
        super().__init__(cov, n_observations)

    def transform(self) -> np.array:
        """
        De-noises a covariance matrix as outlined in section 4.2 of "A Robust Estimator of the Efficient Frontier"
        @return transformed covariance matrix
        """
        return de_noise_covariance_matrix(self.cov, self.n_observations)
