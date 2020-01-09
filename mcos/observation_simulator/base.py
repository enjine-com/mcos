from abc import abstractmethod, ABC
import numpy as np

class ObservationSimulator(ABC):

    @abstractmethod
    def simulate(self) -> (np.array, np.array):
        """
        Draws empirical means and covariances. See section 4.1 of the "A Robust Estimator of the Efficient Frontier"
        paper.
        :param n_observations:
        @return: Tuple of expected return vector and covariance matrix
        """
        pass