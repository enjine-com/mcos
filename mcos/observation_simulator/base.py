from abc import abstractmethod, ABC


class ObservationSimulator(ABC):

    @abstractmethod
    def simulate(self, n_observations: int):
        """
        Base method that implements the step 1 of the MCOS process:  DRAWING EMPIRICAL MEANS AND COVARIANCES
        :param n_observations:
        """
        pass