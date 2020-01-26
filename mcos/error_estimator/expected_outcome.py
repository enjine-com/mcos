import numpy as np

from mcos.error_estimator import AbstractErrorEstimator


class ExpectedOutcomeErrorEstimator(AbstractErrorEstimator):

    def estimate(self, allocation: np.array) -> float:
        true_allocation = np.repeat(allocation.T, 1, axis=0)
        estimation = (true_allocation - allocation).mean()
        return estimation
        
