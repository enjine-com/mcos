import numpy as np

from mcos.error_estimator import AbstractErrorEstimator


class ExpectedOutcomeErrorEstimator(AbstractErrorEstimator):

    def estimate(self, allocation: np.array, optimal_allocation: np.array) -> float:
        optimal_allocation = np.repeat(optimal_allocation.T, 1, axis=0)
        estimation = (optimal_allocation - allocation).mean()
        return estimation
