import numpy as np

from mcos.error_estimator import AbstractErrorEstimator


class ExpectedOutcomeErrorEstimator(AbstractErrorEstimator):
    """Error Estimator that calculates the mean difference in expected outcomes"""

    def estimate(self, mu: np.array, cov: np.array, allocation: np.array, optimal_allocation: np.array) -> float:
        return np.dot(optimal_allocation - allocation, mu)
