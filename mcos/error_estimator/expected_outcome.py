import numpy as np

from mcos.error_estimator import AbstractErrorEstimator


class ExpectedOutcomeErrorEstimator(AbstractErrorEstimator):

    def estimate(self, allocation: np.array, optimal_allocation: np.array) -> float:
        return (optimal_allocation - allocation).mean()
