import numpy as np

from mcos.error_estimator import ExpectedOutcomeErrorEstimator

class TestExpectedOutcomeErrorEstimator:
    
    def test_estimate(self):
        allocation = np.array([
            0.01269, 0.09202, 0.19856, 0.09642, 0.07158,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.06129, 0.24562, 0.18413, 0.0, 0.03769
        ])

        ExpectedOutcomeErrorEstimator().estimate(allocation)
        # TODO: Finish test