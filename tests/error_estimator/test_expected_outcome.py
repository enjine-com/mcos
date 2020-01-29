import numpy as np
from numpy.testing import assert_almost_equal
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from mcos.error_estimator import ExpectedOutcomeErrorEstimator

class TestExpectedOutcomeErrorEstimator:
    
    def test_estimate(self, prices_df):
        mu = mean_historical_return(prices_df).values
        cov = sample_cov(prices_df).values

        allocation = np.array([
            0.01269, 0.09202, 0.19856, 0.09642, 0.07158,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.06129, 0.24562, 0.18413, 0.0, 0.03769
        ])

        optimal_allocation = np.array([
            0.0, 0.15, 0.27, 0.1, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.02, 0.35, 0.1, 0.0, 0.01
        ])

        estimation = ExpectedOutcomeErrorEstimator().estimate(mu, cov, allocation, optimal_allocation)

        assert_almost_equal(estimation, 0.00277877)
