import numpy as np
from numpy.testing import assert_array_almost_equal
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from mcos.optimizer import MarkowitzOptimizer, NCOOptimizer


class TestMarkowitzOptimizer:

    def test_allocate(self, prices_df):
        mu = mean_historical_return(prices_df).values
        cov = sample_cov(prices_df).values

        weights = MarkowitzOptimizer().allocate(mu, cov)

        assert_array_almost_equal(weights, np.array([
            0.01269, 0.09202, 0.19856, 0.09642, 0.07158,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.06129, 0.24562, 0.18413, 0.0, 0.03769
        ]))

    def test_name(self):
        assert MarkowitzOptimizer().name == 'markowitz'


class TestNCOOptimizier:

    def test_allocate_max_sharpe(self, prices_df):
        mu = mean_historical_return(prices_df).values
        cov = sample_cov(prices_df).values

        weights = NCOOptimizer().allocate(mu, cov)
        assert len(weights) == 20

    def test_allocate_min_variance(self, prices_df):
        cov = sample_cov(prices_df).values

        weights = NCOOptimizer().allocate(mu=None, cov=cov)
        assert len(weights) == 20
