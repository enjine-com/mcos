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
        assert_array_almost_equal(weights, [0.1563231, 0.08557526, 0.15554872, 0.13261652, 0.06284074, -0.04279202,
                                            -0.00453155, 0.07075119, -0.14191424, -0.04364646, -0.01391411, 0.02153863,
                                            -0.00173282, - 0.00286055, -0.03839033, 0.05647993, 0.29571663, 0.17691535,
                                            -0.00041939, 0.07589542])

    def test_allocate_min_variance(self, prices_df):
        cov = sample_cov(prices_df).values

        weights = NCOOptimizer().allocate(mu=None, cov=cov)
        assert_array_almost_equal(weights, [0.08187591, -0.00788608, 0.08432646, 0.12744211, -0.00260273, 0.05578475,
                                            -0.00262913, 0.15081227, -0.01954205, 0.08665619, 0.145736, -0.01389383,
                                            0.01004943, 0.19745493, 0.01791739, -0.00585418, -0.00558448, 0.09229292,
                                            -0.0221254, 0.02976952])
