import numpy as np
from numpy.testing import assert_array_almost_equal
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from mcos.optimizer import MarkowitzOptimizer, NCOOptimizer, HRPOptimizer
from numpy.testing import assert_almost_equal


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


class TestNCOOptimizer:

    def test_allocate_max_sharpe(self, prices_df):
        mu = mean_historical_return(prices_df).values
        cov = sample_cov(prices_df).values

        weights = NCOOptimizer().allocate(mu, cov)
        assert_array_almost_equal(weights, [0.1563231, 0.08557526, 0.15554872, 0.13261652, 0.06284074, -0.04279202,
                                            -0.00453155, 0.07075119, -0.14191424, -0.04364646, -0.01391411, 0.02153863,
                                            -0.00173282, -0.00286055, -0.03839033, 0.05647993, 0.29571663, 0.17691535,
                                            -0.00041939, 0.07589542])

    def test_allocate_min_variance(self, prices_df):
        cov = sample_cov(prices_df).values

        weights = NCOOptimizer().allocate(mu=None, cov=cov)
        assert_array_almost_equal(weights, [0.08187591, -0.00788608, 0.08432646, 0.12744211, -0.00260273, 0.05578475,
                                            -0.00262913, 0.15081227, -0.01954205, 0.08665619, 0.145736, -0.01389383,
                                            0.01004943, 0.19745493, 0.01791739, -0.00585418, -0.00558448, 0.09229292,
                                            -0.0221254, 0.02976952])

    def test_name(self):
        assert NCOOptimizer().name == 'NCO'


class TestHRPOptimizer:

    def test_allocate(self, prices_df):
        # cov = np.array([[0.0625, 0.0225, -0.0125, -0.02], [0.0225, 0.09, 0.0, 0.108],
        #                 [-0.0125, 0.0, 0.25, 0.14], [-0.02, 0.108, 0.14, 0.16]])

        mu = mean_historical_return(prices_df).values
        cov = sample_cov(prices_df).values
        results = HRPOptimizer().allocate(mu, cov)
        assert_almost_equal(results, np.array(
                [0.02182165, 0.01831474, 0.01928443, 0.08689840, 0.04869276, 0.07706293,
                 0.02267915, 0.10417196, 0.10925653, 0.02917370, 0.07311320, 0.04267337,
                 0.06600238, 0.04105381, 0.02740903, 0.03910023, 0.02767476, 0.01861446,
                 0.05383763, 0.07316479]))

    def test_name(self):
        assert HRPOptimizer().name == 'HRP'
