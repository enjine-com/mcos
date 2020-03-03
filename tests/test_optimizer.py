import numpy as np
from numpy.testing import assert_array_almost_equal
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from mcos.optimizer import MarkowitzOptimizer, NCOOptimizer, HRPOptimizer, RiskParityOptimizer
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
        mu = mean_historical_return(prices_df).values
        cov = sample_cov(prices_df).values

        weights = HRPOptimizer().allocate(mu, cov)
        assert_almost_equal(weights, np.array(
                [0.02182165, 0.01831474, 0.01928443, 0.08689840, 0.04869276, 0.07706293,
                 0.02267915, 0.10417196, 0.10925653, 0.02917370, 0.07311320, 0.04267337,
                 0.06600238, 0.04105381, 0.02740903, 0.03910023, 0.02767476, 0.01861446,
                 0.05383763, 0.07316479]))

    def test_name(self):
        assert HRPOptimizer().name == 'HRP'


class TestRiskParityOptimizer:

    mu = [0.14, 0.12, 0.15, 0.07]

    cov = [[1.23, 0.375, 0.7, 0.3],
           [0.375, 1.22, 0.72, 0.135],
           [0.7, 0.72, 3.21, -0.32],
           [0.3, 0.135, -0.32, 0.52]]

    def test_allocate(self):
        x_t = [0.25, 0.25, 0.25, 0.25]  # your risk budget percent of total portfolio risk (equal risk)
        w0 = [1/4]*4

        weights = RiskParityOptimizer().allocate(self.mu, self.cov, x_t, w0)
        assert_almost_equal(weights, np.array(
            [0.19543974,  0.21521557,  0.16260951,  0.42673519]))

    def test_allocate_custom_risk_budget(self):
        x_t = [0.25, 0.25, 0.25, 0.25]  # your risk budget percent of total portfolio risk (equal risk)
        w0 = [1/4]*4

        weights = RiskParityOptimizer().allocate(self.mu, self.cov, x_t, w0)
        assert_almost_equal(weights, np.array(
            [0.22837243, 0.25116466, 0.08875776, 0.43170515]))

    def test_name(self):
        assert RiskParityOptimizer().name == 'Risk Parity'

