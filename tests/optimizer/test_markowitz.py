import numpy as np
from numpy.testing import assert_array_almost_equal
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from mcos.optimizer import MarkowitzOptimizer


class TestMarkowitzOptimizer:
    
    def test_allocate(self, prices_df):
        mu = mean_historical_return(prices_df).values
        cov = sample_cov(prices_df).values
        
        weights = MarkowitzOptimizer().allocate(mu, cov)

        assert_array_almost_equal(weights, np.array([
            1.26864712e-02, 9.20150411e-02, 1.98561424e-01, 9.64204515e-02,
            7.15845953e-02, 1.13388593e-16, 1.33115644e-16, 9.62050054e-17,
            9.71530649e-17, 1.00476713e-17, 6.08818118e-17, 5.02624023e-17,
            4.54800454e-17, 6.99460900e-17, 7.47511984e-17, 6.12866760e-02,
            2.45623070e-01, 1.84128277e-01, 9.63328196e-17, 3.76939933e-02
        ]))

    def test_name(self):
        assert MarkowitzOptimizer().name == 'markowitz'
