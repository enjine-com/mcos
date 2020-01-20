import pytest
from pypfopt import expected_returns, risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier

from mcos.optimizer import MarkowitzOptimizer


@pytest.fixture
def optimizer():
    return MarkowitzOptimizer()


class TestMarkowitzOptimizer:
    
    def test_allocate(self, optimizer, stock_prices_df):
        mu = expected_returns.mean_historical_return(stock_prices_df)
        cov = risk_models.sample_cov(stock_prices_df)
        
        actual_weights = optimizer.allocate(mu, cov)

        ef = EfficientFrontier(mu, cov)
        expected_weights = ef.max_sharpe()
        
        assert actual_weights == expected_weights


    def test_name(self, optimizer):
        assert optimizer.name == 'markowitz'
