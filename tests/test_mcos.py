import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from mcos.covariance_transformer import DeNoiserCovarianceTransformer
from mcos.error_estimator import ExpectedOutcomeErrorEstimator, SharpeRatioErrorEstimator, VarianceErrorEstimator
from mcos.mcos import simulate_optimizations
from mcos.observation_simulator import MuCovLedoitWolfObservationSimulator, MuCovObservationSimulator
from mcos.optimizer import MarkowitzOptimizer, NCOOptimizer

np.random.seed(0) # use a random seed for predictable numbers


@pytest.mark.parametrize('estimator, transformers, expected_mean, expected_stdev', [
    (
        ExpectedOutcomeErrorEstimator(), 
        [DeNoiserCovarianceTransformer()],
        np.array([0.05183701, 0.07230958]),
        np.array([0.00979434, 0.00997131])
    ),
    (
        ExpectedOutcomeErrorEstimator(),
        [],
        np.array([ 0.06518316, -0.00120242]),
        np.array([0.0256253 , 0.05974583])
    ),
    (
        SharpeRatioErrorEstimator(), 
        [DeNoiserCovarianceTransformer()],
        np.array([0.63895563, 0.49235394]),
        np.array([0.06549062, 0.31084956])
    ),
    (
        SharpeRatioErrorEstimator(), 
        [],
        np.array([0.27274608, 0.09228378]),
        np.array([0.05512106, 0.10013101])
    ),
    (
        VarianceErrorEstimator(), 
        [DeNoiserCovarianceTransformer()],
        np.array([0.01097861, 0.0191336]),
        np.array([0.00200612, 0.00226925])
    ),
    (
        VarianceErrorEstimator(), 
        [],
        np.array([0.01272781, 0.0910386]),
        np.array([0.00508202, 0.08666079])
    ),
])
def test_simulate_observations_mu_cov_ledoit(estimator, transformers, expected_mean, expected_stdev, prices_df):
    mu = mean_historical_return(prices_df).values
    cov = sample_cov(prices_df).values
    
    simulator = MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5)
    df = simulate_optimizations(simulator,
                                n_sims=3, 
                                optimizers=[MarkowitzOptimizer(), NCOOptimizer()],
                                error_estimator=estimator,
                                covariance_transformers=transformers)

    assert_almost_equal(df['mean'].values, expected_mean)
    assert_almost_equal(df['stdev'].values, expected_stdev)
    