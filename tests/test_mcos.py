import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from mcos.covariance_transformer import DeNoiserCovarianceTransformer
from mcos.error_estimator import ExpectedOutcomeErrorEstimator, SharpeRatioErrorEstimator, VarianceErrorEstimator
from mcos.mcos import simulate_optimizations
from mcos.observation_simulator import MuCovObservationSimulator, MuCovLedoitWolfObservationSimulator
from mcos.optimizer import MarkowitzOptimizer, NCOOptimizer


np.random.seed(0)  # use a random seed for predictable numbers

prices_df = pd.read_csv('tests/stock_prices.csv', parse_dates=True, index_col='date')
mu = mean_historical_return(prices_df).values
cov = sample_cov(prices_df).values


@pytest.mark.parametrize('simulator, estimator, transformers, expected_mean, expected_stdev', [
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.07580845, 0.05966212]),
        np.array([0.03445259, 0.03214469])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [],
        np.array([0.08776566, 0.05161399]),
        np.array([0.0357396, 0.10285164])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.69489515, 0.54737912]),
        np.array([0.14173187, 0.47500561])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [],
        np.array([0.33034865, 0.30012442]),
        np.array([0.25863781, 0.19326429])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.01599112, 0.029859]),
        np.array([0.00042718, 0.01132642])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [],
        np.array([0.03622878, 0.20588246]),
        np.array([0.01082472, 0.09488349])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.04442412, 0.06373909]),
        np.array([0.01887905, 0.0202428])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [],
        np.array([0.03792157, 0.03298662]),
        np.array([0.04289238, 0.04494566])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.66794992, 0.71132483]),
        np.array([0.067115, 0.04504799])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [],
        np.array([0.2498265, 0.08673413]),
        np.array([0.05304052, 0.21159622])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.00868587, 0.01344688]),
        np.array([0.00133915, 0.00188729])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [],
        np.array([0.01388797, 0.05482958]),
        np.array([0.00080001, 0.05113352])
    )
])
def test_simulate_observations(simulator, estimator, transformers, expected_mean, expected_stdev):
    df = simulate_optimizations(simulator,
                                n_sims=3,
                                optimizers=[MarkowitzOptimizer(), NCOOptimizer()],
                                error_estimator=estimator,
                                covariance_transformers=transformers)

    assert_almost_equal(df['mean'].values, expected_mean)
    assert_almost_equal(df['stdev'].values, expected_stdev)
