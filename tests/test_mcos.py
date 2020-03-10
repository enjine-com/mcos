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
from mcos.optimizer import HRPOptimizer, MarkowitzOptimizer, NCOOptimizer


np.random.seed(0)  # use a random seed for predictable numbers

prices_df = pd.read_csv('tests/stock_prices.csv', parse_dates=True, index_col='date')
mu = mean_historical_return(prices_df).values
cov = sample_cov(prices_df).values


@pytest.mark.parametrize('simulator, estimator, transformers, expected_mean, expected_stdev', [
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.07580845, 0.05966212, -0.02893896]),
        np.array([0.03445259, 0.03214469, 0.01724587])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [],
        np.array([0.08776566, 0.05161399, -0.03332758]),
        np.array([0.0357396, 0.10285164, 0.02141213])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.69489515, 0.54737912, -0.1598759]),
        np.array([0.14173187, 0.47500561, 0.32839209])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [],
        np.array([0.33034865, 0.30012442, 0.01991391]),
        np.array([0.25863781, 0.19326429, 0.43201426])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.01599112, 0.029859, 0.01122259]),
        np.array([0.00042718, 0.01132642, 0.00416026])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [],
        np.array([0.03622878, 0.20588246, 0.01203012]),
        np.array([0.01082472, 0.09488349, 0.00180139])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.0444241, 0.0637391, -0.0214148]),
        np.array([0.0188791, 0.0202428, 0.0081114])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [],
        np.array([0.03792157, 0.03298662, -0.04898991]),
        np.array([0.04289238, 0.04494566, 0.00527806])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.66794992,  0.71132483, -0.579754]),
        np.array([0.067115, 0.04504799, 0.0442776])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [],
        np.array([0.2498265, 0.08673413, -0.52148866]),
        np.array([0.05304052, 0.21159622, 0.06312135])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.00868587, 0.01344688, 0.00258779]),
        np.array([0.00133915, 0.00188729, 0.0004534])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [],
        np.array([0.013888, 0.0548296, 0.0046528]),
        np.array([0.0008, 0.0511335, 0.0017761])
    )
])
def test_simulate_observations(simulator, estimator, transformers, expected_mean, expected_stdev):
    df = simulate_optimizations(simulator,
                                n_sims=3,
                                optimizers=[MarkowitzOptimizer(), NCOOptimizer(), HRPOptimizer()],
                                error_estimator=estimator,
                                covariance_transformers=transformers)

    assert_almost_equal(df['mean'].values, expected_mean)
    assert_almost_equal(df['stdev'].values, expected_stdev)
