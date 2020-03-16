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
from mcos.optimizer import HRPOptimizer, MarkowitzOptimizer, NCOOptimizer, RiskParityOptimizer


prices_df = pd.read_csv('tests/stock_prices.csv', parse_dates=True, index_col='date')
mu = mean_historical_return(prices_df).values
cov = sample_cov(prices_df).values


@pytest.mark.parametrize('simulator, estimator, transformers, expected_mean, expected_stdev', [
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.07580845,  0.05966212, -0.02893896,  0.0085226]),
        np.array([0.03445259, 0.03214469, 0.01724587, 0.01244282])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [],
        np.array([0.05043029, -0.0761952 , -0.03200537, -0.00413669]),
        np.array([0.05422127, 0.25850676, 0.0196157 , 0.01376204])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.05183939,  0.07230958, -0.02051239, -0.0086959]),
        np.array([0.00979491, 0.00997131, 0.00640547, 0.00077058])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [],
        np.array([0.05010366,  0.05885478, -0.0375485 , -0.00431366]),
        np.array([0.02262029, 0.02348383, 0.00511152, 0.01081385])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.0344777, 0.0384286, 0.0100253]),
        np.array([0.01721, 0.0161673, 0.0033896])
    ),
    (
        MuCovObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [],
        np.array([0.0541653, 3.2502716, 0.0201481]),
        np.array([0.0086772, 2.040143, 0.0042452])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.0518401, 0.0723096, -0.0231457]),
        np.array([0.0097951, 0.0099713, 0.0083273])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        ExpectedOutcomeErrorEstimator(),
        [],
        np.array([0.0501037,  0.0588548, -0.0375485]),
        np.array([0.0226203, 0.0234838, 0.0051115])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.5263593, 0.5868035, -0.4764438]),
        np.array([0.1273877, 0.0830139, 0.076446])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        SharpeRatioErrorEstimator(),
        [],
        np.array([0.3768555,  0.3574208, -0.4637792]),
        np.array([0.1779364, 0.1389299, 0.0346609])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [DeNoiserCovarianceTransformer()],
        np.array([0.0100163, 0.0152269, 0.0023996]),
        np.array([0.0011785, 0.0011361, 0.0011716])
    ),
    (
        MuCovLedoitWolfObservationSimulator(mu, cov, n_observations=5),
        VarianceErrorEstimator(),
        [],
        np.array([0.018974, 0.0269581, 0.0065477]),
        np.array([0.0031962, 0.0026226, 0.0009597])
    )
])
def test_simulate_observations(simulator, estimator, transformers, expected_mean, expected_stdev):
    np.random.seed(0)  # use a random seed for predictable numbers

    df = simulate_optimizations(simulator,
                                n_sims=3,
                                optimizers=[MarkowitzOptimizer(), NCOOptimizer(), HRPOptimizer(), RiskParityOptimizer()],
                                error_estimator=estimator,
                                covariance_transformers=transformers)
    import pdb;
    pdb.set_trace()
    assert_almost_equal(df['mean'].values, expected_mean, decimal=1)
    assert_almost_equal(df['stdev'].values, expected_stdev, decimal=1)
