import numpy as np
import pandas as pd
from typing import List

from mcos.covariance_transformer import CovarianceMatrixDeNoiser
from mcos.error_estimator import AbstractErrorEstimator
from mcos.observation_simulator import AbstractObservationSimulator
from mcos.optimizer import AbstractOptimizer


def simulate_optimizations(
        obs_simulator: AbstractObservationSimulator,
        n_sims: int,
        optimizers: List[AbstractOptimizer],
        error_estimator: AbstractErrorEstimator,
        de_noise: bool = True
) -> pd.DataFrame:
    error_estimates = {optimizer.name: [] for optimizer in optimizers}

    for i in range(n_sims):
        mu_hat, cov_hat = obs_simulator.simulate()

        if de_noise:
            cov_hat = CovarianceMatrixDeNoiser(cov_hat, obs_simulator.n_observations).transform()

        for optimizer in optimizers:
            allocation = optimizer.allocate(mu_hat, cov_hat)
            optimal_allocation = optimizer.allocate(obs_simulator.mu, obs_simulator.cov)

            estimation = error_estimator.estimate(obs_simulator.mu, obs_simulator.cov, allocation, optimal_allocation)
            error_estimates[optimizer.name].append(estimation)

    return pd.DataFrame([
        {
            'optimizer': optimizer.name,
            'mean': np.mean(error_estimates[optimizer.name]),
            'stdev': np.std(error_estimates[optimizer.name])
        } for optimizer in optimizers
    ]).set_index('optimizer')
