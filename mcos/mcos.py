import numpy as np
import pandas as pd
from typing import List
from mcos.de_noising.de_noise import de_noise_covariance_matrix
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
            cov_hat = de_noise_covariance_matrix(
                covariance_matrix=cov_hat,
                q=obs_simulator.n_observations * 1.0 / cov_hat.shape[1],
                bandwidth=.25
            )

        for optimizer in optimizers:
            allocation = optimizer.allocate(mu_hat, cov_hat)
            optimal_allocation = optimizer.allocate(obs_simulator.mu, obs_simulator.cov)

            estimation = error_estimator.estimate(obs_simulator.mu, obs_simulator.cov, allocation, optimal_allocation)
            error_estimates[optimizer.name].append(estimation)

    return pd.DataFrame([
        {
            'optimizer': optimizer.name,
            'mean': np.mean(error_estimates[optimizer.name]),
            'stdev': np.std(error_estimator[optimizers.name])
        } for optimizer in optimizers
    ]).set_index('optimizer')
