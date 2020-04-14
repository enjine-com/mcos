import numpy as np
import pandas as pd
from typing import List

from mcos.covariance_transformer import AbstractCovarianceTransformer
from mcos.error_estimator import AbstractErrorEstimator
from mcos.observation_simulator import AbstractObservationSimulator, MuCovLedoitWolfObservationSimulator, \
    MuCovObservationSimulator
from mcos.optimizer import AbstractOptimizer
from mcos.utils import convert_price_history


def simulate_optimizations(
        obs_simulator: AbstractObservationSimulator,
        n_sims: int,
        optimizers: List[AbstractOptimizer],
        error_estimator: AbstractErrorEstimator,
        covariance_transformers: List[AbstractCovarianceTransformer]
) -> pd.DataFrame:
    error_estimates = {optimizer.name: [] for optimizer in optimizers}

    for i in range(n_sims):
        mu_hat, cov_hat = obs_simulator.simulate()

        for transformer in covariance_transformers:
            cov_hat = transformer.transform(cov_hat, obs_simulator.n_observations)

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


def simulate_optimization_from_price_history(
        price_history: pd.DataFrame,
        observation_name: str,
        n_sims: int,
        optimizers: List[AbstractOptimizer],
        error_estimator: AbstractErrorEstimator,
        covariance_transformers: List[AbstractCovarianceTransformer]):

    mu, cov = convert_price_history(price_history)

    if observation_name.lower() == "mucovledoitwolfobservationsimulator":
        sim = MuCovLedoitWolfObservationSimulator(mu, cov, n_sims)
    elif observation_name == "mucovobservationsimulator":
        sim = MuCovObservationSimulator(mu, cov, n_sims)
    else:
        raise ValueError("Invalid observation simulator name")

    return simulate_optimizations(sim, n_sims, optimizers, error_estimator, covariance_transformers)
