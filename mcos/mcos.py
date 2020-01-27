from typing import List

from mcos.error_estimator import AbstractErrorEstimator
from mcos.observation_simulator.base import ObservationSimulator
from mcos.optimizer import AbstractOptimizer
import numpy as np
import pandas as pd


def simulate_optimizations(obs_simulator: ObservationSimulator, n_sims: int, optimizers: List[AbstractOptimizer],
             error_estimator: AbstractErrorEstimator, de_noise: bool = True) -> pd.DataFrame:
    error_estimates = {optimizer.name: [] for optimizer in optimizers}

    for i in range(n_sims):
        mu_hat, cov_hat = obs_simulator.simulate()

        if de_noise:
            pass  # insert denoising function here

        for optimizer in optimizers:
            allocation = optimizer.allocate(mu_hat, cov_hat)
            optimal = optimal_allocation(mu_hat, cov_hat)
            error_estimates[optimizer.name].append(error_estimator.estimate(allocation, optimal))

    return pd.DataFrame([
        {
            'optimizer': optimizer.name,
            'mean': np.mean(error_estimates[optimizer.name]),
            'stdev': np.std(error_estimator[optimizers.name])
        } for optimizer in optimizers
    ]).set_index('optimizer')


def optimal_allocation(mu: np.array, cov: np.array):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0],1))
    allocation = np.dot(inv,mu)
    allocation /= np.dot(ones.T, allocation)
    return allocation
