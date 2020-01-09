from typing import List

from mcos.error_estimator.base import ErrorEstimator
from mcos.observation_simulator.base import ObservationSimulator
from mcos.optimizer.base import Optimizer
import numpy as np
import pandas as pd


def simulate_optimizations(obs_simulator: ObservationSimulator, n_sims: int, optimizers: List[Optimizer],
             error_estimator: ErrorEstimator, de_noise: bool = True) -> pd.DataFrame:
    error_estimates = {optimizer.name: [] for optimizer in optimizers}

    for i in range(n_sims):
        mu_hat, cov_hat = obs_simulator.simulate()

        if de_noise:
            pass  # insert denoising function here

        for optimizer in optimizers:
            allocation = optimizer.allocate(mu_hat, cov_hat)
            error_estimates[optimizer.name].append(error_estimator.estimate(allocation))

    return pd.DataFrame([
        {
            'optimizer': optimizer.name,
            'mean': np.mean(error_estimates[optimizer.name]),
            'stdev': np.std(error_estimator[optimizers.name])
        } for optimizer in optimizers
    ]).set_index('optimizer')
