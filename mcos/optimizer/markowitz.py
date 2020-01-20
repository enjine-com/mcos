import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier

from mcos.optimizer import AbstractOptimizer


class MarkowitzOptimizer(AbstractOptimizer):

    def allocate(self, mu: np.array, cov: np.array) -> np.array:
        ef = EfficientFrontier(mu, cov)
        return ef.max_sharpe()

    @property
    def name(self) -> str:
        return 'markowitz'
