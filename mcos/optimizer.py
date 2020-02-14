import numpy as np
from abc import ABC, abstractmethod
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from mcos.de_noise import cov_to_corr


class AbstractOptimizer(ABC):
    """Helper class that provides a standard way to create a new Optimizer using inheritance"""

    @abstractmethod
    def allocate(self, mu: np.array, cov: np.array) -> np.array:
        """
        Create an optimal portfolio allocation given the expected returns vector and covariance matrix. See section 4.3
        of the "A Robust Estimator of the Efficient Frontier" paper.
        @param mu: Expected return vector
        @param cov: Expected covariance matrix
        @return Vector of weights
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of this optimizer. The name will be displayed in the MCOS results DataFrame.
        """
        pass


class MarkowitzOptimizer(AbstractOptimizer):
    """Optimizer based on the Modern Portfolio Theory pioneered by Harry Markowitz's paper 'Portfolio Selection'"""

    def allocate(self, mu: np.array, cov: np.array) -> np.array:
        ef = EfficientFrontier(mu, cov)
        ef.max_sharpe()
        weights = ef.clean_weights()

        return np.array(list(weights.values()))

    @property
    def name(self) -> str:
        return 'markowitz'


class NCOOptimizer(AbstractOptimizer):
    """
    NCO Optimizer
    """

    def allocate(self, mu: np.array, cov: np.array) -> np.array:
        return self._nco(cov, mu)

    @property
    def name(self) -> str:
        return 'NCO'

    def _nco(self, cov, mu=None, max_num_clusters=None):
        cov = pd.DataFrame(cov)

        if mu is not None:
            mu = pd.Series(mu)

        corr = cov_to_corr(cov)
        clusters = self._cluster_k_means_base(corr, max_num_clusters, n_init=10)
        w_intra = pd.DataFrame(0, index=cov.index, columns=clusters.keys())
        for key, value in clusters.items():
            cov_ = cov.loc[value, value].values
            mu_ = None if mu is None else mu.loc[value].values.reshape(-1, 1)
            w_intra.loc[value, key] = self._opt_port(cov_, mu_).flatten()

        cov = w_intra.T.dot(np.dot(cov, w_intra))  # reduce covariance matrix
        mu = None if mu is None else w_intra.T.dot(mu)

        w_inter = pd.Series(self._opt_port(cov, mu).flatten(), index=cov.index)
        nco = w_intra.mul(w_inter, axis=1).sum(axis=1).values.reshape(-1, 1)

        return nco.flatten()

    def _cluster_k_means_base(self, corr, max_num_clusters=None, n_init=10):
        distance_matrix = ((1 - corr.fillna(0)) / 2.) ** .5
        silhouettes = pd.Series()
        if max_num_clusters is None:
            max_num_clusters = corr.shape[0] // 2

        for _ in range(n_init):
            for i in range(2, max_num_clusters + 1):  # find optimal num clusters
                kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=1)

                kmeans_ = kmeans_.fit(distance_matrix)
                silh_ = silhouette_samples(distance_matrix, kmeans_.labels_)
                stat1 = silh_.mean() / silh_.std()
                stat2 = silhouettes.mean() / silhouettes.std()

                if np.isnan(stat2) or stat1 > stat2:
                    silhouettes, kmeans = silh_, kmeans_

        clusters = {
            i: corr.columns[np.where(kmeans.labels_ == i)].tolist()
            for i in np.unique(kmeans.labels_)
        }  # cluster members

        return clusters

    def _opt_port(self, cov, mu=None):
        inv = np.linalg.inv(cov)
        ones = np.ones(shape=(inv.shape[0], 1))

        if mu is None:
            mu = ones

        w = np.dot(inv, mu)
        w /= np.dot(ones.T, w)
        return w
