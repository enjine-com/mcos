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
    def allocate(self, mu: np.array, cov: np.array) -> np.array:
        return self._nco(cov, mu)

    @property
    def name(self) -> str:
        return 'NCO'

    def _nco(self, cov, mu=None, max_num_clusters=None):
        cov = pd.DataFrame(cov)

        if mu is not None:
            mu = pd.Series(mu)

        corr1 = cov_to_corr(cov)
        corr1, clstrs, _ = self._cluster_k_means_base(corr1, max_num_clusters, n_init=10)
        w_intra = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())
        for i in clstrs:
            cov_ = cov.loc[clstrs[i], clstrs[i]].values
            mu_ = (None if mu is None else mu.loc[clstrs[i]].values.reshape(-1, 1))
            w_intra.loc[clstrs[i], i] = self._opt_port(cov_, mu_).flatten()

        cov_ = w_intra.T.dot(np.dot(cov, w_intra))  # reduce covariance matrix
        mu_ = (None if mu is None else w_intra.T.dot(mu))

        w_inter = pd.Series(self._opt_port(cov_, mu_).flatten(), index=cov_.index)
        nco = w_intra.mul(w_inter, axis=1).sum(axis=1).values.reshape(-1, 1)

        return nco.flatten()

    def _cluster_k_means_base(self, corr0, max_num_clusters=None, n_init=10):
        dist, silh = ((1 - corr0.fillna(0)) / 2.) ** .5, pd.Series()  # distance matrix

        if max_num_clusters is None:
            max_num_clusters = corr0.shape[0] // 2

        for init in range(n_init):
            for i in range(2, max_num_clusters + 1):  # find optimal num clusters
                kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=1)

        kmeans_ = kmeans_.fit(dist)
        silh_ = silhouette_samples(dist, kmeans_.labels_)
        stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())

        if np.isnan(stat[1]) or stat[0] > stat[1]:
            silh, kmeans = silh_, kmeans_
        newIdx = np.argsort(kmeans.labels_)

        corr1 = corr0.iloc[newIdx]  # reorder rows
        corr1 = corr1.iloc[:, newIdx]  # reorder columns
        clstrs = {
            i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist()
            for i in np.unique(kmeans.labels_)
        }  # cluster members

        silh = pd.Series(silh, index=dist.index)
        return corr1, clstrs, silh

    def _opt_port(self, cov, mu=None):
        inv = np.linalg.inv(cov)
        ones = np.ones(shape=(inv.shape[0], 1))

        if mu is None:
            mu = ones

        w = np.dot(inv, mu)
        w /= np.dot(ones.T, w)
        return w
