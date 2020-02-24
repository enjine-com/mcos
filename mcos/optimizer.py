from typing import Dict

import numpy as np
from abc import ABC, abstractmethod
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from mcos.covariance_transformer import cov_to_corr


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
    Nested clustered optimization (NCO) optimizer based on section 4.3 of "A Robust Estimator of the Efficient Frontier
    """

    def __init__(self, max_num_clusters: int = None, num_clustering_trials=10):
        """
        Set optional variables used during calculations
        :param max_num_clusters: max number of clusters to use during KMeans clustering
        :param num_clustering_trials: number of times to perform KMeans clustering with [1,max_num_clusters] clusters
        """
        self.max_num_clusters = max_num_clusters
        self.num_clustering_trials = num_clustering_trials

    @property
    def name(self) -> str:
        return 'NCO'

    def allocate(self, mu: np.array, cov: np.array) -> np.array:
        """
        Perform the NCO method described in section 4.3 of "A Robust Estimator of the Efficient Frontier"

        Excerpt from section 4.3:

        The NCO method estimates 𝜔̂∗ while controlling for the signal-induced estimation errors
        explained in section 3.2. NCO works as follows:

        First, we cluster the covariance matrix into subsets of highly-correlated variables.
        One possible clustering algorithm is the partitioning method discussed in López de Prado and Lewis [2019],
        but hierarchical methods may also be applied. The result is a partition of the original set,
        that is, a collection of mutually disjoint nonempty subsets of variables.

        Second, we compute optimal allocations for each of these clusters separately.
        This allows us to collapse the original covariance matrix into a reduced covariance matrix,
        where each cluster is represented as a single variable. The collapsed correlation matrix is
        closer to an identity matrix than the original correlation matrix was, and therefore more
        amenable to optimization problems (recall the discussion in section 3.2).

        Third, we compute the optimal allocations across the reduced covariance matrix.

        Fourth,
        the final allocations are the dot-product of the intra-cluster allocations and the inter-cluster allocations.

        By splitting the problem into two separate tasks, NCO contains the instability within each cluster:
        the instability caused by intra-cluster noise does not propagate across clusters.
        See López de Prado [2019] for examples, code and additional details regarding NCO.

        :param cov: Covariance matrix
        :param mu: Expected return vector
        :return: Min variance portfolio if mu is None, max sharpe ratio portfolio if mu is not None
        """
        cov = pd.DataFrame(cov)

        if mu is not None:
            mu = pd.Series(mu)
        # get correlation matrix
        corr = cov_to_corr(cov)

        # find the optimal partition of clusters
        clusters = self._cluster_k_means_base(corr)

        # calculate intra-cluster allocations by finding the optimal portfolio for each cluster
        intra_cluster_allocations = pd.DataFrame(0, index=cov.index, columns=clusters.keys())
        for cluster_id, cluster in clusters.items():
            cov_ = cov.loc[cluster, cluster].values
            mu_ = mu.loc[cluster].values.reshape(-1, 1) if mu is not None else None
            intra_cluster_allocations.loc[cluster, cluster_id] = self._get_optimal_portfolio(cov_, mu_)

        # reduce covariance matrix
        cov = intra_cluster_allocations.T.dot(np.dot(cov, intra_cluster_allocations))
        mu = intra_cluster_allocations.T.dot(mu) if mu is not None else None

        # calculate inter_cluster allocations on reduced covariance matrix
        inter_cluster_allocations = pd.Series(self._get_optimal_portfolio(cov, mu), index=cov.index)

        # final allocations are the dot-product of the intra-cluster allocations and the inter-cluster allocations
        return intra_cluster_allocations \
            .mul(inter_cluster_allocations, axis=1) \
            .sum(axis=1).values \
            .reshape(-1, 1) \
            .flatten()

    def _cluster_k_means_base(self, corr: np.array) -> Dict[int, int]:
        """
        Using KMeans clustering, group the matrix into groups of highly correlated variables.
        The result is a partition of the original set,
        that is, a collection of mutually disjoint nonempty subsets of variables.
        :param corr: correlation matrix
        :return: The optimal partition of clusters
        """
        distance_matrix = ((1 - corr.fillna(0)) / 2.) ** .5
        silhouettes = pd.Series()

        max_num_clusters = self.max_num_clusters
        if max_num_clusters is None:
            # if the max number of clusters wasn't specified, declare it based on corr
            max_num_clusters = corr.shape[0] // 2

        for _ in range(self.num_clustering_trials):
            for i in range(2, max_num_clusters + 1):  # find optimal num clusters
                kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=1, random_state=42)

                kmeans_ = kmeans_.fit(distance_matrix)
                silhouettes_ = silhouette_samples(distance_matrix, kmeans_.labels_)

                new_calc = silhouettes_.mean() / silhouettes_.std()
                old_calc = silhouettes.mean() / silhouettes.std()

                if np.isnan(old_calc) or new_calc > old_calc:
                    silhouettes, kmeans = silhouettes_, kmeans_

        clusters = {
            i: corr.columns[np.where(kmeans.labels_ == i)].tolist()
            for i in np.unique(kmeans.labels_)
        }  # cluster members

        return clusters

    def _get_optimal_portfolio(self, cov: np.array, mu: np.array) -> np.array:
        """
        compute the optimal allocations across the reduced covariance matrix
        :param cov: covariance matrix
        :param mu: vector of expected returns
        :return: optimal portfolio allocation
        """
        inv = np.linalg.inv(cov)
        ones = np.ones(shape=(inv.shape[0], 1))

        if mu is None:
            mu = ones

        w = np.dot(inv, mu)
        w /= np.dot(ones.T, w)
        return w.flatten()
