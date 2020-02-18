from mcos.covariance_transformer import CovarianceMatrixDeNoiser
from numpy.testing import assert_almost_equal
from pypfopt.risk_models import sample_cov


class TestCovarianceMatrixDeNoiser:
    def test_transform(self, prices_df, de_noised_covariance_matrix_results):
        covariance_matrix = sample_cov(prices_df).values
        n_observations = prices_df.size
        results = CovarianceMatrixDeNoiser().transform(covariance_matrix, n_observations)
        assert results.shape == (20, 20)
        assert_almost_equal(results, de_noised_covariance_matrix_results)
