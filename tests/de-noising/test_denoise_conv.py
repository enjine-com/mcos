from pypfopt.risk_models import sample_cov
import pytest
from mcos.denoising import denoise_conv


class TestDeNoiseConv:
    def test_fit_KDE(self):
        pass

    def test_mp_PDF(self):
        pass

    def test_err_PDFs(self):
        pass

    def test_find_max_eval(self):
        pass

    def test_corr_to_cov(self):
        pass

    def test_cov_to_corr(self):
        pass

    def test_get_PCA(self):
        pass

    def test_denoised_corr(self):
        pass

    def test_de_noise_cov(self, covariance_matrix):
        results = denoise_conv.de_noise_cov(covariance_matrix, .5, .25)
        assert results.shape == (20, 20)

    @pytest.fixture
    def covariance_matrix(self, prices_df):
        return sample_cov(prices_df).values
