import numpy as np
from numpy.testing import assert_almost_equal

from mcos.error_estimator import ExpectedOutcomeErrorEstimator

class TestExpectedOutcomeErrorEstimator:
    
    def test_estimate(self):
        allocation = np.array([
            0.01269, 0.09202, 0.19856, 0.09642, 0.07158,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.06129, 0.24562, 0.18413, 0.0, 0.03769
        ])

        optimal_allocation = np.array([
            7.25281436e-02,  1.06906248e-01,  1.94837456e-01,  1.34410113e-01, 7.87365673e-02, 
            -7.59976774e-02, -5.16106485e-02,  4.61772510e-02, -1.47621585e-01, -7.60084126e-02,
            1.32030839e-02, -6.71331007e-03, -5.43546989e-02,  4.55553915e-02,  2.73124408e-04,
            8.56106000e-02, 3.48230364e-01,  2.24646885e-01, -1.32869686e-02,  7.44780725e-02])

        estimation = ExpectedOutcomeErrorEstimator().estimate(allocation, optimal_allocation)

        assert_almost_equal(estimation, -4.310000065177455e-11)

        