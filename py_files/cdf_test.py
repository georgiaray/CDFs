import numpy as np
import pytest
from scipy.stats import norm

def numerical_cdf_from_pdf(pdf_func, x_grid):
    """Numerically compute the CDF from a PDF using the trapezoidal rule."""
    y_pdf = pdf_func(x_grid)
    cdf = np.cumsum(y_pdf) * np.diff(x_grid, prepend=x_grid[0])
    # Normalize so that CDF goes from 0 to 1
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    return cdf

class TestPDFandCDF:
    def test_cdf_monotonicity(self):
        """Test that the CDF is monotonically non-decreasing."""
        # Example: standard normal PDF and its CDF
        x = np.linspace(-5, 5, 1000)
        pdf = norm.pdf
        cdf = norm.cdf

        y_cdf = cdf(x)
        assert np.all(np.diff(y_cdf) >= 0), "CDF should be monotonically non-decreasing"

    def test_cdf_limits(self):
        """Test that the CDF approaches 0 at -inf and 1 at +inf."""
        x = np.linspace(-10, 10, 1000)
        cdf = norm.cdf(x)
        assert np.isclose(cdf[0], 0, atol=1e-4), "CDF at -inf should be ~0"
        assert np.isclose(cdf[-1], 1, atol=1e-4), "CDF at +inf should be ~1"

    def test_cdf_matches_numerical_integration(self):
        """Test that the CDF matches the numerical integral of the PDF."""
        x = np.linspace(-5, 5, 1000)
        pdf = norm.pdf
        cdf_true = norm.cdf(x)
        cdf_num = numerical_cdf_from_pdf(pdf, x)
        # Allow for small numerical error and possible offset
        # Align the two CDFs by subtracting their minimums
        cdf_true = (cdf_true - cdf_true.min()) / (cdf_true.max() - cdf_true.min())
        assert np.allclose(cdf_num, cdf_true, atol=2e-2), "Numerical CDF should match analytical CDF"

    def test_cdf_is_between_0_and_1(self):
        """Test that the CDF is always between 0 and 1."""
        
        x = np.linspace(-5, 5, 1000)
        cdf = norm.cdf(x)
        assert np.all((0 <= cdf) & (cdf <= 1)), "CDF values should be in [0, 1]"

    def test_pdf_is_derivative_of_cdf(self):
        """Test that the PDF is the derivative of the CDF."""
        x = np.linspace(-5, 5, 1000)
        cdf = norm.cdf(x)
        pdf = norm.pdf(x)
        # Numerical derivative
        d_cdf = np.gradient(cdf, x)
        # Compare in the central region to avoid edge effects
        assert np.allclose(d_cdf[100:-100], pdf[100:-100], atol=1e-3), "PDF should be derivative of CDF"

    def test_custom_pdf_and_cdf(self):
        """Test a custom PDF and its CDF."""
        # Exponential distribution: PDF = lambda * exp(-lambda x), x >= 0
        # CDF = 1 - exp(-lambda x)
        lam = 2.0
        def pdf(x):
            return lam * np.exp(-lam * x) * (x >= 0)
        def cdf(x):
            return (1 - np.exp(-lam * x)) * (x >= 0) + 0.0 * (x < 0)
        x = np.linspace(-1, 5, 1000)
        y_cdf = cdf(x)
        y_cdf_num = numerical_cdf_from_pdf(pdf, x)
        # Only compare for x >= 0
        mask = x >= 0
        y_cdf = (y_cdf - y_cdf[mask].min()) / (y_cdf[mask].max() - y_cdf[mask].min())
        y_cdf_num = (y_cdf_num - y_cdf_num[mask].min()) / (y_cdf_num[mask].max() - y_cdf_num[mask].min())
        assert np.allclose(y_cdf_num[mask], y_cdf[mask], atol=2e-2), "Numerical CDF should match analytical CDF for exponential"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
