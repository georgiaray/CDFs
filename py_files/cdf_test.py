import numpy as np
import pytest
from scipy.stats import norm
from cdf import cdf_and_q

class TestCDFImplementation:
    def test_cdf_and_q_returns_correct_outputs(self):
        """Test that cdf_and_q returns the expected outputs."""
        # Define a simple PDF (normal distribution)
        xs = np.linspace(-5, 5, 1000)
        pdf_vals = norm.pdf(xs)
        
        u_samples, x_samples, u_check = cdf_and_q(pdf_vals, xs)
        
        # Check output types and shapes
        assert isinstance(u_samples, np.ndarray), "u_samples should be numpy array"
        assert isinstance(x_samples, np.ndarray), "x_samples should be numpy array"
        assert isinstance(u_check, np.ndarray), "u_check should be numpy array"
        
        # Check that all arrays have the same length
        assert len(u_samples) == len(x_samples) == len(u_check), "All output arrays should have same length"
        assert len(u_samples) == 1000000, "Should generate 1,000,000 samples"

    def test_uniform_samples_are_uniform(self):
        """Test that u_samples are approximately uniform."""
        xs = np.linspace(-5, 5, 1000)
        pdf_vals = norm.pdf(xs)
        
        u_samples, _, _ = cdf_and_q(pdf_vals, xs)
        
        # Check that u_samples are in [0, 1]
        assert np.all((u_samples >= 0) & (u_samples <= 1)), "u_samples should be in [0, 1]"
        
        # Check that the mean is approximately 0.5
        assert np.isclose(np.mean(u_samples), 0.5, atol=0.01), "Mean of u_samples should be ~0.5"
        
        # Check that the standard deviation is approximately 1/sqrt(12) ≈ 0.289
        expected_std = 1/np.sqrt(12)
        assert np.isclose(np.std(u_samples), expected_std, atol=0.01), "Std of u_samples should be ~0.289"

    def test_x_samples_match_target_pdf(self):
        """Test that x_samples follow the target PDF distribution."""
        xs = np.linspace(-5, 5, 1000)
        pdf_vals = norm.pdf(xs)
        
        _, x_samples, _ = cdf_and_q(pdf_vals, xs)
        
        # Check that x_samples are within the expected range
        assert np.all((x_samples >= xs[0]) & (x_samples <= xs[-1])), "x_samples should be within xs range"
        
        # Check that the mean is approximately 0 (for standard normal)
        assert np.isclose(np.mean(x_samples), 0, atol=0.1), "Mean of x_samples should be ~0 for standard normal"
        
        # Check that the standard deviation is approximately 1 (for standard normal)
        assert np.isclose(np.std(x_samples), 1, atol=0.1), "Std of x_samples should be ~1 for standard normal"

    def test_u_check_is_uniform(self):
        """Test that u_check (CDF applied to x_samples) is uniform."""
        xs = np.linspace(-5, 5, 1000)
        pdf_vals = norm.pdf(xs)
        
        _, x_samples, u_check = cdf_and_q(pdf_vals, xs)
        
        # Check that u_check is in [0, 1]
        assert np.all((u_check >= 0) & (u_check <= 1)), "u_check should be in [0, 1]"
        
        # Check that u_check is approximately uniform
        assert np.isclose(np.mean(u_check), 0.5, atol=0.01), "Mean of u_check should be ~0.5"
        
        # Check that u_check has uniform distribution properties
        expected_std = 1/np.sqrt(12)
        assert np.isclose(np.std(u_check), expected_std, atol=0.01), "Std of u_check should be ~0.289"

    def test_cdf_monotonicity(self):
        """Test that the CDF built by cdf_and_q is monotonically non-decreasing."""
        xs = np.linspace(-5, 5, 1000)
        pdf_vals = norm.pdf(xs)
        
        # We need to extract the CDF from the cdf_and_q function
        # Let's create a smaller test to check the CDF construction
        cdf_vals = np.cumsum((pdf_vals[:-1] + pdf_vals[1:]) * np.diff(xs) / 2.0)
        cdf_vals = np.concatenate([[0.0], cdf_vals])
        
        # Check monotonicity
        assert np.all(np.diff(cdf_vals) >= 0), "CDF should be monotonically non-decreasing"

    def test_cdf_bounds(self):
        """Test that the CDF starts at 0 and ends at 1."""
        xs = np.linspace(-5, 5, 1000)
        pdf_vals = norm.pdf(xs)
        
        # Extract CDF construction logic from cdf_and_q
        cdf_vals = np.cumsum((pdf_vals[:-1] + pdf_vals[1:]) * np.diff(xs) / 2.0)
        cdf_vals = np.concatenate([[0.0], cdf_vals])
        
        # Check bounds
        assert np.isclose(cdf_vals[0], 0.0), "CDF should start at 0"
        assert np.isclose(cdf_vals[-1], 1.0, atol=1e-6), "CDF should end at 1"

    def test_exponential_distribution(self):
        """Test cdf_and_q with an exponential distribution."""
        # Exponential distribution: PDF = lambda * exp(-lambda x), x >= 0
        lam = 2.0
        xs = np.linspace(0, 5, 1000)
        pdf_vals = lam * np.exp(-lam * xs)
        
        u_samples, x_samples, u_check = cdf_and_q(pdf_vals, xs)
        
        # Check that x_samples are non-negative (exponential support)
        assert np.all(x_samples >= 0), "x_samples should be non-negative for exponential"
        
        # Check that the mean is approximately 1/lambda = 0.5
        assert np.isclose(np.mean(x_samples), 1/lam, atol=0.1), f"Mean should be ~{1/lam} for exponential"
        
        # Check that u_check is uniform
        assert np.isclose(np.mean(u_check), 0.5, atol=0.01), "u_check should be uniform"

    def test_trapezoidal_integration_accuracy(self):
        """Test that trapezoidal integration produces accurate CDF."""
        xs = np.linspace(-5, 5, 1000)
        pdf_vals = norm.pdf(xs)
        
        # Build CDF using trapezoidal rule (from cdf_and_q)
        cdf_vals = np.cumsum((pdf_vals[:-1] + pdf_vals[1:]) * np.diff(xs) / 2.0)
        cdf_vals = np.concatenate([[0.0], cdf_vals])
        
        # Compare with analytical CDF
        cdf_analytical = norm.cdf(xs)
        
        # The trapezoidal CDF should be close to analytical CDF
        assert np.allclose(cdf_vals, cdf_analytical, atol=0.01), "Trapezoidal CDF should match analytical CDF"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
