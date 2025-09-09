import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm, uniform, beta
import matplotlib.pyplot as plt
from pdf import make_pdf

class TestPDFGenerator:
    """Test suite for the PDF generator to ensure it produces valid probability density functions."""
    
    def test_pdf_non_negativity(self):
        """Test that f(x) ≥ 0 for all x ∈ ℝ (non-negativity condition)."""
        # Test with various input functions
        test_functions = [
            lambda x: np.sin(x),  # Can be negative, but squared should be non-negative
            lambda x: np.cos(x),  # Can be negative, but squared should be non-negative
            lambda x: x**2,       # Always non-negative
            lambda x: np.exp(-x**2),  # Always positive
            lambda x: 1 / (1 + x**2),  # Always positive
            lambda x: np.abs(x),  # Always non-negative
        ]
        
        for func in test_functions:
            xs, pdf = make_pdf(func, N=1000)
            assert np.all(pdf >= 0), f"PDF should be non-negative for function {func.__name__}"
            assert not np.any(np.isnan(pdf)), f"PDF should not contain NaN values for function {func.__name__}"
            assert not np.any(np.isinf(pdf)), f"PDF should not contain infinite values for function {func.__name__}"
    
    def test_pdf_piecewise_continuity(self):
        """Test that f is piecewise continuous by checking for reasonable smoothness."""
        test_functions = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: x**2,
            lambda x: np.exp(-x**2),
        ]
        
        for func in test_functions:
            xs, pdf = make_pdf(func, N=1000)
            
            # Check that there are no sudden jumps (discontinuities)
            # Compute the gradient and check it's not too large
            pdf_gradient = np.gradient(pdf, xs)
            max_gradient = np.max(np.abs(pdf_gradient))
            
            # The gradient shouldn't be extremely large (indicating discontinuities)
            # This is a heuristic test - adjust threshold as needed
            assert max_gradient < 1000, f"PDF should be reasonably smooth for function {func.__name__}"
            
            # Check that there are no NaN or infinite values
            assert not np.any(np.isnan(pdf)), f"PDF should be continuous (no NaN) for function {func.__name__}"
            assert not np.any(np.isinf(pdf)), f"PDF should be continuous (no inf) for function {func.__name__}"
    
    def test_pdf_normalization(self):
        """Test that ∫₋∞^∞ f(x)dx = 1 (normalization condition)."""
        test_functions = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: x**2,
            lambda x: np.exp(-x**2),
            lambda x: 1 / (1 + x**2),
            lambda x: np.abs(x),
        ]
        
        for func in test_functions:
            xs, pdf = make_pdf(func, N=2000)  # Use more points for better integration accuracy
            
            # Numerical integration using trapezoidal rule
            integral = np.trapz(pdf, xs)
            
            # The integral should be close to 1
            assert np.isclose(integral, 1.0, atol=1e-6), f"PDF should integrate to 1, got {integral} for function {func.__name__}"
    
    def test_pdf_probability_integration(self):
        """Test that P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx for various intervals."""
        test_functions = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: np.exp(-x**2),
        ]
        
        for func in test_functions:
            xs, pdf = make_pdf(func, N=2000)
            
            # Test multiple intervals
            intervals = [
                (xs[0], xs[-1]),  # Full domain
                (xs[len(xs)//4], xs[3*len(xs)//4]),  # Middle half
                (xs[len(xs)//3], xs[2*len(xs)//3]),  # Middle third
            ]
            
            for a, b in intervals:
                # Find indices corresponding to a and b
                idx_a = np.argmin(np.abs(xs - a))
                idx_b = np.argmin(np.abs(xs - b))
                
                # Ensure a < b
                if idx_a > idx_b:
                    idx_a, idx_b = idx_b, idx_a
                
                # Extract the interval
                interval_xs = xs[idx_a:idx_b+1]
                interval_pdf = pdf[idx_a:idx_b+1]
                
                # Compute probability using integration
                probability = np.trapz(interval_pdf, interval_xs)
                
                # Probability should be between 0 and 1
                assert 0 <= probability <= 1, f"Probability should be in [0,1], got {probability}"
                
                # Probability should be positive for non-empty intervals
                if len(interval_xs) > 1:
                    assert probability > 0, f"Probability should be positive for non-empty interval"
    
    def test_pdf_domain_consistency(self):
        """Test that the PDF domain is consistent and reasonable."""
        test_functions = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: x**2,
        ]
        
        for func in test_functions:
            xs, pdf = make_pdf(func, N=1000)
            
            # Check that xs and pdf have the same length
            assert len(xs) == len(pdf), "Domain and PDF should have the same length"
            
            # Check that xs is sorted (domain should be ordered)
            assert np.all(np.diff(xs) > 0), "Domain should be strictly increasing"
            
            # Check that domain is reasonable (not too extreme)
            domain_width = xs[-1] - xs[0]
            assert 0.1 < domain_width < 1000, f"Domain width should be reasonable, got {domain_width}"
    
    def test_pdf_with_known_distributions(self):
        """Test PDF generation with functions that approximate known distributions."""
        # Test with Gaussian-like function
        def gaussian_like(x):
            return np.exp(-x**2 / 2)
        
        xs, pdf = make_pdf(gaussian_like, N=2000)
        
        # Check basic properties
        assert np.all(pdf >= 0), "Gaussian-like PDF should be non-negative"
        assert np.isclose(np.trapz(pdf, xs), 1.0, atol=1e-6), "Gaussian-like PDF should be normalized"
        
        # Check that it's roughly symmetric (if domain is symmetric)
        if abs(xs[0] + xs[-1]) < 1e-6:  # Domain is roughly symmetric
            mid_idx = len(xs) // 2
            left_half = pdf[:mid_idx]
            right_half = pdf[mid_idx:][::-1]  # Reverse to compare
            min_len = min(len(left_half), len(right_half))
            if min_len > 0:
                symmetry_error = np.mean(np.abs(left_half[:min_len] - right_half[:min_len]))
                assert symmetry_error < 0.1, f"Gaussian-like PDF should be roughly symmetric, error: {symmetry_error}"
    
    def test_pdf_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with constant function
        def constant_func(x):
            return np.ones_like(x)
        
        xs, pdf = make_pdf(constant_func, N=100)
        assert np.all(pdf >= 0), "Constant function PDF should be non-negative"
        assert np.isclose(np.trapz(pdf, xs), 1.0, atol=1e-6), "Constant function PDF should be normalized"
        
        # Test with very small domain
        def small_domain_func(x):
            return np.ones_like(x)
        
        xs, pdf = make_pdf(small_domain_func, N=10)
        assert len(xs) == len(pdf), "Small domain should still produce consistent results"
        assert np.all(pdf >= 0), "Small domain PDF should be non-negative"
    
    def test_pdf_numerical_stability(self):
        """Test numerical stability with various function types."""
        # Test with functions that could cause numerical issues
        test_functions = [
            lambda x: np.exp(x),  # Can grow very large
            lambda x: np.exp(-x**2),  # Can become very small
            lambda x: 1 / (1 + x**2),  # Well-behaved
            lambda x: np.sin(x) * np.cos(x),  # Oscillatory
        ]
        
        for func in test_functions:
            xs, pdf = make_pdf(func, N=1000)
            
            # Check for numerical issues
            assert not np.any(np.isnan(pdf)), f"PDF should not contain NaN values for {func.__name__}"
            assert not np.any(np.isinf(pdf)), f"PDF should not contain infinite values for {func.__name__}"
            assert np.all(np.isfinite(pdf)), f"PDF should contain only finite values for {func.__name__}"
            
            # Check that PDF values are reasonable (not too large or too small)
            assert np.all(pdf >= 0), f"PDF should be non-negative for {func.__name__}"
            assert np.max(pdf) < 1e6, f"PDF values should not be extremely large for {func.__name__}"
            assert np.min(pdf) >= 0, f"PDF values should not be negative for {func.__name__}"
    
    def test_pdf_multiple_runs_consistency(self):
        """Test that multiple runs produce consistent results (within expected variance)."""
        def test_func(x):
            return np.exp(-x**2)
        
        # Run multiple times and check consistency
        results = []
        for _ in range(200):
            xs, pdf = make_pdf(test_func, N=1000)
            integral = np.trapz(pdf, xs)
            results.append(integral)
        
        # All integrals should be close to 1
        for integral in results:
            assert np.isclose(integral, 1.0, atol=1e-6), f"All runs should produce normalized PDFs, got {integral}"
        
        # Results should be reasonably consistent (small variance due to random domain)
        results_array = np.array(results)
        std_dev = np.std(results_array)
        assert std_dev < 0.1, f"Results should be reasonably consistent, std dev: {std_dev}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
