#cdf.py
'''This file contains the functions to build the CDF and the inverse CDF of a given pdf. It also contains the functions to sample from the pdf and the inverse CDF.'''

import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

def cdf_and_q(pdf, xs, rng = None):
    rng = np.random.default_rng(rng)
    # build CDF values
    cdf_vals = np.cumsum((pdf[:-1] + pdf[1:]) * np.diff(xs) / 2.0) #note that the pdf and xs we are using are from the original pdf as defined just above this cell 
    #performing the trapezoidal integration per interval  
    cdf_vals = np.concatenate([[0.0], cdf_vals]) #accumulates those values from the starting point (in this case, it's not possible to have a true negative infinity value so that's why we are anchoring the CDF at 0 at the leftmost point of xs, effectively treating xs[0] as our numerical -∞; this is okay from a probability perspective because we are not integrating over the entire real line, we are only integrating over the support of the pdf)
    cdf_vals /= cdf_vals[-1]

    # enforce strict monotonicity for inversion
    cdf_vals = np.maximum.accumulate(cdf_vals)
    eps = 1e-15
    cdf_vals = np.clip(cdf_vals + eps*np.arange(len(cdf_vals)), 0.0, 1.0)

    F = PchipInterpolator(xs, cdf_vals, extrapolate=True)
    #here we are just making this a continuous function that can be evaluated at any point in the domain of xs by interpolating the values of cdf_vals at the points of xsß

    # inverse CDF (quantile function)
    Q = PchipInterpolator(cdf_vals, xs, extrapolate=True)
    #here we are making the inverse of the CDF, so that we can map from the uniform distribution to the original pdfß

    # generate samples
    u_samples = rng.random(1000000)
    u_samples = np.clip(u_samples, 1e-12, 1 - 1e-12)           # avoid edges
    x_samples = Q(u_samples)           

    # apply CDF transform back to uniform
    u_check = F(x_samples)
    return u_samples, x_samples, u_check

def plot_cdf_transform(u_samples, x_samples, u_check, xs, pdf):
    plt.figure(figsize=(18, 4))

    # Use a nicer color palette
    hist_color1 = "#4E79A7"  # blue
    hist_color2 = "#F28E2B"  # orange
    hist_color3 = "#59A14F"  # green
    pdf_color = "#E15759"    # red
    uniform_line_color = "#B07AA1"  # purple

    # Plot u_samples (should be uniform)
    plt.subplot(1, 3, 1)
    plt.hist(u_samples, bins=30, density=True, alpha=0.8, color=hist_color1, edgecolor='white')
    plt.axhline(1, color=uniform_line_color, lw=2, linestyle="--", label="Uniform(0,1)")
    plt.title("u_samples (Uniform; randomly sampled from the uniform distribution)")
    plt.legend()

    # Plot x_samples (should match target PDF)
    plt.subplot(1, 3, 2)
    plt.hist(x_samples, bins=30, density=True, alpha=0.8, color=hist_color2, edgecolor='white', label="x_samples")
    plt.plot(xs, pdf, color=pdf_color, lw=2, label="Target PDF")
    plt.title("x_samples (Mapped to PDF via the inverse CDF)")
    plt.legend()

    # Plot u_check (should be uniform)
    plt.subplot(1, 3, 3)
    plt.hist(u_check, bins=30, density=True, alpha=0.8, color=hist_color3, edgecolor='white', label="u_check")
    plt.axhline(1, color=uniform_line_color, lw=2, linestyle="--", label="Uniform(0,1)")
    plt.title("u_check (Transformed Back)")
    plt.legend()

    plt.tight_layout()
    plt.show()


