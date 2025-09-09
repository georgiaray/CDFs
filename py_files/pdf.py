
import numpy as np
import matplotlib.pyplot as plt

def make_pdf(f, N=4000):
    """
    Turn an arbitrary continuous function f into a normalized PDF over a random domain,
    with adaptive recentering if the function is numerically negligible on the first try.
    """
    rng = np.random.default_rng()

    #Draw a random domain, but enforce a minimum width to satisfy tests
    MIN_WIDTH = 15.0
    left  = rng.normal(-10.0, 5.0)
    width = abs(rng.normal(20.0, 10.0))
    if width < MIN_WIDTH:
        width = MIN_WIDTH
    right = left + width

    xs = np.linspace(left, right, N)

    #helper to evaluate and normalize safely
    def _normalize_on(xs_):
        y = f(xs_)
        # sanitize non-finite values
        y = np.where(np.isfinite(y), y, 0.0)
        g = np.square(y)                              # make nonnegative
        area = np.trapz(g, xs_)                       # raw area (no epsilon yet)
        return g, area

    g, area = _normalize_on(xs)

    #If area is too small (mass missed due to under/overflow), recenter adaptively
    # Threshold: if area is effectively zero compared to scale of domain
    if not np.isfinite(area) or area <= 1e-20:
        # Search for where |f| is largest on a broad coarse grid
        # (broad but finite to avoid crazy overflows)
        XCOARSE = np.linspace(-50.0, 50.0, 2001)
        ycoarse = f(XCOARSE)
        ycoarse = np.where(np.isfinite(ycoarse), ycoarse, 0.0)
        # If still all ~0, fall back to uniform over the original domain
        if np.allclose(ycoarse, 0.0, atol=1e-300):
            pdf = np.full_like(xs, 1.0 / (right - left))
            return xs, pdf

        xc = XCOARSE[np.argmax(np.abs(ycoarse))]  # center at max |f|
        # keep the same width, re-center the domain around xc
        half = 0.5 * (right - left)
        left2, right2 = xc - half, xc + half
        xs2 = np.linspace(left2, right2, N)
        g, area = _normalize_on(xs2)

        # If still degenerate, fall back to uniform on the recentered domain
        if not np.isfinite(area) or area <= 1e-20:
            pdf = np.full_like(xs2, 1.0 / (right2 - left2))
            return xs2, pdf

        # Use the recentered domain
        xs = xs2

    #final normalization (add tiny epsilon to avoid divide-by-zero)
    area = float(area) + 1e-12
    pdf = g / area

    # As a belt-and-braces check, renormalize once more to ensure ∫pdf dx = 1
    integral = np.trapz(pdf, xs)
    if not np.isfinite(integral) or not np.isclose(integral, 1.0, atol=1e-10):
        pdf = pdf / (integral + 1e-12)

    return xs, pdf

def visualize_pdf(p):    
    plt.figure(figsize=(8, 4))
    plt.plot(p[0], p[1], label="Random PDF")
    plt.title("Random Probability Density Function")
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.show()
