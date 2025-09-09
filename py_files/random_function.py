# random_function.py
'''The only requirement for CDFs to work for mapping to a perfect uniform distribution is that the underlying probability distribution must be continuous. So, in this notebook, I will build and test a random function generator, ensuring it is continuous.

We want to vary the: 
1. Location (mean / shift) – e.g. move the distribution left or right.
2. Scale (variance / spread) – make it wider or narrower.
3. Skewness – symmetry vs lopsided (normal vs lognormal).
4. Tail heaviness (kurtosis) – light tails vs heavy tails (normal vs t-distribution).
5. Support bounds – some distributions are on (-∞, ∞), some on [0, ∞), some on a finite interval 
6. Multimodality – single peak vs multiple peaks.
7. Underlying distribution (gaussian, polynomial, etc.)'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.special import erfc, erf

def _pos():
    # positive random scale (fully random, no fixed bounds)
    return np.random.lognormal(mean=0.0, sigma=np.random.lognormal(0.0, 0.5))

def _n_terms():
    # >=1 random number of terms
    return 1 + np.random.poisson(lam=np.random.lognormal(0.0, 0.5))

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# skewed Gaussian bump: phi(z) * Phi(alpha z)
def _skew_bump(x, c, w, alpha):
    z = (x - c) / (w + 1e-12)
    # standard normal pdf and cdf
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + erf(alpha * z / np.sqrt(2.0)))
    return pdf * cdf

# exGaussian bump: convolution of Normal(mean=c, sd=w) with Exp(rate=lambda>0)
def _exgauss(x, c, w, lam):
    # closed form using erfc; numerically stable-ish for moderate values
    # exGaussian pdf (scaled bump)
    z = (x - c)
    term = lam * np.exp(0.5 * (lam * w)**2 - lam * z)
    arg = (lam * w**2 - z) / (np.sqrt(2) * w)
    return term * 0.5 * erfc(arg)

def create_random_function():
    """
    Returns a continuous callable f(x) with fully random family & parameters.
    Covers:
      1) Location/shift (via global affine input transform)
      2) Scale/spread (via parameters and input scaling)
      3) Skewness (via skewed bumps and exGaussian components)
      4) Tail heaviness (families include heavy/very light tails)
      5) Support bounds (optional smooth windowing: finite or half-infinite)
      6) Multimodality (mixtures / sums)
    The callable is vectorized over NumPy arrays. No hard-coded bounds.
    """
    families = [
        "sin_sum", "poly", "gauss_mix", "exp_sum",
        "logistic_sum", "tanh_rbf", "skew_bump_mix", "exgauss_mix"
    ]
    kind = np.random.choice(families)

    # ----- global affine input transform: x' = s_x * x + mu -----
    s_x = np.random.lognormal(0.0, _pos())     # positive input scale
    mu  = np.random.normal(0.0, _pos())        # location shift

    # ----- random continuous support window (optional) -----
    # With prob ~0.5, apply a window. Type picked at random:
    #  - finite: roughly [a, b] via smooth sigmoids
    #  - half-infinite: roughly [a, ∞) or (-∞, b] via one sigmoid
    use_window = np.random.rand() < 0.5
    window_type = None
    a = b = k1 = k2 = None
    if use_window:
        window_type = np.random.choice(["finite", "right", "left"])
        # gates centers from Normal, slopes positive LogNormal (no fixed bounds)
        if window_type == "finite":
            a_raw = np.random.normal(0.0, _pos())
            b_raw = np.random.normal(0.0, _pos())
            a, b = (a_raw, b_raw) if a_raw < b_raw else (b_raw, a_raw)
            k1 = np.random.lognormal(0.0, _pos())
            k2 = np.random.lognormal(0.0, _pos())
        else:
            # one-sided gate
            a = np.random.normal(0.0, _pos())
            k1 = np.random.lognormal(0.0, _pos())

    # ----- build core function g(x') by family -----
    if kind == "sin_sum":
        m = _n_terms()
        s_a, s_w, s_phi = _pos(), _pos(), _pos()
        A   = np.random.normal(0.0, s_a,   m)
        W   = np.abs(np.random.standard_cauchy(m)) * s_w  # heavy-tailed freq
        Phi = np.random.normal(0.0, s_phi, m)
        def g(xp):
            return sum(a * np.sin(w * xp + p) for a, w, p in zip(A, W, Phi))

    elif kind == "poly":
        deg = 1 + np.random.poisson(lam=np.random.lognormal(0.0, 0.5))
        s_c = _pos()
        coeffs = np.random.normal(0.0, s_c, deg + 1)
        def g(xp):
            y = 0.0
            for c in coeffs[::-1]:
                y = y * xp + c
            return y

    elif kind == "gauss_mix":
        m = _n_terms()
        s_c, s_w, s_h = _pos(), _pos(), _pos()
        C = np.random.normal(0.0, s_c, m)
        W = np.random.lognormal(0.0, _pos(), m)  # positive widths
        H = np.random.normal(0.0, s_h, m)
        def g(xp):
            return sum(h * np.exp(-0.5 * ((xp - c) / w)**2) for h, c, w in zip(H, C, W))

    elif kind == "exp_sum":
        m = _n_terms()
        s_a, s_b = _pos(), _pos()
        A = np.random.normal(0.0, s_a, m)
        B = np.random.normal(0.0, s_b, m)  # growth/decay
        def g(xp):
            return sum(a * np.exp(b * xp) for a, b in zip(A, B))

    elif kind == "logistic_sum":
        m = _n_terms()
        L  = np.random.lognormal(0.0, _pos(), m)  # max levels
        k  = np.random.lognormal(0.0, _pos(), m)  # steepness
        x0 = np.random.normal(0.0, _pos(), m)     # midpoints
        def g(xp):
            return sum(Li / (1.0 + np.exp(-ki * (xp - x0i))) for Li, ki, x0i in zip(L, k, x0))

    elif kind == "tanh_rbf":
        m = _n_terms()
        s_a, s_b, s_c = _pos(), _pos(), _pos()
        A = np.random.normal(0.0, s_a, m)
        B = np.random.normal(0.0, s_b, m)
        C = np.random.normal(0.0, s_c, m)
        def g(xp):
            return sum(a * np.tanh(b * xp + c) for a, b, c in zip(A, B, C))

    elif kind == "skew_bump_mix":
        m = _n_terms()
        s_c, s_w, s_h, s_alpha = _pos(), _pos(), _pos(), _pos()
        C = np.random.normal(0.0, s_c, m)
        W = np.random.lognormal(0.0, _pos(), m)
        H = np.random.normal(0.0, s_h, m)
        ALPHA = np.random.normal(0.0, s_alpha, m)  # controls skew sign/magnitude
        def g(xp):
            return sum(h * _skew_bump(xp, c, w, a) for h, c, w, a in zip(H, C, W, ALPHA))

    elif kind == "exgauss_mix":
        m = _n_terms()
        s_c, s_w, s_h, s_lam = _pos(), _pos(), _pos(), _pos()
        C = np.random.normal(0.0, s_c, m)
        W = np.random.lognormal(0.0, _pos(), m)              # sd
        H = np.random.normal(0.0, s_h, m)                    # amplitudes
        LAM = np.random.lognormal(0.0, _pos(), m) + 1e-12    # positive rates
        def g(xp):
            return sum(h * _exgauss(xp, c, w, l) for h, c, w, l in zip(H, C, W, LAM))

    # ----- assemble final callable f(x): apply affine + optional window -----
    def f(x):
        x = np.asarray(x)
        xp = s_x * x + mu  # global affine input
        y = g(xp)
        if use_window:
            if window_type == "finite":
                # smooth gate to roughly [a, b]
                y = y * _sigmoid(k1 * (x - a)) * _sigmoid(k2 * (b - x))
            elif window_type == "right":
                # roughly [a, ∞)
                y = y * _sigmoid(k1 * (x - a))
            else:
                # roughly (-∞, a]
                y = y * _sigmoid(k1 * (a - x))
        return y

    # attach metadata for inspection
    f.kind = kind
    f.meta = {
        "kind": kind,
        "input_scale": s_x,
        "input_shift": mu,
        "window": {
            "enabled": use_window,
            "type": window_type,
            "a": a, "b": b, "k1": k1, "k2": k2
        }
    }
    return f


def visualize_random_function(f_x):
    # Define a range of x values for visualization
    x_vals = np.linspace(-10, 10, 500)
    y_vals = f_x(x_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f"Random {f_x.kind} function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Visualization of a Random Function")
    plt.legend()
    plt.grid(True)
    plt.show()
