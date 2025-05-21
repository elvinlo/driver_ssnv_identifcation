import numpy as np, pandas as pd
from scipy.stats import t, gamma, cauchy, weibull_min, poisson, nbinom

def generate_driver_mutation_data(
    n: int = 2000,
    K: int = 10,
    p: int = 25,
    decay: float = 0.8,
    random_state: int | None = 42,
):
    """
    Simulate an ordinal-class dataset that mimics driver-mutation recurrence levels.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    K : int
        Number of ordinal classes (recurrence levels 1..K).
    p : int
        Feature dimension (≈25 by default).
    decay : float
        Class-frequency decay rate.  Class k appears with probability
        proportional to `decay**(k-1)`.
    random_state : int | None
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n, p)
        Feature matrix.
    y : ndarray of shape (n,)
        Ordinal class labels taking values in {1, …, K}.
    """
    rng = np.random.default_rng(random_state)

    # --- Draw class labels with exponentially decaying frequencies ---
    class_probs = np.array([decay ** (k - 1) for k in range(1, K + 1)], dtype=float)
    class_probs /= class_probs.sum()
    y = rng.choice(np.arange(1, K + 1), size=n, p=class_probs)

    # --- Two shared latent factors induce correlation inside feature groups ---
    Z1 = rng.normal(0.0, 1.0, size=n)        # “global severity’’ factor
    Z2 = rng.normal(0.0, 1.0, size=n)        # extra noise / batch effect

    X = np.empty((n, p))

    # --------- Group A (features 0‑9): Gaussian, roughly linear in class ----------
    for j in range(10):
        mu = 0.30 * y + 0.50 * Z1                # monotone but noisy
        X[:, j] = rng.normal(loc=mu, scale=1.0)

    # --------- Group B (features 10‑17): mixture of Gaussians ----------
    # Each feature is drawn from a bimodal mixture; the mixing weight
    # changes non‑monotonically with the class, creating “confusable’’ modes.
    for j in range(10, 18):
        mu_hi = 0.20 * y             # main mode increases with class
        mu_lo = mu_hi - 1.2          # second mode sits ~1.2 units lower
        weight_hi = 0.50 + 0.30 * np.sin(y / 2.0)   # oscillating weights
        choose_hi = rng.uniform(size=n) < weight_hi
        X[:, j] = (
            rng.normal(mu_hi, 0.8) * choose_hi
            + rng.normal(mu_lo, 0.8) * (~choose_hi)
            + 0.40 * Z2                 # share latent factor inside group
        )

    # --------- Group C (features 18‑24): heavy‑tailed (Cauchy) ----------
    for j in range(18, 25):
        loc = 0.25 * y                # location still moves with class
        scale = 1.0 + 0.10 * (j - 18) # slightly different scales
        X[:, j] = rng.standard_cauchy(size=n) * scale + loc + 0.30 * Z1

    return X, y