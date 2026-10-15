"""Generate the inverse-demand intercept, not observed sales."""

import numpy as np


def generate_demand(n=96, seed=7, trend=0.18, seasonality=8.0, volatility=3.0,
                    base=85.0, shock_persistence=0.35):
    """A_t = base + trend*t + seasonality*sin(2*pi*t/12) + AR(1) shock."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    innovations = rng.normal(0.0, volatility, n)
    shocks = np.zeros(n)
    for k in range(n):
        shocks[k] = innovations[k] + (shock_persistence * shocks[k - 1] if k else 0.0)
    intercept = base + trend * t + seasonality * np.sin(2 * np.pi * t / 12) + shocks
    return intercept


def features(n):
    """Known calendar and time features; no realized future demand enters here."""
    t = np.arange(n)
    return np.column_stack((t / 12.0, np.sin(2 * np.pi * t / 12),
                            np.cos(2 * np.pi * t / 12)))
