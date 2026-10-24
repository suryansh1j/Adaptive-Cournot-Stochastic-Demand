"""Three or more symmetric-cost Cournot firms with damped best responses."""

import numpy as np


def equilibrium(intercept, n_firms=3, slope=0.08, marginal_cost=12.0):
    """Interior symmetric Nash quantity per firm: (A-c)/(b*(n+1))."""
    return max(0.0, (intercept - marginal_cost) / (slope * (n_firms + 1)))


def best_response(intercept, rivals_quantity, slope=0.08, marginal_cost=12.0):
    return max(0.0, (intercept - marginal_cost - slope * rivals_quantity) / (2 * slope))


def simulate(actual_intercepts, decision_intercepts, n_firms=3, slope=0.08,
             marginal_cost=12.0, learning_rate=0.55, initial_quantity=0.0):
    """At t, firms use forecast A_hat_t and rivals' quantities from t-1."""
    actual = np.asarray(actual_intercepts, dtype=float)
    decision = np.asarray(decision_intercepts, dtype=float)
    if actual.shape != decision.shape or actual.ndim != 1:
        raise ValueError("Actual and decision intercepts must be equal-length vectors")
    if n_firms < 3 or slope <= 0 or not (0 < learning_rate <= 1):
        raise ValueError("Need >=3 firms, positive slope, and 0 < learning_rate <= 1")
    initial = np.asarray(initial_quantity, dtype=float)
    old = (np.full(n_firms, float(initial)) if initial.ndim == 0 else initial.copy())
    if old.shape != (n_firms,) or np.any(old < 0):
        raise ValueError("Initial quantity must be nonnegative scalar or one value per firm")
    rows = []
    for t, (a, a_hat) in enumerate(zip(actual, decision)):
        unconstrained = (a_hat - marginal_cost - slope * (old.sum() - old)) / (2 * slope)
        response = np.maximum(0.0, unconstrained)
        q = (1 - learning_rate) * old + learning_rate * response
        total = q.sum()
        price = a - slope * total
        profits = q * (price - marginal_cost)
        q_star = equilibrium(a, n_firms, slope, marginal_cost)
        rows.append({"period": t, "actual_A": a, "decision_A": a_hat,
                     "total_quantity": total, "price": price,
                     "equilibrium_quantity_per_firm": q_star,
                     "equilibrium_total_quantity": n_firms * q_star,
                     "distance_to_equilibrium": float(np.linalg.norm(q - q_star)),
                     "best_responses_interior": bool(np.all(unconstrained > 0)),
                     **{f"firm_{i + 1}_quantity": q[i] for i in range(n_firms)},
                     **{f"firm_{i + 1}_profit": profits[i] for i in range(n_firms)}})
        old = q
    return rows


def stability_eigenvalues(n_firms=3, learning_rate=0.55):
    """Local interior update eigenvalues for common and difference modes."""
    return {"common_mode": 1 - learning_rate * (n_firms + 1) / 2,
            "difference_mode": 1 - learning_rate / 2,
            "stable_interior": 0 < learning_rate < 4 / (n_firms + 1)}
