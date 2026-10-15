"""Small checks for economics, update timing, and reproducibility."""

import numpy as np

from .cournot import best_response, equilibrium, simulate, stability_eigenvalues
from .demand import generate_demand


def main():
    a, b, c, n = 85.0, .08, 12.0, 3
    q_star = equilibrium(a, n, b, c)
    assert np.isclose(q_star, best_response(a, (n - 1) * q_star, b, c))
    assert generate_demand(seed=19).tolist() == generate_demand(seed=19).tolist()
    assert not np.array_equal(generate_demand(seed=19), generate_demand(seed=20))
    fixed = simulate(np.full(80, a), np.full(80, a), initial_quantity=[0, 80, 160])
    assert fixed[-1]["distance_to_equilibrium"] < 1e-6
    assert stability_eigenvalues()["stable_interior"]
    # Decisions at t use the forecast; realized A_t only changes price/profit.
    low = simulate([80.0], [85.0])[0]
    high = simulate([100.0], [85.0])[0]
    assert np.isclose(low["total_quantity"], high["total_quantity"])
    assert high["price"] > low["price"]
    print("All economic and reproducibility checks passed")


if __name__ == "__main__":
    main()
