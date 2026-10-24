"""Run the complete, reproducible study with `python -m market_project.run`."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cournot import equilibrium, simulate, stability_eigenvalues
from .demand import generate_demand
from .forecast import evaluate_forecast


TRAIN_SIZE = 72
N_PERIODS = 96
SEEDS = (11, 23, 37)
SCENARIOS = {
    "baseline": {"trend": 0.18, "seasonality": 8.0, "volatility": 3.0},
    "high_trend": {"trend": 0.36, "seasonality": 8.0, "volatility": 3.0},
    "high_seasonality": {"trend": 0.18, "seasonality": 14.0, "volatility": 3.0},
    "high_volatility": {"trend": 0.18, "seasonality": 8.0, "volatility": 7.0},
}


def save_plot(path, draw):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    draw(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    demand = generate_demand(n=N_PERIODS, seed=7, **SCENARIOS["baseline"])
    result = evaluate_forecast(demand, TRAIN_SIZE)
    initial = equilibrium(demand[TRAIN_SIZE - 1]) * np.array([0.9, 1.0, 1.1])
    competition = pd.DataFrame(simulate(result["actual"], result["predicted"],
                                        initial_quantity=initial))
    competition.insert(0, "calendar_period", np.arange(TRAIN_SIZE, N_PERIODS))
    competition.to_csv(output_dir / "main_competition.csv", index=False)
    pd.DataFrame({"period": np.arange(N_PERIODS), "synthetic_A": demand}).to_csv(
        output_dir / "synthetic_demand.csv", index=False)
    pd.DataFrame({"period": np.arange(TRAIN_SIZE, N_PERIODS),
                  "actual_A": result["actual"], "sarimax_forecast": result["predicted"],
                  "seasonal_naive": result["baseline"], "lower_95": result["lower"],
                  "upper_95": result["upper"]}).to_csv(output_dir / "forecasts.csv", index=False)

    def demand_plot(ax):
        t = np.arange(N_PERIODS)
        ax.plot(t, demand, color="black", label="Synthetic A (actual)")
        ax.plot(t[TRAIN_SIZE:], result["predicted"], color="tab:blue", label="SARIMAX one step")
        ax.plot(t[TRAIN_SIZE:], result["baseline"], color="tab:orange", alpha=.8,
                label="Seasonal naive")
        ax.fill_between(t[TRAIN_SIZE:], result["lower"], result["upper"],
                        color="tab:blue", alpha=.13, label="95% forecast interval")
        ax.axvline(TRAIN_SIZE - .5, color="gray", linestyle="--", label="Train/test split")
        ax.set(xlabel="Month", ylabel="Inverse-demand intercept A", title="Demand and one-step forecasts")
        ax.legend(fontsize=8, ncol=2)
    save_plot(output_dir / "demand_forecast.png", demand_plot)

    def residual_plot(ax):
        ax.plot(result["train_residuals"], color="tab:purple")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(xlabel="Training residual index (first 12 excluded)", ylabel="Residual",
               title="SARIMAX training residuals")
    save_plot(output_dir / "training_residuals.png", residual_plot)

    fixed_A = 85.0
    fixed_initial = np.array([0.0, 80.0, 160.0])
    fixed = pd.DataFrame(simulate(np.full(80, fixed_A), np.full(80, fixed_A),
                                  initial_quantity=fixed_initial))
    fixed.to_csv(output_dir / "fixed_demand_convergence.csv", index=False)
    def fixed_plot(ax):
        for i in range(1, 4):
            ax.plot(fixed["period"], fixed[f"firm_{i}_quantity"], label=f"Firm {i}")
        ax.axhline(equilibrium(fixed_A), color="black", linestyle="--", label="Nash quantity")
        ax.set(xlabel="Update step", ylabel="Quantity per firm", title="Convergence under fixed demand")
        ax.legend()
    save_plot(output_dir / "fixed_convergence.png", fixed_plot)

    def moving_plot(ax):
        ax.plot(competition["calendar_period"], competition["total_quantity"],
                label="Adaptive total quantity")
        ax.plot(competition["calendar_period"], competition["equilibrium_total_quantity"],
                linestyle="--", label="Moving Nash total quantity")
        ax.set(xlabel="Month", ylabel="Total quantity", title="Tracking changing demand")
        ax.legend()
    save_plot(output_dir / "moving_equilibrium.png", moving_plot)

    experiment_rows = []
    stability = stability_eigenvalues()
    for scenario, params in SCENARIOS.items():
        for seed in SEEDS:
            path = generate_demand(n=N_PERIODS, seed=seed, **params)
            forecast = evaluate_forecast(path, TRAIN_SIZE)
            initial = equilibrium(path[TRAIN_SIZE - 1]) * np.array([0.9, 1.0, 1.1])
            game = pd.DataFrame(simulate(forecast["actual"], forecast["predicted"],
                                         initial_quantity=initial))
            experiment_rows.append({
                "scenario": scenario, "seed": seed, **params,
                "forecast_mae": forecast["diagnostics"]["sarimax"]["mae"],
                "baseline_mae": forecast["diagnostics"]["seasonal_naive"]["mae"],
                "mean_equilibrium_distance": game["distance_to_equilibrium"].mean(),
                "mean_total_profit": game[[f"firm_{i}_profit" for i in range(1, 4)]].sum(axis=1).mean(),
                "mean_price": game["price"].mean(),
                "fit_converged": forecast["diagnostics"]["converged"],
                "positive_equilibrium_each_period": bool(np.min(forecast["actual"]) > 12.0),
                "all_best_responses_interior": bool(game["best_responses_interior"].all()),
                "fixed_demand_common_eigenvalue": stability["common_mode"],
                "fixed_demand_difference_eigenvalue": stability["difference_mode"],
                "fixed_demand_locally_stable": stability["stable_interior"],
            })
    experiments = pd.DataFrame(experiment_rows)
    experiments.to_csv(output_dir / "experiments_by_seed.csv", index=False)
    summary = experiments.groupby("scenario", sort=False).agg(
        mean_tracking_distance=("mean_equilibrium_distance", "mean"),
        sd_tracking_distance=("mean_equilibrium_distance", "std"),
        mean_forecast_mae=("forecast_mae", "mean"),
        mean_total_profit=("mean_total_profit", "mean"),
        positive_equilibrium_each_period=("positive_equilibrium_each_period", "all"),
        all_best_responses_interior=("all_best_responses_interior", "all"),
        fixed_demand_locally_stable=("fixed_demand_locally_stable", "all"),
    ).reset_index()
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    baseline = experiments.loc[experiments["scenario"] == "baseline"].set_index("seed")
    contrasts = []
    for scenario in SCENARIOS:
        if scenario == "baseline":
            continue
        changed = experiments.loc[experiments["scenario"] == scenario].set_index("seed")
        for seed in SEEDS:
            contrasts.append({
                "scenario": scenario, "seed": seed,
                "tracking_distance_change": changed.loc[seed, "mean_equilibrium_distance"]
                                            - baseline.loc[seed, "mean_equilibrium_distance"],
                "forecast_mae_change": changed.loc[seed, "forecast_mae"]
                                       - baseline.loc[seed, "forecast_mae"],
                "total_profit_change": changed.loc[seed, "mean_total_profit"]
                                       - baseline.loc[seed, "mean_total_profit"],
            })
    pd.DataFrame(contrasts).to_csv(output_dir / "paired_changes_from_baseline.csv", index=False)

    def experiment_plot(ax):
        labels = summary["scenario"].str.replace("_", " ")
        ax.bar(labels, summary["mean_tracking_distance"],
               yerr=summary["sd_tracking_distance"], capsize=4, color="teal")
        ax.set(xlabel="One factor changed from baseline", ylabel="Mean Euclidean distance",
               title="Moving-equilibrium tracking (3 seeds; bars show SD)")
        ax.tick_params(axis="x", labelrotation=15)
    save_plot(output_dir / "experiment_tracking.png", experiment_plot)

    summary_json = {
        "data_label": "synthetic inverse-demand intercept, not observed sales",
        "generator": "A_t = 85 + trend*t + seasonality*sin(2*pi*t/12) + u_t; u_t = 0.35*u_(t-1) + innovation_t",
        "main_forecast": result["diagnostics"],
        "fixed_demand": {
            "A": fixed_A, "n_firms": 3, "b": 0.08, "c": 12.0, "learning_rate": 0.55,
            "theoretical_quantity_per_firm": equilibrium(fixed_A),
            "initial_distance": float(np.linalg.norm(fixed_initial - equilibrium(fixed_A))),
            "distance_after_first_update": float(fixed["distance_to_equilibrium"].iloc[0]),
            "final_distance": float(fixed["distance_to_equilibrium"].iloc[-1]),
            "linear_update_eigenvalues": stability_eigenvalues(),
        },
        "changing_demand": {
            "mean_tracking_distance": float(competition["distance_to_equilibrium"].mean()),
            "final_tracking_distance": float(competition["distance_to_equilibrium"].iloc[-1]),
        },
        "experiment_seeds": list(SEEDS), "scenarios": SCENARIOS,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(json.dumps(summary_json, indent=2))
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    main(args.output)
