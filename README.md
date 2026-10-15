# Dynamic Multi-Agent Competition Under Stochastic Demand and Adaptive Learning

A small, reproducible **synthetic-data** project. The generated series is the
inverse-demand **intercept** `A_t`, not observed sales. This lets us connect a
forecast directly to the market equation without treating sales as demand.

## Run (PowerShell, from this folder)

Use Python 3.11 or newer. Create an environment, install packages, and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m market_project.checks
.\.venv\Scripts\python.exe -m market_project.run
```

If `py` is unavailable, replace `py -3` with the path to an installed Python.
The entry point writes CSV, JSON, and PNG files to `results/`. Repeating the run
with the same software versions gives the same generated demand and experiments.

## The four objectives

**1. Demand and SARIMAX.** In `market_project/demand.py`, month `t` has

`A_t = 85 + trend*t + seasonality*sin(2*pi*t/12) + u_t`,
`u_t = 0.35*u_(t-1) + innovation_t`, with seeded Gaussian innovations.

`market_project/forecast.py` fits SARIMAX(1,0,0)x(0,0,0,12) with a constant
and known time/sine/cosine covariates. The first 72 months train the model;
the last 24 test it. Each test forecast is made **before** that month's actual
`A_t` is appended to the model state, with no refit. The simple baseline is
last year's value for the same month. Validation includes test MAE, RMSE,
95% interval coverage, model fit convergence, AIC, and training-residual
Ljung-Box and Jarque-Bera p-values. A p-value is evidence for checking a model,
not proof that it is correct. See `results/forecasts.csv`, `summary.json`,
`demand_forecast.png`, and `training_residuals.png`.

**2. Cournot competition.** In `market_project/cournot.py`, three firms choose
nonnegative quantities. Total output is `Q_t = sum_i q_(i,t)`, realized price
is `P_t = A_t - b*Q_t`, cost is `c*q_(i,t)`, and profit is
`pi_(i,t) = q_(i,t)*(P_t-c)`. We use `b=0.08`, `c=12`. A firm facing rivals'
output `Q_-i` has best response

`BR_i = max(0, (A_hat_t - c - b*Q_-i)/(2*b))`.

All firms update simultaneously from last period's quantities:

`q_(i,t) = (1-alpha)*q_(i,t-1) + alpha*BR_i(A_hat_t, Q_(-i,t-1))`,
with `alpha=0.55`. They use the forecast to decide, while **realized** `A_t`
determines price and profit. See `results/main_competition.csv`.

**3. Equilibrium and learning.** For constant `A>c` and identical costs, the
first-order condition is `A-c-b*Q-b*q_i=0`. At a symmetric Nash equilibrium,
`q_i*=(A-c)/(b*(n+1))`. The `max(0,...)` form handles low demand. For an
interior equilibrium, the damped update has common-mode eigenvalue
`1-alpha*(n+1)/2` and difference-mode eigenvalue `1-alpha/2`.
With three firms and `alpha=0.55`, both have magnitude below 1, so the fixed
interior equilibrium is locally stable. `results/fixed_convergence.png` and
`fixed_demand_convergence.csv` show numerical convergence from unequal initial
outputs (0, 80, and 160). For the changing-demand runs, firms start near the
last training period's equilibrium with 10% differences in their quantities.
Changing demand creates a **moving** equilibrium. We measure Euclidean
distance between the actual quantity vector and the Nash quantity vector
calculated from realized `A_t`; this is tracking error, not a fixed-game
convergence claim. See `results/moving_equilibrium.png` and `summary.json`.

**4. Controlled experiments.** `market_project/run.py` changes one generator
parameter at a time from baseline: trend `0.18 -> 0.36`, seasonality
`8 -> 14`, or innovation standard deviation `3 -> 7`. Each setting runs
seeds 11, 23, 37, with a fresh SARIMAX forecast and competition simulation.
See `results/experiments_by_seed.csv`, `experiment_summary.csv`,
`paired_changes_from_baseline.csv`, and `experiment_tracking.png`. The paired
changes compare each scenario to baseline using the same seed. Bar-chart error bars are standard deviations
across these three seeds, not confidence intervals. Compare observed numbers;
no direction of effect is assumed in code.

The experiment table also records the **fixed-demand stability** result for
each scenario. In this linear interior game, the update eigenvalues depend on
`n` and `alpha`, not on the level or process of `A_t`. Thus changing trend,
seasonality, or volatility does not change local fixed-demand stability under
the same update rule. It can still change **tracking error** when the target
equilibrium moves. The `all_periods_interior` field confirms that the interior
stability calculation applies to each simulated path.

## Files and practical limits

- `market_project/checks.py`: equilibrium best-response consistency, seeded
  reproducibility, fixed-demand convergence, and decision timing.
- `results/`: generated output; safe to recreate with the entry point.

The synthetic generator and the SARIMAX model share a known annual calendar,
which makes this a controlled demonstration. The experiment has only three
seeds per scenario and 24 test months per seed, so small differences should not
be presented as general statistical conclusions. All firms have identical
linear costs; capacity limits and entry are outside this model.

# Adaptive-Cournot-Stochastic-Demand
