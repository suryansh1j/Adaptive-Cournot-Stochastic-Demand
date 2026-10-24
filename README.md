# Forecasting Demand and Firm Competition

**Current stage (17 October 2026):** a reproducible simulation using synthetic demand. The [project page](site/index.html) presents this stage on a simple website.

This project simulates **three firms selling the same product**. They compete by choosing **how much to produce** each month. When they produce more in total, the market price falls.

## How it works

1. The project creates 96 months of artificial demand. Demand generally rises, follows a yearly pattern, and has random shocks. It is **not real sales data**.
2. A forecasting model learns from the first 72 months and predicts demand one month at a time for the next 24.
3. Each firm uses that forecast and the **other firms' previous production** to choose its own quantity. It moves 55% of the way toward its preferred quantity each month.
4. Actual demand and the firms' combined production set the price. The project calculates each firm's profit and compares its production with the theoretical equilibrium, where no firm would benefit from changing its quantity alone.

The project also checks whether the forecast beats a simple “same month last year” prediction, whether production settles down when demand is fixed, and what happens with faster growth, stronger seasons, or more random demand. Charts and tables are saved in [`results/`](results/).

## What this stage shows

In the main 24-month test, the forecast's average absolute error was **1.65**, versus **3.72** for the same-month-last-year prediction. With demand held constant, the firms' quantities converge toward the theoretical equilibrium. Across three simulated runs per scenario, more volatile demand increased the average distance from the changing equilibrium from **15.66** to **32.94**.

These findings describe the simulation's assumptions and generated data; they are not estimates from a real market.

The [literature review and next experiments](LITERATURE_REVIEW.md) explain how the current model relates to published work on Cournot adjustment, uncertain demand, and risk aversion. Its stability calculation concerns a fixed-demand market; the changing-demand equilibrium plotted here is a realized-demand benchmark.

## Run it

```powershell
python -m pip install -r requirements.txt
python -m market_project.checks
python -m market_project.run
```

The last command regenerates `results/`. The committed results and [`site/index.html`](site/index.html) show the October 17 snapshot; numerical results can differ slightly across supported package versions. Refresh the site's written findings after regenerating the results.
