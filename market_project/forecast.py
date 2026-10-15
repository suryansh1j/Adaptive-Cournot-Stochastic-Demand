"""Time-ordered SARIMAX validation and one-step forecasts."""

import numpy as np
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .demand import features


def error_metrics(actual, predicted):
    error = np.asarray(actual) - np.asarray(predicted)
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2)))}


def evaluate_forecast(intercept, train_size=72):
    """Fit on the past; reveal each test value only after predicting that period."""
    y = np.asarray(intercept, dtype=float)
    if not (24 <= train_size <= len(y) - 12):
        raise ValueError("Need at least 24 training and 12 test observations")
    x = features(len(y))
    model = SARIMAX(y[:train_size], exog=x[:train_size], order=(1, 0, 0),
                    seasonal_order=(0, 0, 0, 12), trend="c",
                    enforce_stationarity=True, enforce_invertibility=True)
    fitted = model.fit(disp=False, maxiter=150)

    # Initial states can distort early residuals, so exclude the first season.
    residuals = np.asarray(fitted.resid)[12:]
    lb = acorr_ljungbox(residuals, lags=[12], return_df=True)
    normality = jarque_bera(residuals)
    predictions, lower, upper = [], [], []
    current = fitted
    for t in range(train_size, len(y)):
        forecast = current.get_forecast(steps=1, exog=x[t:t + 1])
        predictions.append(float(forecast.predicted_mean[0]))
        interval = np.asarray(forecast.conf_int(alpha=0.05))[0]
        lower.append(float(interval[0]))
        upper.append(float(interval[1]))
        current = current.append(y[t:t + 1], exog=x[t:t + 1], refit=False)

    actual = y[train_size:]
    pred = np.asarray(predictions)
    baseline = y[train_size - 12:len(y) - 12]
    diagnostics = {
        "train_size": train_size, "test_size": len(actual),
        "sarimax_order": [1, 0, 0], "seasonal_order": [0, 0, 0, 12],
        "exog": ["time_in_years", "annual_sine", "annual_cosine"],
        "converged": bool(fitted.mle_retvals.get("converged", False)),
        "aic": float(fitted.aic),
        "ljung_box_lag12_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "jarque_bera_pvalue": float(normality.pvalue),
        "test_95pct_interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "sarimax": error_metrics(actual, pred),
        "seasonal_naive": error_metrics(actual, baseline),
    }
    return {"actual": actual, "predicted": pred, "baseline": baseline,
            "lower": np.asarray(lower), "upper": np.asarray(upper),
            "train_residuals": residuals, "diagnostics": diagnostics}
