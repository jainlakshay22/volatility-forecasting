"""Check the saved baseline validation forecasts and scores."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
SPLIT_FOLDER = PROJECT_FOLDER / "data/splits"
RESULTS_FOLDER = PROJECT_FOLDER / "results/baselines"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
MODELS = [
    "Persistence",
    "Five-day average",
    "Historical mean",
    "Linear HAR",
]


def calculate_scores(actual_log, forecast_log):
    """Recalculate the three validation measurements."""

    difference = actual_log - forecast_log
    log_mse = np.mean(difference ** 2)
    log_mae = np.mean(np.abs(difference))

    ratio = np.exp(actual_log) / np.exp(forecast_log)
    qlike = np.mean(ratio - np.log(ratio) - 1)

    return log_mse, log_mae, qlike


def main():
    """Verify dates, predictions, metrics, and test-data isolation."""

    forecast_path = RESULTS_FOLDER / "validation_forecasts.csv"
    score_path = RESULTS_FOLDER / "validation_scores.csv"
    coefficient_path = RESULTS_FOLDER / "linear_har_coefficients.csv"

    forecasts = pd.read_csv(forecast_path)
    scores = pd.read_csv(score_path)
    coefficients = pd.read_csv(coefficient_path)

    forecasts["date"] = pd.to_datetime(forecasts["date"])

    if forecasts.isna().any().any():
        raise ValueError("A validation forecast value is missing")

    if not np.isfinite(forecasts.select_dtypes(include="number")).all().all():
        raise ValueError("A validation forecast is not finite")

    if (forecasts["forecast_variance"] <= 0).any():
        raise ValueError("A variance forecast is not positive")

    # This stage is validation-only. No 2025 target date may appear.
    if (forecasts["date"] >= pd.Timestamp("2025-01-01")).any():
        raise ValueError("Test-period data appears in validation forecasts")

    for symbol in SYMBOLS:
        validation_path = SPLIT_FOLDER / (symbol + "_validation.csv")
        validation = pd.read_csv(validation_path)
        validation_dates = pd.to_datetime(validation["date"]).reset_index(drop=True)

        for model_name in MODELS:
            one_forecast = forecasts[
                (forecasts["symbol"] == symbol)
                & (forecasts["model"] == model_name)
            ].copy()
            one_forecast = one_forecast.sort_values("date").reset_index(drop=True)

            if not one_forecast["date"].equals(validation_dates):
                raise ValueError(
                    model_name + " has incorrect validation dates for " + symbol
                )

            actual_log = one_forecast["actual_log_variance"].to_numpy()
            forecast_log = one_forecast["forecast_log_variance"].to_numpy()
            expected_mse, expected_mae, expected_qlike = calculate_scores(
                actual_log, forecast_log
            )

            saved_score = scores[
                (scores["symbol"] == symbol)
                & (scores["model"] == model_name)
            ]

            if len(saved_score) != 1:
                raise ValueError("A model score row is missing or repeated")

            saved_score = saved_score.iloc[0]

            if not np.isclose(saved_score["log_mse"], expected_mse):
                raise ValueError("Saved log-MSE is incorrect")

            if not np.isclose(saved_score["log_mae"], expected_mae):
                raise ValueError("Saved log-MAE is incorrect")

            if not np.isclose(saved_score["qlike"], expected_qlike):
                raise ValueError("Saved QLIKE is incorrect")

    # Each HAR model must have one intercept and three feature coefficients.
    coefficient_counts = coefficients.groupby("symbol").size()
    for symbol in SYMBOLS:
        if coefficient_counts[symbol] != 4:
            raise ValueError("Incorrect number of HAR coefficients for " + symbol)

    print(scores.sort_values(["symbol", "qlike"]).to_string(index=False))
    print("\nAll baseline checks passed.")
    print("No test-period dates were used.")


if __name__ == "__main__":
    main()
