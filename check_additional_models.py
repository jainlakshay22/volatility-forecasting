"""Check the exploratory additional-model comparison."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
SPLIT_FOLDER = PROJECT_FOLDER / "data/splits"
RESULTS_FOLDER = PROJECT_FOLDER / "results/additional_models"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
MODELS = ["Linear HAR", "Ridge HAR", "Random Forest", "Gradient Boosting"]


def calculate_scores(actual_log, forecast_log):
    """Recalculate validation scores."""

    difference = actual_log - forecast_log
    log_mse = np.mean(difference ** 2)
    log_mae = np.mean(np.abs(difference))
    ratio = np.exp(actual_log) / np.exp(forecast_log)
    qlike = np.mean(ratio - np.log(ratio) - 1)
    return log_mse, log_mae, qlike


def main():
    """Check dates, predictions, scores, and test-period isolation."""

    forecasts = pd.read_csv(RESULTS_FOLDER / "validation_forecasts.csv")
    scores = pd.read_csv(RESULTS_FOLDER / "validation_scores.csv")
    importance = pd.read_csv(
        RESULTS_FOLDER / "coefficients_and_importance.csv"
    )
    forecasts["date"] = pd.to_datetime(forecasts["date"])

    if forecasts.isna().any().any():
        raise ValueError("An additional-model forecast value is missing")

    if (forecasts["date"] >= pd.Timestamp("2025-01-01")).any():
        raise ValueError("A 2025 test date appears in exploratory results")

    if (forecasts["forecast_variance"] <= 0).any():
        raise ValueError("A forecast variance is not positive")

    for symbol in SYMBOLS:
        validation = pd.read_csv(
            SPLIT_FOLDER / (symbol + "_validation.csv")
        )
        validation_dates = pd.to_datetime(validation["date"]).reset_index(drop=True)

        for model_name in MODELS:
            one_model = forecasts[
                (forecasts["symbol"] == symbol)
                & (forecasts["model"] == model_name)
            ].sort_values("date").reset_index(drop=True)

            if not one_model["date"].equals(validation_dates):
                raise ValueError("Validation dates are incorrect for " + model_name)

            actual_log = one_model["actual_log_variance"].to_numpy()
            forecast_log = one_model["forecast_log_variance"].to_numpy()
            expected_mse, expected_mae, expected_qlike = calculate_scores(
                actual_log, forecast_log
            )

            saved = scores[
                (scores["symbol"] == symbol)
                & (scores["model"] == model_name)
            ].iloc[0]

            if not np.isclose(saved["log_mse"], expected_mse):
                raise ValueError("Saved log-MSE is incorrect")

            if not np.isclose(saved["log_mae"], expected_mae):
                raise ValueError("Saved log-MAE is incorrect")

            if not np.isclose(saved["qlike"], expected_qlike):
                raise ValueError("Saved QLIKE is incorrect")

    expected_importance_rows = len(SYMBOLS) * len(MODELS) * 3
    if len(importance) != expected_importance_rows:
        raise ValueError("The feature-importance table has the wrong size")

    print(scores.sort_values(["symbol", "qlike"]).to_string(index=False))
    print("\nAll additional-model checks passed.")
    print("No 2025 test dates were used.")


if __name__ == "__main__":
    main()
