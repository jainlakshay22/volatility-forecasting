"""Check the frozen-model 2025 test evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
SPLIT_FOLDER = PROJECT_FOLDER / "data/splits"
BASELINE_FOLDER = PROJECT_FOLDER / "results/baselines"
RESULTS_FOLDER = PROJECT_FOLDER / "results/final_evaluation"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
MODELS = [
    "Persistence",
    "Five-day average",
    "Historical mean",
    "Linear HAR",
]


def calculate_scores(actual_log, forecast_log):
    """Independently recalculate final test scores."""

    difference = actual_log - forecast_log
    log_mse = np.mean(difference ** 2)
    log_mae = np.mean(np.abs(difference))

    ratio = np.exp(actual_log) / np.exp(forecast_log)
    qlike = np.mean(ratio - np.log(ratio) - 1)
    return log_mse, log_mae, qlike


def main():
    """Verify model selection, fitting dates, test dates, and scores."""

    selections = pd.read_csv(RESULTS_FOLDER / "model_selection.csv")
    forecasts = pd.read_csv(RESULTS_FOLDER / "test_forecasts.csv")
    scores = pd.read_csv(RESULTS_FOLDER / "test_scores.csv")
    validation_scores = pd.read_csv(BASELINE_FOLDER / "validation_scores.csv")

    forecasts["date"] = pd.to_datetime(forecasts["date"])

    if forecasts.isna().any().any():
        raise ValueError("A test forecast value is missing")

    if (forecasts["forecast_variance"] <= 0).any():
        raise ValueError("A test variance forecast is not positive")

    # Confirm model selection using validation results, not test results.
    for symbol in SYMBOLS:
        symbol_validation = validation_scores[
            validation_scores["symbol"] == symbol
        ].sort_values("qlike")
        validation_winner = symbol_validation.iloc[0]["model"]

        saved_selection = selections[selections["symbol"] == symbol].iloc[0]
        if saved_selection["selected_model"] != validation_winner:
            raise ValueError("Saved model selection does not match validation")

        if saved_selection["selected_before_test"] != True:
            raise ValueError("The selected-before-test flag is not true")

        test_path = SPLIT_FOLDER / (symbol + "_test.csv")
        test_data = pd.read_csv(test_path)
        test_dates = pd.to_datetime(test_data["date"]).reset_index(drop=True)

        for model_name in MODELS:
            one_forecast = forecasts[
                (forecasts["symbol"] == symbol)
                & (forecasts["model"] == model_name)
            ].sort_values("date").reset_index(drop=True)

            if not one_forecast["date"].equals(test_dates):
                raise ValueError("Saved test dates are incorrect")

            actual_log = one_forecast["actual_log_variance"].to_numpy()
            forecast_log = one_forecast["forecast_log_variance"].to_numpy()
            expected_mse, expected_mae, expected_qlike = calculate_scores(
                actual_log, forecast_log
            )

            saved_score = scores[
                (scores["symbol"] == symbol)
                & (scores["model"] == model_name)
            ].iloc[0]

            if saved_score["fit_end"] != "2024-12-31":
                raise ValueError("A model was fitted beyond the allowed date")

            if saved_score["test_start"] != "2025-01-01":
                raise ValueError("The test period starts on the wrong date")

            if not np.isclose(saved_score["log_mse"], expected_mse):
                raise ValueError("Final log-MSE is incorrect")

            if not np.isclose(saved_score["log_mae"], expected_mae):
                raise ValueError("Final log-MAE is incorrect")

            if not np.isclose(saved_score["qlike"], expected_qlike):
                raise ValueError("Final QLIKE is incorrect")

    print(scores.sort_values(["symbol", "qlike"]).to_string(index=False))
    print("\nAll final-evaluation checks passed.")
    print("Model fitting ended before the 2025 test period began.")


if __name__ == "__main__":
    main()
