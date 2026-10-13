"""Check the paired uncertainty analysis."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
FINAL_FOLDER = PROJECT_FOLDER / "results/final_evaluation"
UNCERTAINTY_FOLDER = PROJECT_FOLDER / "results/uncertainty"


def qlike_loss(actual_log, forecast_log):
    """Recalculate daily QLIKE loss."""

    ratio = np.exp(actual_log) / np.exp(forecast_log)
    return ratio - np.log(ratio) - 1


def main():
    """Verify paired dates, point estimates, and interval directions."""

    forecasts = pd.read_csv(FINAL_FOLDER / "test_forecasts.csv")
    results = pd.read_csv(UNCERTAINTY_FOLDER / "paired_uncertainty.csv")
    forecasts["date"] = pd.to_datetime(forecasts["date"])

    if len(results) != 12:
        raise ValueError("The uncertainty table should contain 12 comparisons")

    for row_number in range(len(results)):
        row = results.iloc[row_number]
        symbol = row["symbol"]
        benchmark_name = row["benchmark"]

        selected = forecasts[
            (forecasts["symbol"] == symbol)
            & (forecasts["model"] == "Linear HAR")
        ].sort_values("date").reset_index(drop=True)

        benchmark = forecasts[
            (forecasts["symbol"] == symbol)
            & (forecasts["model"] == benchmark_name)
        ].sort_values("date").reset_index(drop=True)

        if not selected["date"].equals(benchmark["date"]):
            raise ValueError("Dates do not match for an uncertainty comparison")

        actual_log = selected["actual_log_variance"].to_numpy()
        selected_log = selected["forecast_log_variance"].to_numpy()
        benchmark_log = benchmark["forecast_log_variance"].to_numpy()

        if row["loss"] == "QLIKE":
            selected_loss = qlike_loss(actual_log, selected_log)
            benchmark_loss = qlike_loss(actual_log, benchmark_log)
        else:
            selected_loss = (actual_log - selected_log) ** 2
            benchmark_loss = (actual_log - benchmark_log) ** 2

        expected_difference = (benchmark_loss - selected_loss).mean()

        if not np.isclose(row["mean_loss_improvement"], expected_difference):
            raise ValueError("A saved mean loss improvement is incorrect")

        if row["confidence_interval_lower"] > row["confidence_interval_upper"]:
            raise ValueError("A confidence interval is reversed")

        expected_excludes_zero = row["confidence_interval_lower"] > 0
        if row["confidence_interval_excludes_zero"] != expected_excludes_zero:
            raise ValueError("An interval interpretation is incorrect")

        probability = row["bootstrap_probability_improvement"]
        if probability < 0 or probability > 1:
            raise ValueError("A bootstrap probability is outside zero to one")

    print(results.to_string(index=False))
    print("\nAll uncertainty checks passed.")


if __name__ == "__main__":
    main()
