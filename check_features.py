"""Check that forecasting features use only past information."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
FEATURE_FOLDER = PROJECT_FOLDER / "data/features"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
WEEK_LENGTH = 5
MONTH_LENGTH = 22


def values_match(saved_values, expected_values):
    """Compare two columns while treating matching missing values as equal."""

    return np.allclose(
        saved_values,
        expected_values,
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


def check_one_symbol(symbol):
    """Recalculate and check every feature for one asset."""

    all_features_path = FEATURE_FOLDER / (symbol + "_all_features.csv")
    model_data_path = FEATURE_FOLDER / (symbol + "_model_data.csv")

    if not all_features_path.exists() or not model_data_path.exists():
        raise FileNotFoundError(
            "Feature files are missing for "
            + symbol
            + ". Run create_features.py first."
        )

    data = pd.read_csv(all_features_path)
    model_data = pd.read_csv(model_data_path)

    data["date"] = pd.to_datetime(data["date"])
    model_data["date"] = pd.to_datetime(model_data["date"])

    if data["date"].duplicated().any():
        raise ValueError("Repeated dates were found for " + symbol)

    # Recalculate log variance directly from the saved realized variance.
    expected_log_variance = np.log(data["realized_variance"])
    if not values_match(data["log_realized_variance"], expected_log_variance):
        raise ValueError("The log-variance calculation is incorrect")

    # Recalculate all three features independently.
    past_log_variance = expected_log_variance.shift(1)
    expected_daily = past_log_variance
    expected_weekly = past_log_variance.rolling(
        window=WEEK_LENGTH,
        min_periods=WEEK_LENGTH,
    ).mean()
    expected_monthly = past_log_variance.rolling(
        window=MONTH_LENGTH,
        min_periods=MONTH_LENGTH,
    ).mean()

    if not values_match(data["daily_log_variance"], expected_daily):
        raise ValueError("The daily feature is incorrect")

    if not values_match(data["weekly_log_variance"], expected_weekly):
        raise ValueError("The weekly feature is incorrect")

    if not values_match(data["monthly_log_variance"], expected_monthly):
        raise ValueError("The monthly feature is incorrect")

    # Confirm that the model file contains exactly the usable dates.
    needed_columns = [
        "target_log_variance",
        "daily_log_variance",
        "weekly_log_variance",
        "monthly_log_variance",
    ]
    expected_usable = data[needed_columns].notna().all(axis=1)
    expected_dates = data.loc[expected_usable, "date"].reset_index(drop=True)
    saved_dates = model_data["date"].reset_index(drop=True)

    if not expected_dates.equals(saved_dates):
        raise ValueError("The model file contains the wrong dates")

    if model_data[needed_columns].isna().any().any():
        raise ValueError("The model file contains a missing feature or target")

    # For each target date, the daily feature must come from the previous
    # calendar date. This explicit date check guards against future leakage.
    previous_dates = data["date"].shift(1)
    one_day = pd.Timedelta(days=1)
    correct_previous_date = (data["date"] - previous_dates) == one_day

    usable_rows = data["usable_for_model"] == True
    if not correct_previous_date[usable_rows].all():
        raise ValueError("A feature does not come from the previous calendar day")

    summary = {
        "symbol": symbol,
        "calendar_rows": len(data),
        "model_rows": len(model_data),
        "first_model_date": model_data["date"].min().date(),
        "last_model_date": model_data["date"].max().date(),
        "feature_checks_passed": True,
        "past_only_check_passed": True,
    }

    return summary


def main():
    """Check both assets and print the results."""

    summaries = []

    for symbol in SYMBOLS:
        summaries.append(check_one_symbol(symbol))

    summary_table = pd.DataFrame(summaries)
    print(summary_table.to_string(index=False))
    print("\nAll feature and past-only checks passed.")


if __name__ == "__main__":
    main()
