"""Create past-only features for daily volatility forecasting."""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------
# 1. Project settings
# ---------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
DAILY_FOLDER = PROJECT_FOLDER / "data/daily"
FEATURE_FOLDER = PROJECT_FOLDER / "data/features"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Five trading days are often treated as one week.
WEEK_LENGTH = 5

# Twenty-two trading days are often treated as one month.
MONTH_LENGTH = 22


# ---------------------------
# 2. Read one daily file
# ---------------------------

def read_daily_data(symbol):
    """Read daily volatility while keeping incomplete calendar days."""

    file_path = DAILY_FOLDER / (symbol + "_daily_volatility.csv")

    if not file_path.exists():
        raise FileNotFoundError(
            "Could not find "
            + str(file_path)
            + ". Run calculate_volatility.py first."
        )

    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    if data["date"].duplicated().any():
        raise ValueError("Repeated dates were found for " + symbol)

    # Make sure every calendar date is present. Missing observations should be
    # represented by NaN values, not by removing the date from the table.
    expected_dates = pd.date_range(
        data["date"].min(), data["date"].max(), freq="D"
    )

    if len(data) != len(expected_dates):
        raise ValueError("The daily calendar has missing rows for " + symbol)

    if not (data["date"].to_numpy() == expected_dates.to_numpy()).all():
        raise ValueError("The daily dates are not continuous for " + symbol)

    return data


# ---------------------------
# 3. Create past-only features
# ---------------------------

def add_features(data):
    """Create daily, weekly, and monthly log-variance features."""

    data = data.copy()

    # Logarithms require positive values.
    valid_variance = data["realized_variance"].dropna()
    if (valid_variance <= 0).any():
        raise ValueError("Realized variance must be positive before taking logs")

    # Volatility can be strongly right-skewed. Taking the logarithm of variance
    # makes very large observations less dominant.
    data["log_realized_variance"] = np.log(data["realized_variance"])

    # shift(1) is the most important line in this script. It moves today's
    # variance down one row, so a row dated today uses information ending
    # yesterday. Without this shift, the features would contain future data.
    past_log_variance = data["log_realized_variance"].shift(1)

    # Daily feature: yesterday's log realized variance.
    data["daily_log_variance"] = past_log_variance

    # Weekly feature: the average of the previous five calendar days.
    data["weekly_log_variance"] = past_log_variance.rolling(
        window=WEEK_LENGTH,
        min_periods=WEEK_LENGTH,
    ).mean()

    # Monthly feature: the average of the previous 22 calendar days.
    data["monthly_log_variance"] = past_log_variance.rolling(
        window=MONTH_LENGTH,
        min_periods=MONTH_LENGTH,
    ).mean()

    # The value we want a model to predict is today's log realized variance.
    data["target_log_variance"] = data["log_realized_variance"]

    # A row is usable only when its target and all three features are present.
    needed_columns = [
        "target_log_variance",
        "daily_log_variance",
        "weekly_log_variance",
        "monthly_log_variance",
    ]
    data["usable_for_model"] = data[needed_columns].notna().all(axis=1)

    return data


# ---------------------------
# 4. Save the feature files
# ---------------------------

def process_one_symbol(symbol):
    """Create and save features for one asset."""

    daily_data = read_daily_data(symbol)
    all_features = add_features(daily_data)

    FEATURE_FOLDER.mkdir(parents=True, exist_ok=True)

    # This file keeps every calendar day. It is useful for checking why some
    # rows cannot be used by a model.
    all_features_path = FEATURE_FOLDER / (symbol + "_all_features.csv")
    all_features.to_csv(
        all_features_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    # This smaller file contains only rows that are ready for modelling.
    model_columns = [
        "date",
        "symbol",
        "realized_variance",
        "realized_volatility",
        "target_log_variance",
        "daily_log_variance",
        "weekly_log_variance",
        "monthly_log_variance",
    ]

    model_data = all_features[all_features["usable_for_model"] == True].copy()
    model_data = model_data[model_columns]

    model_data_path = FEATURE_FOLDER / (symbol + "_model_data.csv")
    model_data.to_csv(
        model_data_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print("\nCreated features for", symbol)
    print("All calendar rows:", len(all_features))
    print("Rows ready for modelling:", len(model_data))
    print("First usable target date:", model_data["date"].min().date())
    print("Last usable target date:", model_data["date"].max().date())


def main():
    """Create feature files for Bitcoin and Ethereum."""

    for symbol in SYMBOLS:
        process_one_symbol(symbol)

    print("\nFeature creation is complete.")


if __name__ == "__main__":
    main()
