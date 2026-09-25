"""Check the chronological training, validation, and test files."""

from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
FEATURE_FOLDER = PROJECT_FOLDER / "data/features"
SPLIT_FOLDER = PROJECT_FOLDER / "data/splits"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FEATURE_COLUMNS = [
    "daily_log_variance",
    "weekly_log_variance",
    "monthly_log_variance",
]
TARGET_COLUMN = "target_log_variance"


def read_csv_with_dates(file_path):
    """Read one CSV file and convert its date column."""

    if not file_path.exists():
        raise FileNotFoundError("Could not find " + str(file_path))

    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])
    return data


def check_one_symbol(symbol):
    """Check all three splits for one asset."""

    original_path = FEATURE_FOLDER / (symbol + "_model_data.csv")
    train_path = SPLIT_FOLDER / (symbol + "_train.csv")
    validation_path = SPLIT_FOLDER / (symbol + "_validation.csv")
    test_path = SPLIT_FOLDER / (symbol + "_test.csv")

    original = read_csv_with_dates(original_path)
    train = read_csv_with_dates(train_path)
    validation = read_csv_with_dates(validation_path)
    test = read_csv_with_dates(test_path)

    split_tables = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # Each file must be sorted and contain each target date only once.
    for split_name, data in split_tables.items():
        if data["date"].duplicated().any():
            raise ValueError(split_name + " has repeated dates for " + symbol)

        if not data["date"].is_monotonic_increasing:
            raise ValueError(split_name + " is not in date order for " + symbol)

        columns_to_check = FEATURE_COLUMNS + [TARGET_COLUMN]
        if data[columns_to_check].isna().any().any():
            raise ValueError(split_name + " contains missing model values")

    # The periods must appear in the correct historical order.
    if train["date"].max() >= validation["date"].min():
        raise ValueError("Training and validation periods overlap")

    if validation["date"].max() >= test["date"].min():
        raise ValueError("Validation and test periods overlap")

    # No target date may appear in more than one split.
    train_dates = set(train["date"])
    validation_dates = set(validation["date"])
    test_dates = set(test["date"])

    if len(train_dates.intersection(validation_dates)) > 0:
        raise ValueError("A date appears in both training and validation")

    if len(train_dates.intersection(test_dates)) > 0:
        raise ValueError("A date appears in both training and test")

    if len(validation_dates.intersection(test_dates)) > 0:
        raise ValueError("A date appears in both validation and test")

    # Joining the three files must reproduce the original model data exactly.
    joined = pd.concat([train, validation, test], ignore_index=True)
    joined = joined.sort_values("date").reset_index(drop=True)
    original = original.sort_values("date").reset_index(drop=True)

    if not joined.equals(original):
        raise ValueError("The split files do not reproduce the original data")

    summary = {
        "symbol": symbol,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "total_rows": len(joined),
        "ordering_check": True,
        "overlap_check": True,
        "reconstruction_check": True,
    }

    return summary


def main():
    """Check the splits for both assets."""

    summaries = []

    for symbol in SYMBOLS:
        summaries.append(check_one_symbol(symbol))

    summary_table = pd.DataFrame(summaries)
    print(summary_table.to_string(index=False))
    print("\nAll chronological split checks passed.")


if __name__ == "__main__":
    main()
