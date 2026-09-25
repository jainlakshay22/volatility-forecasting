"""Split the feature data into training, validation, and test periods."""

from pathlib import Path

import pandas as pd


# ---------------------------
# 1. Project settings
# ---------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
FEATURE_FOLDER = PROJECT_FOLDER / "data/features"
SPLIT_FOLDER = PROJECT_FOLDER / "data/splits"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# These dates are fixed before model training.
# We do not use a random split because this is a time-series problem.
TRAIN_END = pd.Timestamp("2023-12-31")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")


# ---------------------------
# 2. Read the model-ready data
# ---------------------------

def read_model_data(symbol):
    """Read the feature data created by create_features.py."""

    file_path = FEATURE_FOLDER / (symbol + "_model_data.csv")

    if not file_path.exists():
        raise FileNotFoundError(
            "Could not find "
            + str(file_path)
            + ". Run create_features.py first."
        )

    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    if data["date"].duplicated().any():
        raise ValueError("Repeated dates were found for " + symbol)

    return data


# ---------------------------
# 3. Make the chronological split
# ---------------------------

def make_splits(data):
    """Divide rows by target date without shuffling them."""

    train = data[data["date"] <= TRAIN_END].copy()

    validation = data[
        (data["date"] >= VALIDATION_START)
        & (data["date"] <= VALIDATION_END)
    ].copy()

    test = data[
        (data["date"] >= TEST_START)
        & (data["date"] <= TEST_END)
    ].copy()

    return train, validation, test


# ---------------------------
# 4. Save one asset's splits
# ---------------------------

def save_one_symbol(symbol):
    """Create and save the three files for one asset."""

    data = read_model_data(symbol)
    train, validation, test = make_splits(data)

    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise ValueError("One of the data splits is empty for " + symbol)

    SPLIT_FOLDER.mkdir(parents=True, exist_ok=True)

    train_path = SPLIT_FOLDER / (symbol + "_train.csv")
    validation_path = SPLIT_FOLDER / (symbol + "_validation.csv")
    test_path = SPLIT_FOLDER / (symbol + "_test.csv")

    train.to_csv(train_path, index=False, date_format="%Y-%m-%d")
    validation.to_csv(validation_path, index=False, date_format="%Y-%m-%d")
    test.to_csv(test_path, index=False, date_format="%Y-%m-%d")

    summary_rows = [
        {
            "symbol": symbol,
            "split": "train",
            "first_date": train["date"].min().date(),
            "last_date": train["date"].max().date(),
            "rows": len(train),
        },
        {
            "symbol": symbol,
            "split": "validation",
            "first_date": validation["date"].min().date(),
            "last_date": validation["date"].max().date(),
            "rows": len(validation),
        },
        {
            "symbol": symbol,
            "split": "test",
            "first_date": test["date"].min().date(),
            "last_date": test["date"].max().date(),
            "rows": len(test),
        },
    ]

    print("\nCreated splits for", symbol)
    for row in summary_rows:
        print(
            row["split"],
            "|",
            row["first_date"],
            "to",
            row["last_date"],
            "|",
            row["rows"],
            "rows",
        )

    return summary_rows


def main():
    """Create chronological splits for Bitcoin and Ethereum."""

    all_summary_rows = []

    for symbol in SYMBOLS:
        one_symbol_summary = save_one_symbol(symbol)
        all_summary_rows.extend(one_symbol_summary)

    summary_table = pd.DataFrame(all_summary_rows)
    summary_table.to_csv(SPLIT_FOLDER / "split_summary.csv", index=False)

    print("\nChronological data splitting is complete.")


if __name__ == "__main__":
    main()
