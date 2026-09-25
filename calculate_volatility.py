"""Turn five-minute prices into daily realized volatility."""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------
# 1. Project settings
# ---------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = PROJECT_FOLDER / "data/clean"
OUTPUT_FOLDER = PROJECT_FOLDER / "data/daily"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIVE_MINUTES = pd.Timedelta(minutes=5)
EXPECTED_BARS_PER_DAY = 288


# ---------------------------
# 2. Read one clean price file
# ---------------------------

def read_price_data(symbol):
    """Read the clean five-minute file created by download_data.py."""

    file_name = symbol + "_5m_2020_2025.csv.gz"
    file_path = INPUT_FOLDER / file_name

    if not file_path.exists():
        message = (
            "Could not find "
            + str(file_path)
            + ". Run download_data.py before this script."
        )
        raise FileNotFoundError(message)

    data = pd.read_csv(file_path)
    data["open_time"] = pd.to_datetime(data["open_time"], utc=True)

    # Ensure that prices and times are in the correct order.
    data = data.sort_values("open_time").reset_index(drop=True)

    if data["open_time"].duplicated().any():
        raise ValueError("Repeated timestamps were found for " + symbol)

    if (data["close"] <= 0).any():
        raise ValueError("A zero or negative closing price was found for " + symbol)

    return data


# ---------------------------
# 3. Calculate five-minute returns
# ---------------------------

def add_returns(data):
    """Add log returns and squared log returns to the five-minute data."""

    data = data.copy()

    # shift(1) moves the closing-price column down by one row.
    # Each row can then be compared with the previous row.
    previous_close = data["close"].shift(1)

    # A log return is log(current price / previous price).
    data["log_return"] = np.log(data["close"] / previous_close)

    # Check the amount of time between each row and the previous row.
    time_since_previous_row = data["open_time"].diff()

    # A return is valid only when the two prices are exactly five minutes apart.
    correct_time_gap = time_since_previous_row == FIVE_MINUTES
    data.loc[~correct_time_gap, "log_return"] = np.nan

    # Squared returns are the building blocks of realized variance.
    data["squared_return"] = data["log_return"] ** 2

    # UTC is used so every daily group has the same definition.
    data["date"] = data["open_time"].dt.floor("D")

    return data


# ---------------------------
# 4. Create the daily table
# ---------------------------

def create_daily_data(data, symbol):
    """Group the five-minute values into UTC calendar days."""

    daily_group = data.groupby("date")

    daily = pd.DataFrame()
    daily["open"] = daily_group["open"].first()
    daily["high"] = daily_group["high"].max()
    daily["low"] = daily_group["low"].min()
    daily["close"] = daily_group["close"].last()
    daily["volume"] = daily_group["volume"].sum()

    # These counts tell us whether a day is complete.
    daily["number_of_bars"] = daily_group["close"].count()
    daily["number_of_valid_returns"] = daily_group["log_return"].count()

    # Realized variance is the sum of the squared five-minute returns.
    daily["realized_variance"] = daily_group["squared_return"].sum(min_count=1)

    # The daily log return is the sum of the valid five-minute log returns.
    daily["daily_log_return"] = daily_group["log_return"].sum(min_count=1)

    # Add any completely missing calendar dates to the table.
    first_date = data["date"].min()
    last_date = data["date"].max()
    complete_calendar = pd.date_range(first_date, last_date, freq="D")
    daily = daily.reindex(complete_calendar)
    daily.index.name = "date"

    # A full UTC day has 288 five-minute bars and 288 valid returns.
    daily["complete"] = (
        (daily["number_of_bars"] == EXPECTED_BARS_PER_DAY)
        & (daily["number_of_valid_returns"] == EXPECTED_BARS_PER_DAY)
    )

    # Do not use a partial day's variance as if it represented a full day.
    incomplete_day = ~daily["complete"]
    daily.loc[incomplete_day, "realized_variance"] = np.nan
    daily.loc[incomplete_day, "daily_log_return"] = np.nan

    # Volatility is the square root of variance.
    daily["realized_volatility"] = np.sqrt(daily["realized_variance"])

    # This percentage column is easier to interpret in tables and graphs.
    daily["realized_volatility_percent"] = daily["realized_volatility"] * 100

    daily.insert(0, "symbol", symbol)
    daily = daily.reset_index()

    # Counts should appear as whole numbers. Missing dates receive zero counts.
    daily["number_of_bars"] = daily["number_of_bars"].fillna(0).astype(int)
    daily["number_of_valid_returns"] = (
        daily["number_of_valid_returns"].fillna(0).astype(int)
    )

    return daily


# ---------------------------
# 5. Save one daily file
# ---------------------------

def process_one_symbol(symbol):
    """Calculate and save daily volatility for one asset."""

    prices = read_price_data(symbol)
    prices_with_returns = add_returns(prices)
    daily = create_daily_data(prices_with_returns, symbol)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_FOLDER / (symbol + "_daily_volatility.csv")
    daily.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    complete_days = int(daily["complete"].sum())
    incomplete_days = int((~daily["complete"]).sum())

    print("Saved", output_path)
    print("Complete days:", complete_days)
    print("Incomplete days:", incomplete_days)


def main():
    """Calculate daily volatility for Bitcoin and Ethereum."""

    for symbol in SYMBOLS:
        print("\nCalculating", symbol)
        process_one_symbol(symbol)

    print("\nDaily volatility calculation is complete.")


if __name__ == "__main__":
    main()
