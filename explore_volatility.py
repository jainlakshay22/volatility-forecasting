"""Explore the daily Bitcoin and Ethereum volatility data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------
# 1. Project settings
# ---------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
DAILY_FOLDER = PROJECT_FOLDER / "data/daily"
RESULTS_FOLDER = PROJECT_FOLDER / "results/exploration"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# ---------------------------
# 2. Read one daily file
# ---------------------------

def read_daily_data(symbol):
    """Read one daily volatility file and keep complete days."""

    file_name = symbol + "_daily_volatility.csv"
    file_path = DAILY_FOLDER / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            "Could not find "
            + str(file_path)
            + ". Run calculate_volatility.py first."
        )

    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])

    # Incomplete days do not have a valid full-day volatility measurement.
    complete_data = data[data["complete"] == True].copy()
    complete_data = complete_data.sort_values("date").reset_index(drop=True)

    if len(complete_data) == 0:
        raise ValueError("No complete days were found for " + symbol)

    return data, complete_data


# ---------------------------
# 3. Create summary tables
# ---------------------------

def make_summary(symbol, all_days, complete_days):
    """Create one row of summary statistics for an asset."""

    volatility = complete_days["realized_volatility_percent"]

    summary = {
        "symbol": symbol,
        "first_date": complete_days["date"].min().date(),
        "last_date": complete_days["date"].max().date(),
        "total_calendar_days": len(all_days),
        "complete_days": len(complete_days),
        "incomplete_days": int((all_days["complete"] == False).sum()),
        "average_volatility_percent": volatility.mean(),
        "median_volatility_percent": volatility.median(),
        "standard_deviation_percent": volatility.std(),
        "minimum_volatility_percent": volatility.min(),
        "maximum_volatility_percent": volatility.max(),
        "lag_1_autocorrelation": volatility.autocorr(lag=1),
    }

    return summary


def make_yearly_summary(symbol, complete_days):
    """Calculate volatility statistics for each calendar year."""

    data = complete_days.copy()
    data["year"] = data["date"].dt.year

    yearly = data.groupby("year")["realized_volatility_percent"].agg(
        ["count", "mean", "median", "min", "max"]
    )
    yearly = yearly.reset_index()

    yearly.insert(0, "symbol", symbol)
    yearly = yearly.rename(
        columns={
            "count": "complete_days",
            "mean": "average_volatility_percent",
            "median": "median_volatility_percent",
            "min": "minimum_volatility_percent",
            "max": "maximum_volatility_percent",
        }
    )

    return yearly


def find_highest_volatility_days(symbol, complete_days):
    """Return the ten highest-volatility days for an asset."""

    columns_to_keep = [
        "date",
        "close",
        "daily_log_return",
        "realized_volatility_percent",
    ]

    highest_days = complete_days.nlargest(
        10, "realized_volatility_percent"
    )[columns_to_keep].copy()

    highest_days.insert(0, "symbol", symbol)
    return highest_days


# ---------------------------
# 4. Create charts
# ---------------------------

def plot_volatility_over_time(symbol, complete_days):
    """Save a line chart of daily volatility."""

    plt.figure(figsize=(12, 5))
    plt.plot(
        complete_days["date"],
        complete_days["realized_volatility_percent"],
        color="navy",
        linewidth=0.8,
    )
    plt.title(symbol + " Daily Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = RESULTS_FOLDER / (symbol + "_volatility_over_time.png")
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_volatility_distribution(symbol, complete_days):
    """Save a histogram showing the distribution of volatility."""

    plt.figure(figsize=(8, 5))
    plt.hist(
        complete_days["realized_volatility_percent"],
        bins=50,
        color="steelblue",
        edgecolor="white",
    )
    plt.title(symbol + " Distribution of Daily Realized Volatility")
    plt.xlabel("Volatility (%)")
    plt.ylabel("Number of days")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = RESULTS_FOLDER / (symbol + "_volatility_distribution.png")
    plt.savefig(output_path, dpi=150)
    plt.close()


def compare_btc_and_eth(btc_data, eth_data):
    """Match BTC and ETH by date and create comparison results."""

    btc = btc_data[["date", "realized_volatility_percent"]].copy()
    eth = eth_data[["date", "realized_volatility_percent"]].copy()

    btc = btc.rename(
        columns={"realized_volatility_percent": "BTC_volatility_percent"}
    )
    eth = eth.rename(
        columns={"realized_volatility_percent": "ETH_volatility_percent"}
    )

    comparison = pd.merge(btc, eth, on="date", how="inner")
    correlation = comparison["BTC_volatility_percent"].corr(
        comparison["ETH_volatility_percent"]
    )

    comparison.to_csv(
        RESULTS_FOLDER / "BTC_ETH_daily_comparison.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        comparison["date"],
        comparison["BTC_volatility_percent"],
        label="BTC",
        linewidth=0.8,
    )
    plt.plot(
        comparison["date"],
        comparison["ETH_volatility_percent"],
        label="ETH",
        linewidth=0.8,
        alpha=0.8,
    )
    plt.title("BTC and ETH Daily Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_FOLDER / "BTC_ETH_volatility_comparison.png", dpi=150)
    plt.close()

    return correlation, len(comparison)


# ---------------------------
# 5. Run the exploration
# ---------------------------

def main():
    """Create all exploration tables and charts."""

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    summaries = []
    yearly_tables = []
    highest_day_tables = []
    complete_data_by_symbol = {}

    for symbol in SYMBOLS:
        print("Exploring", symbol)

        all_days, complete_days = read_daily_data(symbol)
        complete_data_by_symbol[symbol] = complete_days

        summaries.append(make_summary(symbol, all_days, complete_days))
        yearly_tables.append(make_yearly_summary(symbol, complete_days))
        highest_day_tables.append(
            find_highest_volatility_days(symbol, complete_days)
        )

        plot_volatility_over_time(symbol, complete_days)
        plot_volatility_distribution(symbol, complete_days)

    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(RESULTS_FOLDER / "summary_statistics.csv", index=False)

    yearly_table = pd.concat(yearly_tables, ignore_index=True)
    yearly_table.to_csv(RESULTS_FOLDER / "yearly_statistics.csv", index=False)

    highest_days = pd.concat(highest_day_tables, ignore_index=True)
    highest_days.to_csv(
        RESULTS_FOLDER / "highest_volatility_days.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    correlation, matched_days = compare_btc_and_eth(
        complete_data_by_symbol["BTCUSDT"],
        complete_data_by_symbol["ETHUSDT"],
    )

    print("\nSummary statistics")
    print(summary_table.to_string(index=False))
    print("\nMatched BTC and ETH days:", matched_days)
    print("BTC and ETH volatility correlation:", round(correlation, 4))
    print("\nResults were saved in", RESULTS_FOLDER)


if __name__ == "__main__":
    main()
