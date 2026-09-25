"""Check the daily files created by calculate_volatility.py."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
DAILY_FOLDER = PROJECT_FOLDER / "data/daily"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def check_one_symbol(symbol):
    """Run simple checks on one daily volatility file."""

    file_path = DAILY_FOLDER / (symbol + "_daily_volatility.csv")

    if not file_path.exists():
        raise FileNotFoundError(
            "Could not find "
            + str(file_path)
            + ". Run calculate_volatility.py first."
        )

    daily = pd.read_csv(file_path)
    daily["date"] = pd.to_datetime(daily["date"], utc=True)

    required_columns = [
        "date",
        "complete",
        "number_of_bars",
        "number_of_valid_returns",
        "realized_variance",
        "realized_volatility",
    ]

    for column in required_columns:
        if column not in daily.columns:
            raise ValueError("Missing column: " + column)

    if daily["date"].duplicated().any():
        raise ValueError("Repeated daily dates were found for " + symbol)

    expected_dates = pd.date_range(
        daily["date"].min(), daily["date"].max(), freq="D"
    )

    if len(daily) != len(expected_dates):
        raise ValueError("The daily calendar has gaps for " + symbol)

    complete_days = daily[daily["complete"] == True].copy()
    incomplete_days = daily[daily["complete"] == False].copy()

    if not (complete_days["number_of_bars"] == 288).all():
        raise ValueError("A complete day does not have 288 bars")

    if not (complete_days["number_of_valid_returns"] == 288).all():
        raise ValueError("A complete day does not have 288 valid returns")

    if complete_days["realized_variance"].isna().any():
        raise ValueError("A complete day has missing variance")

    if (complete_days["realized_variance"] < 0).any():
        raise ValueError("A negative variance was found")

    if incomplete_days["realized_variance"].notna().any():
        raise ValueError("An incomplete day has a variance value")

    # Check that volatility really is the square root of variance.
    calculated_volatility = np.sqrt(complete_days["realized_variance"])
    saved_volatility = complete_days["realized_volatility"]

    if not np.allclose(calculated_volatility, saved_volatility):
        raise ValueError("Variance and volatility do not match")

    average_volatility = complete_days["realized_volatility_percent"].mean()

    summary = {
        "symbol": symbol,
        "first_date": daily["date"].min().date(),
        "last_date": daily["date"].max().date(),
        "total_days": len(daily),
        "complete_days": len(complete_days),
        "incomplete_days": len(incomplete_days),
        "average_daily_volatility_percent": round(average_volatility, 4),
    }

    return summary


def main():
    """Check both assets and print a small summary."""

    summaries = []

    for symbol in SYMBOLS:
        summary = check_one_symbol(symbol)
        summaries.append(summary)

    summary_table = pd.DataFrame(summaries)
    print(summary_table.to_string(index=False))
    print("\nAll daily volatility checks passed.")


if __name__ == "__main__":
    main()
