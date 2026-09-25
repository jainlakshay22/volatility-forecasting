"""Download and clean Binance five-minute Bitcoin and Ethereum data."""

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ---------------------------
# 1. Project settings
# ---------------------------

# This finds the folder that contains this Python file.
PROJECT_FOLDER = Path(__file__).resolve().parent

# These folders do not need to exist yet. The script will create them.
RAW_FOLDER = PROJECT_FOLDER / "data/raw"
CLEAN_FOLDER = PROJECT_FOLDER / "data/clean"
CHECK_FOLDER = PROJECT_FOLDER / "data/checks"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIRST_MONTH = "2020-01"
LAST_MONTH = "2025-12"
TIME_INTERVAL = "5m"

BINANCE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# Binance stores the columns in this order.
COLUMN_NAMES = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


# ---------------------------
# 2. Download helper
# ---------------------------

def download_file(url, save_path):
    """Download one file unless it is already in our raw-data folder."""

    if save_path.exists():
        return

    print("Downloading", url, flush=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(response.content)


def calculate_sha256(file_path):
    """Calculate the SHA-256 fingerprint of one file."""

    file_contents = file_path.read_bytes()
    fingerprint = hashlib.sha256(file_contents).hexdigest()
    return fingerprint


def download_one_month(symbol, month):
    """Download and verify one monthly Binance ZIP file."""

    file_name = symbol + "-" + TIME_INTERVAL + "-" + month + ".zip"
    file_url = (
        BINANCE_URL
        + "/"
        + symbol
        + "/"
        + TIME_INTERVAL
        + "/"
        + file_name
    )

    symbol_folder = RAW_FOLDER / symbol
    zip_path = symbol_folder / file_name
    checksum_path = symbol_folder / (file_name + ".CHECKSUM")

    download_file(file_url, zip_path)
    download_file(file_url + ".CHECKSUM", checksum_path)

    # Binance puts the expected fingerprint at the start of the checksum file.
    checksum_text = checksum_path.read_text()
    expected_fingerprint = checksum_text.split()[0]
    actual_fingerprint = calculate_sha256(zip_path)

    if actual_fingerprint != expected_fingerprint:
        raise ValueError("Checksum failed for " + file_name)

    # testzip() returns None when every file inside the ZIP is readable.
    with zipfile.ZipFile(zip_path) as zip_file:
        damaged_file = zip_file.testzip()

    if damaged_file is not None:
        raise ValueError("Damaged file inside " + file_name)

    source_information = {
        "symbol": symbol,
        "month": month,
        "file_name": file_name,
        "download_url": file_url,
        "sha256": actual_fingerprint,
        "checksum_passed": True,
        "file_size_bytes": zip_path.stat().st_size,
    }

    return zip_path, source_information


# ---------------------------
# 3. Read and check one month
# ---------------------------

def read_one_month(zip_path, month):
    """Read one ZIP file and perform basic data-quality checks."""

    data = pd.read_csv(zip_path, header=None, names=COLUMN_NAMES)

    # Some recent Binance files include a header row. Remove it if it appears.
    data = data[data["open_time"].astype(str) != "open_time"].copy()

    # Change text values into numbers. An invalid value will stop the script.
    for column in COLUMN_NAMES:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if len(data) == 0:
        raise ValueError("No data was found in " + zip_path.name)

    # Old Binance files use milliseconds. Newer files use microseconds.
    typical_timestamp = data["open_time"].median()

    if typical_timestamp >= 1_000_000_000_000_000:
        timestamp_unit = "us"
    else:
        timestamp_unit = "ms"

    data["open_time"] = pd.to_datetime(
        data["open_time"], unit=timestamp_unit, utc=True
    )
    data["close_time"] = pd.to_datetime(
        data["close_time"], unit=timestamp_unit, utc=True
    )

    # Check the timestamps.
    if data["open_time"].duplicated().any():
        raise ValueError("Repeated timestamp in " + zip_path.name)

    if not data["open_time"].is_monotonic_increasing:
        raise ValueError("Unsorted timestamps in " + zip_path.name)

    rounded_times = data["open_time"].dt.floor("5min")
    if not (data["open_time"] == rounded_times).all():
        raise ValueError("Timestamp outside the five-minute grid")

    # Check that the prices and volumes are sensible.
    price_columns = ["open", "high", "low", "close"]
    volume_columns = [
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    columns_to_check = price_columns + volume_columns
    numbers = data[columns_to_check].to_numpy()

    if not np.isfinite(numbers).all():
        raise ValueError("Missing or infinite number in " + zip_path.name)

    if (data[price_columns] <= 0).any().any():
        raise ValueError("Zero or negative price in " + zip_path.name)

    if (data[volume_columns] < 0).any().any():
        raise ValueError("Negative volume in " + zip_path.name)

    if (data["high"] < data[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("Invalid high price in " + zip_path.name)

    if (data["low"] > data[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Invalid low price in " + zip_path.name)

    # Confirm that the file contains only the requested calendar month.
    dates_without_timezone = data["open_time"].dt.tz_localize(None)
    months_in_file = dates_without_timezone.dt.to_period("M")
    requested_month = pd.Period(month, freq="M")

    if not (months_in_file == requested_month).all():
        raise ValueError("Wrong month inside " + zip_path.name)

    # The last column is unused, so we do not need it in the clean data.
    data = data.drop(columns=["ignore"])
    return data


# ---------------------------
# 4. Combine all months
# ---------------------------

def process_one_symbol(symbol):
    """Download and combine all months for one trading pair."""

    monthly_data = []
    source_rows = []
    months = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")

    for month_period in months:
        month = str(month_period)

        zip_path, source_information = download_one_month(symbol, month)
        one_month = read_one_month(zip_path, month)

        monthly_data.append(one_month)
        source_information["rows"] = len(one_month)
        source_rows.append(source_information)

        print("Finished", symbol, month, flush=True)

    # Put the 72 monthly tables into one large table.
    data = pd.concat(monthly_data, ignore_index=True)
    data = data.sort_values("open_time").reset_index(drop=True)
    data.insert(0, "symbol", symbol)

    # Make sure timestamps were not repeated between two monthly files.
    repeated_times = int(data["open_time"].duplicated().sum())
    if repeated_times > 0:
        raise ValueError("Repeated timestamps found for " + symbol)

    # Create the complete calendar that should exist in a perfect dataset.
    expected_times = pd.date_range(
        "2020-01-01 00:00:00+00:00",
        "2025-12-31 23:55:00+00:00",
        freq="5min",
    )
    actual_times = pd.DatetimeIndex(data["open_time"])

    missing_times = expected_times.difference(actual_times)
    extra_times = actual_times.difference(expected_times)

    audit_row = {
        "symbol": symbol,
        "first_time": actual_times.min().isoformat(),
        "last_time": actual_times.max().isoformat(),
        "actual_rows": len(data),
        "expected_rows": len(expected_times),
        "missing_times": len(missing_times),
        "extra_times": len(extra_times),
        "repeated_times": repeated_times,
    }

    missing_table = pd.DataFrame()
    missing_table["missing_open_time"] = missing_times
    missing_table.insert(0, "symbol", symbol)

    # Save the combined clean table for this asset.
    CLEAN_FOLDER.mkdir(parents=True, exist_ok=True)
    output_name = symbol + "_5m_2020_2025.csv.gz"
    output_path = CLEAN_FOLDER / output_name

    data.to_csv(
        output_path,
        index=False,
        compression="gzip",
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )

    print("Saved", output_path, flush=True)
    return source_rows, audit_row, missing_table


# ---------------------------
# 5. Run the complete program
# ---------------------------

def main():
    """Run the extraction for both assets and save the check files."""

    RAW_FOLDER.mkdir(parents=True, exist_ok=True)
    CLEAN_FOLDER.mkdir(parents=True, exist_ok=True)
    CHECK_FOLDER.mkdir(parents=True, exist_ok=True)

    all_source_rows = []
    all_audit_rows = []
    all_missing_tables = []

    for symbol in SYMBOLS:
        source_rows, audit_row, missing_table = process_one_symbol(symbol)

        all_source_rows.extend(source_rows)
        all_audit_rows.append(audit_row)
        all_missing_tables.append(missing_table)

    source_table = pd.DataFrame(all_source_rows)
    source_table.to_csv(CHECK_FOLDER / "source_files.csv", index=False)

    audit_table = pd.DataFrame(all_audit_rows)
    audit_table.to_csv(CHECK_FOLDER / "data_audit.csv", index=False)

    missing_table = pd.concat(all_missing_tables, ignore_index=True)
    missing_table.to_csv(
        CHECK_FOLDER / "missing_times.csv",
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )

    print("Data extraction is complete.", flush=True)


if __name__ == "__main__":
    main()
