# Debata data extraction

This is a new project that starts with an empty `data` folder. It does not read
files or results from any earlier project.

## What the script downloads

- Source: Binance public spot-market archive
- Assets: BTC/USDT and ETH/USDT
- Frequency: five-minute candles
- Dates: January 2020 through December 2025

## How to run it

Open a terminal in this folder and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
```

The script creates the `data` folder automatically. The first run downloads 144
monthly ZIP files and their official checksum files, so it needs an internet
connection and may take several minutes.

## Files created

```text
data/
  raw/                 Original Binance ZIP and checksum files
  clean/               Combined five-minute CSV files
  checks/
    source_files.csv   One row for each downloaded monthly file
    data_audit.csv     Data-quality summary for BTC and ETH
    missing_times.csv  Every missing five-minute timestamp
```

This stage only downloads, combines, and checks prices. It does not calculate
returns, volatility, model features, or forecasts.

## Calculate daily volatility

After `download_data.py` finishes, run:

```bash
python calculate_volatility.py
python check_volatility.py
```

`calculate_volatility.py` performs the following calculation:

```text
five-minute closing prices
    -> five-minute log returns
    -> squared log returns
    -> daily realized variance
    -> daily realized volatility
```

A return is discarded when the two prices are not exactly five minutes apart.
A day is marked incomplete unless it has 288 bars and 288 valid returns. The
script does not fill gaps or pretend that partial days are complete.

The daily results are saved as:

```text
data/daily/BTCUSDT_daily_volatility.csv
data/daily/ETHUSDT_daily_volatility.csv
```

The volatility values are daily values and are not annualized.

## Explore the volatility data

After the daily files pass their checks, run:

```bash
pip install -r requirements.txt
python explore_volatility.py
```

The script creates summary tables and charts in `results/exploration/`. It
reports complete and incomplete days, typical volatility, the most volatile
days, year-by-year statistics, persistence from one day to the next, and the
relationship between Bitcoin and Ethereum volatility.
