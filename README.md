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

## Create forecasting features

After exploring the daily data, run:

```bash
python create_features.py
python check_features.py
```

The feature script creates three past-only inputs for each target date:

- the previous day's log realized variance;
- the average log realized variance over the previous 5 calendar days;
- the average log realized variance over the previous 22 calendar days.

The shift is applied before the rolling averages are calculated, so today's
realized variance cannot enter today's predictors. Missing calendar days remain
missing and invalidate any rolling window that contains them. The results are
saved under `data/features/`.

## Split the modelling data

After the feature checks pass, run:

```bash
python split_data.py
python check_splits.py
```

The split is chronological rather than random:

- training data ends on 31 December 2023;
- validation data covers 1 January through 31 December 2024;
- test data covers 1 January through 31 December 2025.

Training data is used to fit models. Validation data is used to compare model
choices. Test data is reserved for the final evaluation and should not be used
to select features or tune models. The split files are saved under
`data/splits/`.

## Train baseline models

Install the updated requirements and run:

```bash
pip install -r requirements.txt
python train_baselines.py
python check_baselines.py
```

The script fits models only on the training period and evaluates them only on
the 2024 validation period. It compares persistence, a five-day average, the
historical training mean, and a linear HAR model. Results are saved under
`results/baselines/`. The 2025 test files are not read during this stage.

The plotting section creates one chart for each model and a four-panel summary
for each asset. Faint lines show the original daily values, while darker lines
show seven-day rolling averages to make the comparison easier to read. This
smoothing is used only in the charts; all reported scores still use the
original daily forecasts.

## Run the final test evaluation

After choosing the model from validation results, run:

```bash
python evaluate_final_model.py
python check_final_evaluation.py
```

Linear HAR is frozen as the selected model because it had the lowest 2024
validation QLIKE for both assets. It is refitted using the training and
validation periods together, ending on 31 December 2024, and evaluated once on
the untouched 2025 test period. The same three benchmarks are included for
context. Final tables and charts are saved under `results/final_evaluation/`.
