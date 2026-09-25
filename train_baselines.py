"""Train simple volatility models and evaluate them on validation data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------
# 1. Project settings
# ---------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
SPLIT_FOLDER = PROJECT_FOLDER / "data/splits"
RESULTS_FOLDER = PROJECT_FOLDER / "results/baselines"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

FEATURE_COLUMNS = [
    "daily_log_variance",
    "weekly_log_variance",
    "monthly_log_variance",
]
TARGET_COLUMN = "target_log_variance"


# ---------------------------
# 2. Read training and validation data
# ---------------------------

def read_split(symbol, split_name):
    """Read one chronological data split."""

    file_name = symbol + "_" + split_name + ".csv"
    file_path = SPLIT_FOLDER / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            "Could not find " + str(file_path) + ". Run split_data.py first."
        )

    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    columns_to_check = FEATURE_COLUMNS + [TARGET_COLUMN, "realized_variance"]
    if data[columns_to_check].isna().any().any():
        raise ValueError("Missing modelling value in " + file_name)

    return data


# ---------------------------
# 3. Evaluation measurements
# ---------------------------

def calculate_scores(actual_log, forecast_log):
    """Calculate three forecast-error measurements."""

    difference = actual_log - forecast_log

    log_mse = np.mean(difference ** 2)
    log_mae = np.mean(np.abs(difference))

    # Convert log variance back to ordinary variance for QLIKE.
    actual_variance = np.exp(actual_log)
    forecast_variance = np.exp(forecast_log)

    # QLIKE is designed for comparing variance forecasts. Lower is better.
    variance_ratio = actual_variance / forecast_variance
    qlike = np.mean(variance_ratio - np.log(variance_ratio) - 1)

    return log_mse, log_mae, qlike


# ---------------------------
# 4. Save forecasts from one model
# ---------------------------

def make_forecast_table(symbol, model_name, validation, forecast_log):
    """Create a readable table of actual and forecast values."""

    forecast_table = pd.DataFrame()
    forecast_table["date"] = validation["date"]
    forecast_table["symbol"] = symbol
    forecast_table["model"] = model_name
    forecast_table["actual_log_variance"] = validation[TARGET_COLUMN]
    forecast_table["forecast_log_variance"] = forecast_log
    forecast_table["actual_variance"] = validation["realized_variance"]
    forecast_table["forecast_variance"] = np.exp(forecast_log)
    forecast_table["actual_volatility_percent"] = (
        np.sqrt(forecast_table["actual_variance"]) * 100
    )
    forecast_table["forecast_volatility_percent"] = (
        np.sqrt(forecast_table["forecast_variance"]) * 100
    )

    return forecast_table


def make_score_row(symbol, model_name, train, validation, forecast_log):
    """Create one row for the model-comparison table."""

    actual_log = validation[TARGET_COLUMN].to_numpy()
    log_mse, log_mae, qlike = calculate_scores(actual_log, forecast_log)

    score_row = {
        "symbol": symbol,
        "model": model_name,
        "train_start": train["date"].min().date(),
        "train_end": train["date"].max().date(),
        "validation_start": validation["date"].min().date(),
        "validation_end": validation["date"].max().date(),
        "validation_rows": len(validation),
        "log_mse": log_mse,
        "log_mae": log_mae,
        "qlike": qlike,
    }

    return score_row


# ---------------------------
# 5. Train and compare the models
# ---------------------------

def run_one_symbol(symbol):
    """Fit the models for one asset and forecast the validation year."""

    train = read_split(symbol, "train")
    validation = read_split(symbol, "validation")

    all_forecasts = []
    all_scores = []

    # Model 1: predict that tomorrow equals yesterday.
    persistence_forecast = validation["daily_log_variance"].to_numpy()
    all_forecasts.append(
        make_forecast_table(
            symbol, "Persistence", validation, persistence_forecast
        )
    )
    all_scores.append(
        make_score_row(
            symbol, "Persistence", train, validation, persistence_forecast
        )
    )

    # Model 2: use the average of the previous five days.
    weekly_forecast = validation["weekly_log_variance"].to_numpy()
    all_forecasts.append(
        make_forecast_table(
            symbol, "Five-day average", validation, weekly_forecast
        )
    )
    all_scores.append(
        make_score_row(
            symbol, "Five-day average", train, validation, weekly_forecast
        )
    )

    # Model 3: always use the average training target.
    training_average = train[TARGET_COLUMN].mean()
    historical_mean_forecast = np.full(len(validation), training_average)
    all_forecasts.append(
        make_forecast_table(
            symbol, "Historical mean", validation, historical_mean_forecast
        )
    )
    all_scores.append(
        make_score_row(
            symbol, "Historical mean", train, validation, historical_mean_forecast
        )
    )

    # Model 4: linear HAR model using all three features.
    har_model = LinearRegression()
    har_model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    har_forecast = har_model.predict(validation[FEATURE_COLUMNS])

    all_forecasts.append(
        make_forecast_table(symbol, "Linear HAR", validation, har_forecast)
    )
    all_scores.append(
        make_score_row(symbol, "Linear HAR", train, validation, har_forecast)
    )

    coefficient_rows = []

    coefficient_rows.append(
        {
            "symbol": symbol,
            "term": "intercept",
            "coefficient": har_model.intercept_,
        }
    )

    for feature_name, coefficient in zip(FEATURE_COLUMNS, har_model.coef_):
        coefficient_rows.append(
            {
                "symbol": symbol,
                "term": feature_name,
                "coefficient": coefficient,
            }
        )

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    scores = pd.DataFrame(all_scores)
    coefficients = pd.DataFrame(coefficient_rows)

    return forecasts, scores, coefficients


# ---------------------------
# 6. Plot validation forecasts
# ---------------------------

def safe_file_name(model_name):
    """Turn a model name into a simple file name."""

    file_name = model_name.lower()
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace("-", "_")
    return file_name


def add_rolling_averages(one_model):
    """Add seven-day averages for a less noisy visual comparison."""

    one_model = one_model.sort_values("date").copy()

    one_model["actual_7_day_average"] = one_model[
        "actual_volatility_percent"
    ].rolling(window=7, min_periods=1).mean()

    one_model["forecast_7_day_average"] = one_model[
        "forecast_volatility_percent"
    ].rolling(window=7, min_periods=1).mean()

    return one_model


def draw_one_model(axis, one_model, model_name, show_daily_values):
    """Draw actual and forecast volatility for one model."""

    if show_daily_values:
        # The light lines preserve the daily observations without allowing
        # their sharp movements to dominate the chart.
        axis.plot(
            one_model["date"],
            one_model["actual_volatility_percent"],
            color="gray",
            linewidth=0.6,
            alpha=0.25,
            label="Actual daily",
        )
        axis.plot(
            one_model["date"],
            one_model["forecast_volatility_percent"],
            color="cornflowerblue",
            linewidth=0.6,
            alpha=0.25,
            label="Forecast daily",
        )

    # The darker lines show the seven-day averages and are easier to compare.
    axis.plot(
        one_model["date"],
        one_model["actual_7_day_average"],
        color="black",
        linewidth=1.8,
        label="Actual: 7-day average",
    )
    axis.plot(
        one_model["date"],
        one_model["forecast_7_day_average"],
        color="crimson",
        linewidth=1.8,
        label="Forecast: 7-day average",
    )

    axis.set_title(model_name)
    axis.set_xlabel("Date")
    axis.set_ylabel("Daily volatility (%)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)


def plot_forecasts(symbol, forecasts):
    """Save a separate chart for every model and a four-panel dashboard."""

    symbol_forecasts = forecasts[forecasts["symbol"] == symbol].copy()
    model_names = list(symbol_forecasts["model"].unique())

    # First create one large, readable chart for each individual model.
    for model_name in model_names:
        one_model = symbol_forecasts[
            symbol_forecasts["model"] == model_name
        ].copy()
        one_model = add_rolling_averages(one_model)

        figure, axis = plt.subplots(figsize=(12, 5))
        draw_one_model(axis, one_model, model_name, show_daily_values=True)
        figure.suptitle(symbol + " Validation Forecast (2024)")
        figure.tight_layout()

        model_file_name = safe_file_name(model_name)
        output_name = symbol + "_" + model_file_name + "_forecast.png"
        figure.savefig(RESULTS_FOLDER / output_name, dpi=150)
        plt.close(figure)

    # Then create a compact dashboard. Only the rolling averages are shown in
    # this figure so the four panels remain easy to read.
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    flat_axes = axes.flatten()

    for position in range(len(model_names)):
        model_name = model_names[position]
        one_model = symbol_forecasts[
            symbol_forecasts["model"] == model_name
        ].copy()
        one_model = add_rolling_averages(one_model)

        draw_one_model(
            flat_axes[position],
            one_model,
            model_name,
            show_daily_values=False,
        )

    figure.suptitle(
        symbol + " Validation Forecasts: 7-Day Averages (2024)",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(
        RESULTS_FOLDER / (symbol + "_validation_forecasts.png"),
        dpi=150,
    )
    plt.close(figure)


# ---------------------------
# 7. Run the complete comparison
# ---------------------------

def main():
    """Run all models for both assets and save validation results."""

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    forecast_tables = []
    score_tables = []
    coefficient_tables = []

    for symbol in SYMBOLS:
        print("Training baseline models for", symbol)
        forecasts, scores, coefficients = run_one_symbol(symbol)

        forecast_tables.append(forecasts)
        score_tables.append(scores)
        coefficient_tables.append(coefficients)

    all_forecasts = pd.concat(forecast_tables, ignore_index=True)
    all_scores = pd.concat(score_tables, ignore_index=True)
    all_coefficients = pd.concat(coefficient_tables, ignore_index=True)

    all_forecasts.to_csv(
        RESULTS_FOLDER / "validation_forecasts.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    all_scores.to_csv(RESULTS_FOLDER / "validation_scores.csv", index=False)
    all_coefficients.to_csv(
        RESULTS_FOLDER / "linear_har_coefficients.csv", index=False
    )

    for symbol in SYMBOLS:
        plot_forecasts(symbol, all_forecasts)

    display_scores = all_scores.sort_values(["symbol", "qlike"])
    print("\nValidation scores (lower is better)")
    print(display_scores.to_string(index=False))
    print("\nNo test data was used in this script.")


if __name__ == "__main__":
    main()
