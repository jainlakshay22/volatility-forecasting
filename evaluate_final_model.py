"""Refit the selected model and evaluate it on the 2025 test period."""

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
BASELINE_FOLDER = PROJECT_FOLDER / "results/baselines"
RESULTS_FOLDER = PROJECT_FOLDER / "results/final_evaluation"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SELECTED_MODEL = "Linear HAR"

FEATURE_COLUMNS = [
    "daily_log_variance",
    "weekly_log_variance",
    "monthly_log_variance",
]
TARGET_COLUMN = "target_log_variance"


# ---------------------------
# 2. Read saved data
# ---------------------------

def read_split(symbol, split_name):
    """Read one saved chronological split."""

    file_path = SPLIT_FOLDER / (symbol + "_" + split_name + ".csv")

    if not file_path.exists():
        raise FileNotFoundError("Could not find " + str(file_path))

    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    return data


def confirm_model_selection(symbol):
    """Confirm that Linear HAR won using validation data only."""

    validation_score_path = BASELINE_FOLDER / "validation_scores.csv"
    validation_scores = pd.read_csv(validation_score_path)
    symbol_scores = validation_scores[validation_scores["symbol"] == symbol].copy()
    symbol_scores = symbol_scores.sort_values("qlike").reset_index(drop=True)

    best_model = symbol_scores.loc[0, "model"]
    best_score = symbol_scores.loc[0, "qlike"]

    if best_model != SELECTED_MODEL:
        raise ValueError(
            "The frozen model does not match the best validation model for " + symbol
        )

    selection_row = {
        "symbol": symbol,
        "selected_model": SELECTED_MODEL,
        "selection_metric": "validation QLIKE",
        "validation_qlike": best_score,
        "validation_end": symbol_scores.loc[0, "validation_end"],
        "selected_before_test": True,
    }

    return selection_row


# ---------------------------
# 3. Forecast measurements
# ---------------------------

def calculate_scores(actual_log, forecast_log):
    """Calculate log-MSE, log-MAE, and QLIKE."""

    difference = actual_log - forecast_log
    log_mse = np.mean(difference ** 2)
    log_mae = np.mean(np.abs(difference))

    actual_variance = np.exp(actual_log)
    forecast_variance = np.exp(forecast_log)
    ratio = actual_variance / forecast_variance
    qlike = np.mean(ratio - np.log(ratio) - 1)

    return log_mse, log_mae, qlike


def make_forecast_table(symbol, model_name, test, forecast_log):
    """Create a table containing actual and forecast test values."""

    table = pd.DataFrame()
    table["date"] = test["date"]
    table["symbol"] = symbol
    table["model"] = model_name
    table["actual_log_variance"] = test[TARGET_COLUMN]
    table["forecast_log_variance"] = forecast_log
    table["actual_variance"] = test["realized_variance"]
    table["forecast_variance"] = np.exp(forecast_log)
    table["actual_volatility_percent"] = np.sqrt(table["actual_variance"]) * 100
    table["forecast_volatility_percent"] = (
        np.sqrt(table["forecast_variance"]) * 100
    )
    return table


def make_score_row(symbol, model_name, fitting_data, test, forecast_log):
    """Create one row of final test scores."""

    actual_log = test[TARGET_COLUMN].to_numpy()
    log_mse, log_mae, qlike = calculate_scores(actual_log, forecast_log)

    row = {
        "symbol": symbol,
        "model": model_name,
        "fit_start": fitting_data["date"].min().date(),
        "fit_end": fitting_data["date"].max().date(),
        "fit_rows": len(fitting_data),
        "test_start": test["date"].min().date(),
        "test_end": test["date"].max().date(),
        "test_rows": len(test),
        "log_mse": log_mse,
        "log_mae": log_mae,
        "qlike": qlike,
    }
    return row


# ---------------------------
# 4. Refit and test the models
# ---------------------------

def evaluate_one_symbol(symbol):
    """Refit with data through 2024 and forecast the 2025 test period."""

    train = read_split(symbol, "train")
    validation = read_split(symbol, "validation")
    test = read_split(symbol, "test")

    # Model selection is complete, so validation can now join the fitting data.
    fitting_data = pd.concat([train, validation], ignore_index=True)
    fitting_data = fitting_data.sort_values("date").reset_index(drop=True)

    if fitting_data["date"].max() >= test["date"].min():
        raise ValueError("Fitting data overlaps the test period")

    forecast_tables = []
    score_rows = []

    persistence_forecast = test["daily_log_variance"].to_numpy()
    weekly_forecast = test["weekly_log_variance"].to_numpy()

    fitting_average = fitting_data[TARGET_COLUMN].mean()
    historical_mean_forecast = np.full(len(test), fitting_average)

    har_model = LinearRegression()
    har_model.fit(fitting_data[FEATURE_COLUMNS], fitting_data[TARGET_COLUMN])
    har_forecast = har_model.predict(test[FEATURE_COLUMNS])

    model_forecasts = {
        "Persistence": persistence_forecast,
        "Five-day average": weekly_forecast,
        "Historical mean": historical_mean_forecast,
        "Linear HAR": har_forecast,
    }

    for model_name, forecast_log in model_forecasts.items():
        forecast_tables.append(
            make_forecast_table(symbol, model_name, test, forecast_log)
        )
        score_rows.append(
            make_score_row(
                symbol,
                model_name,
                fitting_data,
                test,
                forecast_log,
            )
        )

    coefficient_rows = [
        {
            "symbol": symbol,
            "term": "intercept",
            "coefficient": har_model.intercept_,
        }
    ]

    for feature_name, coefficient in zip(FEATURE_COLUMNS, har_model.coef_):
        coefficient_rows.append(
            {
                "symbol": symbol,
                "term": feature_name,
                "coefficient": coefficient,
            }
        )

    forecasts = pd.concat(forecast_tables, ignore_index=True)
    scores = pd.DataFrame(score_rows)
    coefficients = pd.DataFrame(coefficient_rows)

    return forecasts, scores, coefficients


# ---------------------------
# 5. Create readable test charts
# ---------------------------

def add_rolling_averages(data):
    """Add seven-day averages for charting only."""

    data = data.sort_values("date").copy()
    data["actual_7_day_average"] = data[
        "actual_volatility_percent"
    ].rolling(7, min_periods=1).mean()
    data["forecast_7_day_average"] = data[
        "forecast_volatility_percent"
    ].rolling(7, min_periods=1).mean()
    return data


def draw_model(axis, data, model_name, show_daily):
    """Draw one model against actual test volatility."""

    if show_daily:
        axis.plot(
            data["date"],
            data["actual_volatility_percent"],
            color="gray",
            linewidth=0.6,
            alpha=0.25,
            label="Actual daily",
        )
        axis.plot(
            data["date"],
            data["forecast_volatility_percent"],
            color="cornflowerblue",
            linewidth=0.6,
            alpha=0.25,
            label="Forecast daily",
        )

    axis.plot(
        data["date"],
        data["actual_7_day_average"],
        color="black",
        linewidth=1.8,
        label="Actual: 7-day average",
    )
    axis.plot(
        data["date"],
        data["forecast_7_day_average"],
        color="crimson",
        linewidth=1.8,
        label="Forecast: 7-day average",
    )
    axis.set_title(model_name)
    axis.set_xlabel("Date")
    axis.set_ylabel("Daily volatility (%)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)


def plot_test_results(symbol, forecasts):
    """Save a four-model dashboard and a selected-model chart."""

    symbol_data = forecasts[forecasts["symbol"] == symbol].copy()
    model_names = list(symbol_data["model"].unique())

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    flat_axes = axes.flatten()

    for position in range(len(model_names)):
        model_name = model_names[position]
        one_model = symbol_data[symbol_data["model"] == model_name]
        one_model = add_rolling_averages(one_model)
        draw_model(flat_axes[position], one_model, model_name, show_daily=False)

    figure.suptitle(
        symbol + " Final Test Forecasts: 7-Day Averages (2025)",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(
        RESULTS_FOLDER / (symbol + "_test_dashboard.png"),
        dpi=150,
    )
    plt.close(figure)

    selected_data = symbol_data[symbol_data["model"] == SELECTED_MODEL]
    selected_data = add_rolling_averages(selected_data)

    figure, axis = plt.subplots(figsize=(12, 5))
    draw_model(axis, selected_data, SELECTED_MODEL, show_daily=True)
    figure.suptitle(symbol + " Final Test Forecast (2025)")
    figure.tight_layout()
    figure.savefig(
        RESULTS_FOLDER / (symbol + "_selected_model_test.png"),
        dpi=150,
    )
    plt.close(figure)


# ---------------------------
# 6. Run the final evaluation
# ---------------------------

def main():
    """Run one frozen-model test evaluation for both assets."""

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    selection_rows = []
    forecast_tables = []
    score_tables = []
    coefficient_tables = []

    for symbol in SYMBOLS:
        selection_rows.append(confirm_model_selection(symbol))
        forecasts, scores, coefficients = evaluate_one_symbol(symbol)
        forecast_tables.append(forecasts)
        score_tables.append(scores)
        coefficient_tables.append(coefficients)

    selections = pd.DataFrame(selection_rows)
    all_forecasts = pd.concat(forecast_tables, ignore_index=True)
    all_scores = pd.concat(score_tables, ignore_index=True)
    all_coefficients = pd.concat(coefficient_tables, ignore_index=True)

    # Add an easy-to-read comparison with the persistence benchmark.
    improvement_values = []
    for row_number in range(len(all_scores)):
        row = all_scores.iloc[row_number]
        symbol_scores = all_scores[all_scores["symbol"] == row["symbol"]]
        persistence_qlike = symbol_scores[
            symbol_scores["model"] == "Persistence"
        ]["qlike"].iloc[0]
        improvement = (persistence_qlike - row["qlike"]) / persistence_qlike * 100
        improvement_values.append(improvement)

    all_scores["qlike_improvement_vs_persistence_percent"] = improvement_values

    selections.to_csv(RESULTS_FOLDER / "model_selection.csv", index=False)
    all_forecasts.to_csv(
        RESULTS_FOLDER / "test_forecasts.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    all_scores.to_csv(RESULTS_FOLDER / "test_scores.csv", index=False)
    all_coefficients.to_csv(
        RESULTS_FOLDER / "final_har_coefficients.csv", index=False
    )

    for symbol in SYMBOLS:
        plot_test_results(symbol, all_forecasts)

    print("\nFinal test scores (lower is better)")
    print(all_scores.sort_values(["symbol", "qlike"]).to_string(index=False))
    print("\nThe selected model was frozen using validation results before testing.")


if __name__ == "__main__":
    main()
