"""Compare additional models using training and 2024 validation data only."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


# ---------------------------
# 1. Project settings
# ---------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
SPLIT_FOLDER = PROJECT_FOLDER / "data/splits"
RESULTS_FOLDER = PROJECT_FOLDER / "results/additional_models"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FEATURE_COLUMNS = [
    "daily_log_variance",
    "weekly_log_variance",
    "monthly_log_variance",
]
TARGET_COLUMN = "target_log_variance"
RANDOM_SEED = 42


# ---------------------------
# 2. Read development data
# ---------------------------

def read_split(symbol, split_name):
    """Read training or validation data without reading the test file."""

    if split_name not in ["train", "validation"]:
        raise ValueError("This script may read only train or validation data")

    file_path = SPLIT_FOLDER / (symbol + "_" + split_name + ".csv")
    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    columns_to_check = FEATURE_COLUMNS + [TARGET_COLUMN, "realized_variance"]
    if data[columns_to_check].isna().any().any():
        raise ValueError("Missing model value in " + file_path.name)

    return data


# ---------------------------
# 3. Define the models
# ---------------------------

def make_models():
    """Create models with fixed settings chosen before validation scoring."""

    models = {
        "Linear HAR": LinearRegression(),
        "Ridge HAR": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
            loss="squared_error",
        ),
    }

    return models


# ---------------------------
# 4. Scores and forecast tables
# ---------------------------

def calculate_scores(actual_log, forecast_log):
    """Calculate log-MSE, log-MAE, and QLIKE."""

    difference = actual_log - forecast_log
    log_mse = np.mean(difference ** 2)
    log_mae = np.mean(np.abs(difference))

    ratio = np.exp(actual_log) / np.exp(forecast_log)
    qlike = np.mean(ratio - np.log(ratio) - 1)

    return log_mse, log_mae, qlike


def make_forecast_table(symbol, model_name, validation, forecast_log):
    """Create a table with actual and forecast validation values."""

    table = pd.DataFrame()
    table["date"] = validation["date"]
    table["symbol"] = symbol
    table["model"] = model_name
    table["actual_log_variance"] = validation[TARGET_COLUMN]
    table["forecast_log_variance"] = forecast_log
    table["actual_variance"] = validation["realized_variance"]
    table["forecast_variance"] = np.exp(forecast_log)
    table["actual_volatility_percent"] = np.sqrt(table["actual_variance"]) * 100
    table["forecast_volatility_percent"] = (
        np.sqrt(table["forecast_variance"]) * 100
    )
    return table


def make_score_row(symbol, model_name, train, validation, forecast_log):
    """Create one validation-score row."""

    actual_log = validation[TARGET_COLUMN].to_numpy()
    log_mse, log_mae, qlike = calculate_scores(actual_log, forecast_log)

    row = {
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
    return row


# ---------------------------
# 5. Fit one asset
# ---------------------------

def train_one_symbol(symbol):
    """Fit all four models and forecast the 2024 validation period."""

    train = read_split(symbol, "train")
    validation = read_split(symbol, "validation")

    training_features = train[FEATURE_COLUMNS]
    training_target = train[TARGET_COLUMN]
    validation_features = validation[FEATURE_COLUMNS]

    forecast_tables = []
    score_rows = []
    importance_rows = []

    models = make_models()

    for model_name, model in models.items():
        model.fit(training_features, training_target)
        forecast_log = model.predict(validation_features)

        forecast_tables.append(
            make_forecast_table(symbol, model_name, validation, forecast_log)
        )
        score_rows.append(
            make_score_row(symbol, model_name, train, validation, forecast_log)
        )

        # Linear models have coefficients. Tree models have feature importance.
        if hasattr(model, "coef_"):
            values = model.coef_
            value_name = "coefficient"
        else:
            values = model.feature_importances_
            value_name = "feature importance"

        for feature_name, value in zip(FEATURE_COLUMNS, values):
            importance_rows.append(
                {
                    "symbol": symbol,
                    "model": model_name,
                    "feature": feature_name,
                    "value_type": value_name,
                    "value": value,
                }
            )

    forecasts = pd.concat(forecast_tables, ignore_index=True)
    scores = pd.DataFrame(score_rows)
    importance = pd.DataFrame(importance_rows)

    return forecasts, scores, importance


# ---------------------------
# 6. Clear comparison charts
# ---------------------------

def plot_models(symbol, forecasts):
    """Create one four-panel chart using seven-day averages."""

    symbol_data = forecasts[forecasts["symbol"] == symbol]
    model_names = list(symbol_data["model"].unique())

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    flat_axes = axes.flatten()

    for position in range(len(model_names)):
        model_name = model_names[position]
        data = symbol_data[symbol_data["model"] == model_name].copy()
        data = data.sort_values("date")

        actual_average = data["actual_volatility_percent"].rolling(
            7, min_periods=1
        ).mean()
        forecast_average = data["forecast_volatility_percent"].rolling(
            7, min_periods=1
        ).mean()

        axis = flat_axes[position]
        axis.plot(
            data["date"],
            actual_average,
            color="black",
            linewidth=1.8,
            label="Actual: 7-day average",
        )
        axis.plot(
            data["date"],
            forecast_average,
            color="crimson",
            linewidth=1.8,
            label="Forecast: 7-day average",
        )
        axis.set_title(model_name)
        axis.set_xlabel("Date")
        axis.set_ylabel("Daily volatility (%)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(
        symbol + " Exploratory Model Comparison (2024 Validation)",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(
        RESULTS_FOLDER / (symbol + "_additional_models.png"),
        dpi=150,
    )
    plt.close(figure)


# ---------------------------
# 7. Run the exploratory comparison
# ---------------------------

def main():
    """Train the additional models without opening the test files."""

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    all_forecasts = []
    all_scores = []
    all_importance = []

    for symbol in SYMBOLS:
        print("Training additional models for", symbol)
        forecasts, scores, importance = train_one_symbol(symbol)
        all_forecasts.append(forecasts)
        all_scores.append(scores)
        all_importance.append(importance)

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True)

    forecasts.to_csv(
        RESULTS_FOLDER / "validation_forecasts.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    scores.to_csv(RESULTS_FOLDER / "validation_scores.csv", index=False)
    importance.to_csv(
        RESULTS_FOLDER / "coefficients_and_importance.csv", index=False
    )

    for symbol in SYMBOLS:
        plot_models(symbol, forecasts)

    print("\nExploratory validation scores (lower is better)")
    print(scores.sort_values(["symbol", "qlike"]).to_string(index=False))
    print("\nNo 2025 test files were read.")
    print("These results require a new untouched period for final testing.")


if __name__ == "__main__":
    main()
