"""Estimate uncertainty around the final test-score differences."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
FINAL_FOLDER = PROJECT_FOLDER / "results/final_evaluation"
OUTPUT_FOLDER = PROJECT_FOLDER / "results/uncertainty"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SELECTED_MODEL = "Linear HAR"
BENCHMARKS = ["Persistence", "Five-day average", "Historical mean"]

# Seven-day blocks help preserve short-term dependence between nearby days.
BLOCK_LENGTH = 7
NUMBER_OF_BOOTSTRAPS = 5000
RANDOM_SEED = 42


def qlike_loss(actual_log, forecast_log):
    """Calculate one QLIKE loss value for every forecast date."""

    actual_variance = np.exp(actual_log)
    forecast_variance = np.exp(forecast_log)
    ratio = actual_variance / forecast_variance
    return ratio - np.log(ratio) - 1


def squared_log_error(actual_log, forecast_log):
    """Calculate one squared log error for every forecast date."""

    return (actual_log - forecast_log) ** 2


def bootstrap_mean_difference(loss_difference):
    """Bootstrap the mean of paired daily loss differences."""

    values = np.asarray(loss_difference)
    number_of_days = len(values)
    number_of_blocks = int(np.ceil(number_of_days / BLOCK_LENGTH))
    largest_start = number_of_days - BLOCK_LENGTH

    random_generator = np.random.default_rng(RANDOM_SEED)
    bootstrap_means = []

    for bootstrap_number in range(NUMBER_OF_BOOTSTRAPS):
        sampled_blocks = []

        for block_number in range(number_of_blocks):
            start = random_generator.integers(0, largest_start + 1)
            end = start + BLOCK_LENGTH
            sampled_blocks.append(values[start:end])

        bootstrap_sample = np.concatenate(sampled_blocks)
        bootstrap_sample = bootstrap_sample[:number_of_days]
        bootstrap_means.append(bootstrap_sample.mean())

    bootstrap_means = np.asarray(bootstrap_means)
    lower_limit = np.percentile(bootstrap_means, 2.5)
    upper_limit = np.percentile(bootstrap_means, 97.5)
    probability_of_improvement = np.mean(bootstrap_means > 0)

    return lower_limit, upper_limit, probability_of_improvement


def compare_models(symbol, benchmark_name, forecasts, loss_name):
    """Compare Linear HAR with one benchmark using paired dates."""

    selected = forecasts[
        (forecasts["symbol"] == symbol)
        & (forecasts["model"] == SELECTED_MODEL)
    ].sort_values("date").reset_index(drop=True)

    benchmark = forecasts[
        (forecasts["symbol"] == symbol)
        & (forecasts["model"] == benchmark_name)
    ].sort_values("date").reset_index(drop=True)

    if not selected["date"].equals(benchmark["date"]):
        raise ValueError("Forecast dates do not match for paired comparison")

    actual_log = selected["actual_log_variance"].to_numpy()
    selected_log = selected["forecast_log_variance"].to_numpy()
    benchmark_log = benchmark["forecast_log_variance"].to_numpy()

    if loss_name == "QLIKE":
        selected_loss = qlike_loss(actual_log, selected_log)
        benchmark_loss = qlike_loss(actual_log, benchmark_log)
    elif loss_name == "log MSE":
        selected_loss = squared_log_error(actual_log, selected_log)
        benchmark_loss = squared_log_error(actual_log, benchmark_log)
    else:
        raise ValueError("Unknown loss name: " + loss_name)

    # Positive differences mean Linear HAR has the lower loss.
    paired_difference = benchmark_loss - selected_loss
    mean_difference = paired_difference.mean()
    lower_limit, upper_limit, probability = bootstrap_mean_difference(
        paired_difference
    )

    percentage_improvement = (
        (benchmark_loss.mean() - selected_loss.mean())
        / benchmark_loss.mean()
        * 100
    )

    result = {
        "symbol": symbol,
        "selected_model": SELECTED_MODEL,
        "benchmark": benchmark_name,
        "loss": loss_name,
        "test_days": len(selected),
        "selected_mean_loss": selected_loss.mean(),
        "benchmark_mean_loss": benchmark_loss.mean(),
        "mean_loss_improvement": mean_difference,
        "percentage_improvement": percentage_improvement,
        "confidence_interval_lower": lower_limit,
        "confidence_interval_upper": upper_limit,
        "bootstrap_probability_improvement": probability,
        "confidence_interval_excludes_zero": lower_limit > 0,
    }

    return result


def plot_cumulative_qlike(symbol, forecasts):
    """Plot cumulative QLIKE advantage over persistence."""

    selected = forecasts[
        (forecasts["symbol"] == symbol)
        & (forecasts["model"] == SELECTED_MODEL)
    ].sort_values("date").reset_index(drop=True)

    persistence = forecasts[
        (forecasts["symbol"] == symbol)
        & (forecasts["model"] == "Persistence")
    ].sort_values("date").reset_index(drop=True)

    actual_log = selected["actual_log_variance"].to_numpy()
    selected_loss = qlike_loss(
        actual_log, selected["forecast_log_variance"].to_numpy()
    )
    persistence_loss = qlike_loss(
        actual_log, persistence["forecast_log_variance"].to_numpy()
    )

    # An upward line means Linear HAR is gaining against persistence.
    cumulative_advantage = np.cumsum(persistence_loss - selected_loss)

    plt.figure(figsize=(12, 5))
    plt.plot(selected["date"], cumulative_advantage, color="navy", linewidth=1.5)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(symbol + " Cumulative QLIKE Advantage of Linear HAR")
    plt.xlabel("Date")
    plt.ylabel("Persistence loss minus Linear HAR loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FOLDER / (symbol + "_cumulative_qlike_advantage.png"), dpi=150
    )
    plt.close()


def main():
    """Compare Linear HAR with every benchmark for both assets."""

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    forecasts = pd.read_csv(FINAL_FOLDER / "test_forecasts.csv")
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    results = []

    for symbol in SYMBOLS:
        for benchmark_name in BENCHMARKS:
            results.append(
                compare_models(symbol, benchmark_name, forecasts, "QLIKE")
            )
            results.append(
                compare_models(symbol, benchmark_name, forecasts, "log MSE")
            )

        plot_cumulative_qlike(symbol, forecasts)

    result_table = pd.DataFrame(results)
    result_table.to_csv(OUTPUT_FOLDER / "paired_uncertainty.csv", index=False)

    settings = {
        "method": "paired moving-block bootstrap",
        "block_length_days": BLOCK_LENGTH,
        "number_of_bootstraps": NUMBER_OF_BOOTSTRAPS,
        "random_seed": RANDOM_SEED,
        "confidence_level_percent": 95,
        "interpretation": (
            "Positive loss differences mean Linear HAR has lower loss. "
            "Intervals excluding zero provide stronger evidence of improvement."
        ),
    }
    settings_path = OUTPUT_FOLDER / "uncertainty_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    print(result_table.to_string(index=False))
    print("\nUncertainty results were saved in", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
