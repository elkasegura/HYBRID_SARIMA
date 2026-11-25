"""Forecasting plots for seven unemployment series.

Use in the existing project:
    python forecasting_series.py --base /home/lka/HYBRID_SARIMA

The Observed line uses the original blue #2B5F8A.
Outputs a combined 7-panel figure, a historical-context figure,
and 7 individual series plots, all as PDF + PNG.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


SERIES = ["15-34 M", "15-34 F", "35-49 M", "35-49 F",
          "50-74 M", "50-74 F", "Global_Rate"]

MODEL_PATHS = {
    "SARIMA": ("SARIMA/sarima_predictions.csv", None),
    "LSTM": ("LSTM/lstm_predictions.csv", None),
    "GRU": ("GRU/gru_predictions.csv", None),
    "SARIMA-LSTM_seq": ("SARIMA_LSTM_Sequential", "Hybrid"),
    "SARIMA-GRU_seq": ("SARIMA_GRU_Sequential", "Hybrid"),
}

# Original plot blue, used for the observed series.
COLORS = {
    "Observed": "#2B5F8A",
    "SARIMA": "#D48B23",
    "LSTM": "#238B59",
    "GRU": "#C24B5A",
    "SARIMA-LSTM_seq": "#7453A6",
    "SARIMA-GRU_seq": "#169AA6",
}

# Abbreviated model names
model_names = {
    "SARIMA": "SARIMA",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "SARIMA-LSTM_seq": r"SL$_{seq}$",
    "SARIMA-GRU_seq": r"SG$_{seq}$"
}

# Mapping from model keys to their abbreviated display names for plots.


def load_forecasts(base):
    results = base / "results"
    historical = pd.read_csv(base / "data.csv", encoding="utf-8-sig")
    historical["Period"] = historical["Period"].astype(str).str.strip()
    historical = historical.set_index("Period")

    predictions = {series: {} for series in SERIES}
    for model, (path, value_column) in MODEL_PATHS.items():
        if value_column is None:
            data = pd.read_csv(results / path, index_col=0)
            data.columns = [str(i) for i in range(data.shape[1])]
            for series in SERIES:
                predictions[series][model] = data.loc[series].to_numpy(dtype=float)
        else:
            for series in SERIES:
                file = f"pred_{series.replace(' ', '_')}.csv"
                data = pd.read_csv(results / path / file)
                data["Period"] = data["Period"].astype(str).str.strip()
                predictions[series][model] = data[value_column].to_numpy(dtype=float)
                if model.startswith("SARIMA-LSTM"):
                    forecast_periods = data["Period"].tolist()
                    # The observed values must match the source time series.
                    np.testing.assert_allclose(
                        historical.loc[forecast_periods, series].to_numpy(dtype=float),
                        data["Observed"].to_numpy(dtype=float), rtol=1e-5, atol=1e-5,
                    )

    for series in SERIES:
        for model, values in predictions[series].items():
            if len(values) != len(forecast_periods):
                raise ValueError(f"Unexpected number of test points in {series}, {model}")
    return historical, forecast_periods, predictions


def academic_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })


def draw_forecast(ax, series, historical, periods, predictions, *, context=False):
    all_periods = historical.index.tolist()
    test_x = np.array([all_periods.index(p) for p in periods])

    if context:
        start = max(0, all_periods.index(periods[0]) - 12) # 3 years before test
        blue_x = np.arange(start, test_x[-1] + 1)
        blue_y = historical[series].iloc[start:test_x[-1] + 1]
    else:
        blue_x = test_x
        blue_y = historical.loc[periods, series]

    # Blue observed line: same color as in the user's Python figure.
    ax.plot(blue_x, blue_y, color=COLORS["Observed"], marker="o",
            markersize=3.5, linewidth=2.1, label="Observed", zorder=5)

    for model, values in predictions[series].items():
        ax.plot(test_x, values, color=COLORS[model], linestyle="--",
                linewidth=1.45, alpha=0.91, label=model_names.get(model, model), zorder=3)

    ax.set_title(series.replace("_", " "), fontsize=14, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.55)
    sns.despine(ax=ax)

    if context:
        first = max(0, all_periods.index(periods[0]) - 12)
        tick_x = np.arange(first, test_x[-1] + 1, 4)
        ax.axvline(test_x[0], linestyle=":", linewidth=1.0, color="#8D9296", zorder=1)
    else:
        tick_x = test_x[::2]
    ax.set_xticks(tick_x)
    ax.set_xticklabels([all_periods[j] for j in tick_x], rotation=35,
                       ha="right", rotation_mode="anchor")


def legend_handles():
    return [Line2D([0], [0], color=color, lw=2.1 if name == "Observed" else 1.6,
                   linestyle="-" if name == "Observed" else "--",
                   marker="o" if name == "Observed" else None,
                   markersize=3.5, label=model_names.get(name, name))
            for name, color in COLORS.items()]


def seven_panel(base, out, historical, periods, predictions, *, context=False):
    fig = plt.figure(figsize=(12.3, 12.5))
    gs = fig.add_gridspec(4, 2, hspace=0.64, wspace=0.27)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(6)]
    axes.append(fig.add_subplot(gs[3, :]))
    for ax, series in zip(axes, SERIES):
        draw_forecast(ax, series, historical, periods, predictions, context=context)
    fig.supylabel("Unemployment rate (%)", fontsize=14, x=0.03)
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, fontsize=14, columnspacing=2.2)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.01)
    stem = "forecasting_7_series_context" if context else "forecasting_7_series_test"
    fig.savefig(out / f"{stem}.pdf")
    #fig.savefig(out / f"{stem}.png", dpi=240)
    plt.close(fig)


def individual_plots(out, historical, periods, predictions):
    for series in SERIES:
        fig, ax = plt.subplots(figsize=(9.8, 4.5))
        draw_forecast(ax, series, historical, periods, predictions, context=False)
        ax.set_ylabel("Unemployment rate (%)", fontsize=14)
        ax.set_xlabel("Quarter", fontsize=14)
        fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.995),
                   ncol=3, frameon=False, fontsize=14)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.75, bottom=0.22)
        stem = f"forecast_{series.replace(' ', '_')}"
        fig.savefig(out / f"{stem}.pdf")
        #fig.savefig(out / f"{stem}.png", dpi=240)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder containing data.csv and results/")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    base = args.base.expanduser().resolve()
    out = args.out or (base / "IMG" / "forecasting")
    out.mkdir(parents=True, exist_ok=True)
    academic_style()
    historical, periods, predictions = load_forecasts(base)
    seven_panel(base, out, historical, periods, predictions, context=False)
    seven_panel(base, out, historical, periods, predictions, context=True)
    individual_plots(out, historical, periods, predictions)
    print(f"Created combined and 7 individual plots in {out}")


if __name__ == "__main__":
    main()