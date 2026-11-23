"""Comparación de los híbridos con SARIMA (MAE, MSE, RMSE, MAPE). Solo PDF."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


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


BASE = Path(__file__).resolve().parent
SERIES = ["15-34 M", "15-34 F", "35-49 M", "35-49 F",
          "50-74 M", "50-74 F", "Global_Rate"]
METRICS = ["MAE", "MSE", "RMSE", "MAPE"]
MODELS = {
    "SARIMA-LSTM_seq": ("SARIMA_LSTM_Sequential/sarima_lstm_test.csv",
                        r"SL$_{\mathrm{seq}}$", "#2B5F8A"),
    "SARIMA-GRU_seq": ("SARIMA_GRU_Sequential/sarima_gru_test.csv",
                       r"SG$_{\mathrm{seq}}$", "#D48B23"),
}


def read_metrics(path):
    df = pd.read_csv(path)
    if "Serie" not in df:
        df = df.rename(columns={"Unnamed: 0": "Serie"})
    return df.set_index("Serie")


def main():
    academic_style()
    results = BASE / "results"
    sarima = read_metrics(results / "SARIMA/sarima_results.csv")
    hybrids = {name: read_metrics(results / file)
               for name, (file, _, _) in MODELS.items()}

    fig, axs = plt.subplots(2, 2, figsize=(12.2, 9.25), sharey="row")
    y = np.arange(len(SERIES))

    for i, (ax, metric) in enumerate(zip(axs.flat, METRICS)):
        values = {}
        reference = sarima.loc[SERIES, metric].to_numpy(dtype=float)

        for name, hybrid in hybrids.items():
            prediction = hybrid.loc[SERIES, metric].to_numpy(dtype=float)
            #values[name] = 100 * (reference - prediction) / reference
            values[name] = 100 * (prediction - reference) / reference
            


        all_values = np.concatenate(list(values.values()))
        left = min(-10, all_values.min() - max(11, 0.16 * abs(all_values.min())))
        right = max(9, all_values.max() + 8)
        ax.set_xlim(left, right)
        pad = (right - left) * 0.016

        for k in range(len(SERIES)):
            if k % 2 == 0:
                ax.axhspan(k - 0.5, k + 0.5, color="#EDF0F2", alpha=0.58)

        for (name, (_, label, color)), offset in zip(MODELS.items(), (-0.18, 0.18)):
            ax.barh(y + offset, values[name], height=0.30, color=color, zorder=3)
            for ypos, value in zip(y + offset, values[name]):
                ax.text(value + (pad if value >= 0 else -pad), ypos,
                        f"{value:+.1f}%", ha="left" if value >= 0 else "right",
                        va="center", fontsize=8.6)

        ax.axvline(0, color="#222222", lw=1)
        ax.set_title(metric, fontsize=16, pad=9)
        ax.set_yticks(y)
        ax.set_ylim(len(SERIES) - 0.5, -0.5)
        ax.grid(axis="x", linestyle="--", linewidth=0.65, alpha=0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", length=0, labelsize=12)
        sns.despine(ax=ax, left=True)
        if i % 2 == 0:
            ax.set_yticklabels([s.replace("_", " ") for s in SERIES])
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)

    fig.legend(
        handles=[Patch(color=color, label=label)
                 for _, label, color in MODELS.values()],
        loc="upper center", bbox_to_anchor=(0.5, 0.975),
        ncol=2, frameon=False, fontsize=16,
    )
    fig.supxlabel(" ", fontsize=14, y=0.035)
    fig.subplots_adjust(left=0.145, right=0.982, top=0.89, bottom=0.11,
                        wspace=0.12, hspace=0.31)

    out = BASE / "IMG/hybrid_metric_impact"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hybrid_impact_all_metrics.pdf", facecolor="white")
    plt.close(fig)
    print(f"PDF guardado en: {out / 'hybrid_impact_all_metrics.pdf'}")


if __name__ == "__main__":
    main()
