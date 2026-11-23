"""Diebold-Mariano: imprime en pantalla dos tablas (MSE y MAPE).

Colocar en la raíz de HYBRID_SARIMA y ejecutar:
    python dm_tabla_articulo_pantalla.py

Los valores p son exploratorios: las 14 predicciones provienen de un
único origen y cubren horizontes distintos (1 a 14 pasos).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

SERIES = [
    "15-34 M", "15-34 F", "35-49 M", "35-49 F",
    "50-74 M", "50-74 F", "Global_Rate"
]
METRICS = ("MSE", "MAPE")
MODELS = ("SARIMA", "LSTM", "GRU", "SARIMA-LSTM_seq", "SARIMA-GRU_seq")

# Selección indicada por el usuario (sin cambiarla automáticamente).
BEST_MODELS = {
    "15-34 M": "SARIMA",
    "15-34 F": "SARIMA",
    "35-49 M": "SARIMA-GRU_seq",
    "35-49 F": "SARIMA-GRU_seq",
    "50-74 M": "SARIMA",
    "50-74 F": "LSTM",
    "Global_Rate": "SARIMA",
}

WIDE_FILES = {
    "SARIMA": RESULTS / "SARIMA" / "sarima_predictions.csv",
    "LSTM": RESULTS / "LSTM" / "lstm_predictions.csv",
    "GRU": RESULTS / "GRU" / "gru_predictions.csv",
}
HYBRID_FOLDERS = {
    "SARIMA-LSTM_seq": RESULTS / "SARIMA_LSTM_Sequential",
    "SARIMA-GRU_seq": RESULTS / "SARIMA_GRU_Sequential",
}

# Estimación de largo plazo con rezagos 2, como el cálculo previo.
HAC_LAGS = 2


def predictions():
    """Carga valores reales y pronósticos alineados para cada serie."""
    data = pd.read_csv(BASE / "data.csv")
    data["Period"] = data["Period"].astype(str).str.strip()
    data = data.tail(14).copy()
    periods = data["Period"].tolist()
    if len(periods) != 14 or len(set(periods)) != 14:
        raise ValueError("Se necesitan exactamente 14 períodos únicos de prueba")

    wide = {}
    for name, path in WIDE_FILES.items():
        frame = pd.read_csv(path, index_col=0)
        frame.index = frame.index.astype(str).str.strip()
        wide[name] = frame[[str(i) for i in range(14)]]

    for serie in SERIES:
        real = data[serie].to_numpy(dtype=float)
        pred = {name: frame.loc[serie].to_numpy(dtype=float)
                for name, frame in wide.items()}

        for name, folder in HYBRID_FOLDERS.items():
            path = folder / f"pred_{serie.replace(' ', '_')}.csv"
            frame = pd.read_csv(path)
            frame["Period"] = frame["Period"].astype(str).str.strip()
            frame = frame.set_index("Period").loc[periods]
            if not np.allclose(frame["Observed"].to_numpy(dtype=float), real,
                               rtol=1e-5, atol=1e-5):
                raise ValueError(f"Valores reales distintos en {path}")
            pred[name] = frame["Hybrid"].to_numpy(dtype=float)

        yield serie, real, pred


def loss(real, estimate, metric):
    errors = real - estimate
    if metric == "MSE":
        return errors ** 2
    if metric == "MAPE":
        if np.any(np.isclose(real, 0.0)):
            raise ValueError("MAPE indefinido si existen valores reales iguales a cero")
        return 100 * np.abs(errors / real)
    raise ValueError(metric)


def dm_statistic(loss_1, loss_2):
    """DM tipo HAC-Bartlett, p bilateral normal, uso SOLO exploratorio.

    Signo como en Table 7 del artículo:
    negativo -> favorece al Model 1; positivo -> favorece al Model 2.
    """
    d = np.asarray(loss_1 - loss_2, dtype=float)
    n = len(d)
    u = d - d.mean()
    long_run_var = np.dot(u, u) / n
    for lag in range(1, min(HAC_LAGS, n - 1) + 1):
        autocov = np.dot(u[lag:], u[:-lag]) / n
        long_run_var += 2 * (1 - lag / (HAC_LAGS + 1)) * autocov
    if long_run_var <= 1e-14 or not np.isfinite(long_run_var):
        return float("nan"), float("nan")
    dm = np.sqrt(n) * d.mean() / np.sqrt(long_run_var)
    p = 2 * norm.sf(abs(dm))
    return float(dm), float(p)


def significance(p):
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def p_label(p):
    if not np.isfinite(p):
        return "--"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def make_table():
    rows = []
    for serie, real, forecast in predictions():
        best = BEST_MODELS[serie]
        for metric in METRICS:
            first = loss(real, forecast[best], metric)
            for other in MODELS:
                if other == best:
                    continue
                dm, p = dm_statistic(first, loss(real, forecast[other], metric))
                rows.append({
                    "Serie": serie.replace("_", " "),
                    "Metric": metric,
                    "Model 1": best,
                    "Model 2": other,
                    "DM statistic": round(dm, 3),
                    "p value": p_label(p),
                    "Significance": significance(p),
                })
    return pd.DataFrame(rows)


def main():
    tabla = make_table()
    columnas = ["Serie", "Model 1", "Model 2", "DM statistic", "p value", "Significance"]
    for metrica in METRICS:
        vista = tabla.loc[tabla["Metric"] == metrica, columnas].copy()
        vista["DM statistic"] = vista["DM statistic"].map(lambda x: f"{x:.3f}")
        print(f"\nDiebold-Mariano (DM) — {metrica}")
        print(vista.to_string(index=False))
        print("Note: *** p < 0.001; ** p < 0.01; * p < 0.05; ns = not significant.")
        print("Negative DM favors Model 1; positive DM favors Model 2.")


if __name__ == "__main__":
    main()
