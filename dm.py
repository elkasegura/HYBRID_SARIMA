"""DM / DM-HLN con statsmodels, para las 7 series de HYBRID_SARIMA.

Instalación:
    python -m pip install -U "statsmodels>=0.15.1" pandas numpy

Ejecución desde cualquier directorio:
    python /ruta/HYBRID_SARIMA/dm_statsmodels_pantalla.py

Muestra dos tablas (MSE y MAPE) solamente en la terminal.
No escribe CSV, PDF ni otros archivos.

Nota: en este proyecto las 14 predicciones son horizontes 1..14 desde
un único origen. HORIZON=1 es una referencia para la corrección HLN;
los valores p no constituyen automáticamente una inferencia formal
sobre pronósticos de horizonte fijo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import diebold_mariano_test

HAC_LAGS = 1           # 0, 1, 2, ... Rezagos en la varianza HAC
HORIZON = 1           # h utilizado en la corrección HLN
HARVEY_ADJ = False      # True: DM-HLN; False: DM original

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

SERIES = (
    "15-34 M", "15-34 F", "35-49 M", "35-49 F",
    "50-74 M", "50-74 F", "Global_Rate",
)
MODELS = ("SARIMA", "LSTM", "GRU", "SARIMA-LSTM_seq", "SARIMA-GRU_seq")
BEST_MODELS = {
    "15-34 M": "SARIMA",
    "15-34 F": "SARIMA",
    "35-49 M": "SARIMA-GRU_seq",
    "35-49 F": "SARIMA-GRU_seq",
    "50-74 M": "SARIMA",
    "50-74 F": "LSTM",
    "Global_Rate": "SARIMA",
}

PREDICTIONS_WIDE = {
    "SARIMA": RESULTS / "SARIMA" / "sarima_predictions.csv",
    "LSTM": RESULTS / "LSTM" / "lstm_predictions.csv",
    "GRU": RESULTS / "GRU" / "gru_predictions.csv",
}
PREDICTIONS_HYBRID = {
    "SARIMA-LSTM_seq": RESULTS / "SARIMA_LSTM_Sequential",
    "SARIMA-GRU_seq": RESULTS / "SARIMA_GRU_Sequential",
}


def load_data():
    """Obtiene valores reales y pronósticos alineados (14 por serie)."""
    data = pd.read_csv(BASE / "data.csv")
    data["Period"] = data["Period"].astype(str).str.strip()
    data = data.tail(14).copy()
    periods = data["Period"].tolist()
    if len(periods) != 14 or len(set(periods)) != 14:
        raise ValueError("data.csv debe contener 14 períodos de prueba distintos")

    wide = {}
    for model, file in PREDICTIONS_WIDE.items():
        frame = pd.read_csv(file, index_col=0)
        frame.index = frame.index.astype(str).str.strip()
        wide[model] = frame.loc[:, [str(i) for i in range(14)]]

    for serie in SERIES:
        actual = data[serie].to_numpy(dtype=float)
        forecasts = {
            model: frame.loc[serie].to_numpy(dtype=float)
            for model, frame in wide.items()
        }

        for model, folder in PREDICTIONS_HYBRID.items():
            file = folder / f"pred_{serie.replace(' ', '_')}.csv"
            frame = pd.read_csv(file)
            frame["Period"] = frame["Period"].astype(str).str.strip()
            if frame["Period"].duplicated().any():
                raise ValueError(f"Períodos duplicados: {file}")
            frame = frame.set_index("Period").loc[periods]
            observed = frame["Observed"].to_numpy(dtype=float)
            if not np.allclose(observed, actual, rtol=1e-5, atol=1e-5):
                raise ValueError(f"Valores reales no coinciden: {file}")
            forecasts[model] = frame["Hybrid"].to_numpy(dtype=float)

        if not np.isfinite(actual).all() or any(
            not np.isfinite(p).all() for p in forecasts.values()
        ):
            raise ValueError(f"Hay datos NaN/inf para {serie}")

        yield serie, actual, forecasts


def p_label(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def significance(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    rows = {"MSE": [], "MAPE": []}

    for serie, actual, forecasts in load_data():
        model_1 = BEST_MODELS[serie]
        for metric in ("MSE", "MAPE"):
            for model_2 in MODELS:
                if model_2 == model_1:
                    continue

                # Todo el cálculo DM, HAC y HLN lo hace statsmodels.
                try:
                    res = diebold_mariano_test(
                        actual,
                        forecasts[model_1],
                        forecasts[model_2],
                        criterion=metric.lower(),
                        lags=HAC_LAGS,
                        horizon=HORIZON,
                        harvey_adj=HARVEY_ADJ
                        
                    )
                    stat, p = float(res.statistic), float(res.pvalue)
                    stat_label = f"{stat:.3f}"
                    p_text = p_label(p)
                    sig = significance(p)
                except ValueError as exc:
                    stat_label, p_text, sig = "--", "--", "--"
                    print(f"Aviso ({serie}, {model_1} vs {model_2}, {metric}): {exc}")

                rows[metric].append((
                    serie.replace("_", " "), model_1, model_2,
                    stat_label, p_text, sig,
                ))

    label = "DM-HLN" if HARVEY_ADJ else "DM"
    columns = ["Serie", "Model 1", "Model 2", label, "p value", "Significance"]
    for metric in ("MSE", "MAPE"):
        print(f"\nDIEBOLD-MARIANO {'***' if HARVEY_ADJ else '(original)'} — {metric}")
        print(pd.DataFrame(rows[metric], columns=columns).to_string(index=False))
        print("*** p < 0.001; ** p < 0.01; * p < 0.05; ns = not significant")
        print("Estadístico negativo: favorece Model 1; positivo: favorece Model 2")

    print(f"\nConfiguración: HAC_LAGS={HAC_LAGS}; HORIZON={HORIZON}; HARVEY_ADJ={HARVEY_ADJ}")
    if HARVEY_ADJ:
        print("Nota: h=1 se usa como referencia HLN; las predicciones reales abarcan h=1..14.")


if __name__ == "__main__":
    main()
