"""Robustez con 10 semillas; reproduce las implementaciones vanilla y secuencial."""
from pathlib import Path
from time import perf_counter
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parent  # Guardar randomization.py junto a data.csv
OUT = BASE / 'results' / 'robustness'
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if DEVICE.type == 'cpu':
    torch.set_num_threads(1)
SEEDS = [42, 137, 256, 389, 512, 673, 784, 891, 1024, 2026]
SERIES = ['15-34 M', '15-34 F', '35-49 M', '35-49 F', '50-74 M', '50-74 F', 'Global_Rate']
SARIMA = [
    ((0,1,0),(0,1,1,4)), ((0,1,0),(0,1,1,4)),
    ((0,1,0),(1,1,2,4)), ((0,1,0),(0,1,1,4)),
    ((2,1,1),(0,1,1,4)), ((2,1,0),(0,1,1,4)),
    ((0,1,0),(0,1,1,4))
]
# Cada tupla: (neurons, lr, window, epochs, batch_size, dropout)
CONFIG = {
    'LSTM': [
        ((128,128),.01,4,3,32,.1),
        ((128,128),.01,4,3,32,0.),
        ((64,32),.01,4,5,32,.1),
        ((128,64),.01,4,13,32,0.),
        ((64,64),.01,4,4,32,.1),
        ((128,),.01,4,99,16,0.),
        ((128,128),.01,4,3,32,0.)
    ],
    'GRU': [
        ((128,64),.01,4,2,16,.3),
        ((128,128),.01,4,2,32,.1),
        ((128,),.01,4,3,32,0.),
        ((128,128),.01,4,7,32,.1),
        ((128,128),.01,4,6,16,.3),
        ((128,32),.001,6,13,32,0.),
        ((128,64),.01,4,2,16,.1)
    ],
    'SARIMA-LSTM_seq': [
        ((32,),.01,4,4,16,0.),
        ((128,64),.001,4,4,16,0.),
        ((128,64),.01,4,64,16,.1),
        ((128,32),.001,4,4,16,.3),
        ((64,),.01,8,46,16,0.),
        ((64,64),.01,8,54,32,.1),
        ((128,64),.01,4,5,16,.3)
    ],
    'SARIMA-GRU_seq': [
        ((128,),.01,4,3,32,0.),
        ((128,64),.01,8,97,16,.3),
        ((128,64),.01,4,88,16,0.),
        ((128,),.001,4,5,32,.3),
        ((32,),.01,8,21,16,0.),
        ((128,128),.01,8,59,32,.3),
        ((128,128),.01,8,52,16,.1)
    ]
}

class Recurrent(nn.Module):
    def __init__(self, kind, neurons, dropout, vanilla=False):
        super().__init__()
        unit = nn.LSTM if kind == 'LSTM' else nn.GRU
        self.layers = nn.ModuleList()
        size = 1
        for n in neurons:
            self.layers.append(unit(size, n, batch_first=True))
            size = n
        self.drop = nn.Dropout(dropout)
        self.vanilla = vanilla
        self.fc = nn.Linear(size, 1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            if (i < len(self.layers)-1) if self.vanilla else (i == 0):
                x = self.drop(x)
        x = self.drop(x[:, -1, :]) if self.vanilla and len(self.layers) == 1 else x[:, -1, :]
        return self.fc(x)

def predict(kind, params, historical, actual_test, seed, sarima_test=None):
    neurons, lr, window, epochs, batch, dropout = params
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    vanilla = sarima_test is None
    # Vanilla: mismos tipos float32 usados en los notebooks originales.
    historical = np.asarray(historical, dtype=np.float32 if vanilla else float)
    actual_test = np.asarray(actual_test, dtype=np.float32 if vanilla else float)
    scaler = MinMaxScaler().fit(historical.reshape(-1, 1))
    scaled = scaler.transform(historical.reshape(-1, 1)).ravel()
    X = np.stack([scaled[i-window:i] for i in range(window, len(scaled))])
    X = torch.as_tensor(X[:, :, None], dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(scaled[window:, None], dtype=torch.float32, device=DEVICE)
    # DataLoader sin shuffle en vanilla; randperm en los híbridos.
    if vanilla:
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=batch, shuffle=False)
    model = Recurrent(kind, neurons, dropout, vanilla=vanilla).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):  # Épocas fijas elegidas con validation
        model.train()
        iterator = loader if vanilla else ((X[ids], y[ids]) for ids in torch.randperm(len(X), device=DEVICE).split(batch))
        for xb, yb in iterator:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()

    # Test vanilla: predicción vectorizada con ventanas del notebook original.
    if vanilla:
        test_scaled = scaler.transform(actual_test.reshape(-1, 1)).ravel()
        data_test = np.concatenate([scaled, test_scaled])
        Xt = np.array([data_test[i-window:i] for i in range(len(scaled), len(data_test))], dtype=np.float32)
        model.eval()
        with torch.no_grad():
            out = model(torch.as_tensor(Xt[:, :, None], device=DEVICE)).cpu().numpy()
        return scaler.inverse_transform(out).ravel()

    history = list(historical)
    pred = []
    model.eval()
    with torch.no_grad():
        for i, observed in enumerate(actual_test):
            x = scaler.transform(np.array(history[-window:]).reshape(-1, 1))
            x = torch.as_tensor(x.reshape(1, window, 1), dtype=torch.float32, device=DEVICE)
            value = scaler.inverse_transform(model(x).cpu().numpy())[0, 0]
            pred.append(value)
            # Rolling one-step: SOLO se agrega el observado después de predecir.
            history.append(observed - sarima_test[i] if sarima_test is not None else observed)
    pred = np.array(pred)
    return pred + sarima_test if sarima_test is not None else pred

def metrics(real, pred, insample):
    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    nonzero = real != 0
    denom = np.mean(np.abs(insample[4:] - insample[:-4]))
    return dict(MAE=mae, MSE=mse, RMSE=np.sqrt(mse),
                MAPE=100*np.mean(np.abs((real[nonzero]-pred[nonzero])/real[nonzero])) if nonzero.any() else np.nan,
                MASE=mae/denom if denom > 0 else np.nan, R2=r2_score(real, pred))

def main():
    t0 = perf_counter()
    df = pd.read_csv(BASE / 'data.csv', index_col='Period', parse_dates=True).sort_index()
    split = int(len(df)*.85)  # 70 % train + 15 % validation; 15 % test
    rows = []
    for j, serie in enumerate(SERIES):
        y_trainval = df[serie].iloc[:split].to_numpy(dtype=float)
        y_test = df[serie].iloc[split:].to_numpy(dtype=float)
        order, seasonal = SARIMA[j]
        t = perf_counter()
        fit = SARIMAX(y_trainval, order=order, seasonal_order=seasonal,
                     enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
        residuals = np.asarray(fit.resid)[fit.loglikelihood_burn:]
        sarima_test = np.asarray(fit.forecast(steps=len(y_test)))
        print(f'{serie}: SARIMA ajustado en {perf_counter()-t:.1f} s', flush=True)
        for name, configurations in CONFIG.items():
            hybrid = name.startswith('SARIMA-')
            kind = 'LSTM' if 'LSTM' in name else 'GRU'
            t_group = perf_counter()
            for seed in SEEDS:
                t = perf_counter()
                preds = predict(kind, configurations[j], residuals if hybrid else y_trainval,
                                y_test, seed, sarima_test if hybrid else None)
                rows.append(dict(Serie=serie, Model=name, Seed=seed,
                                 **metrics(y_test if hybrid else y_test.astype(np.float32), preds, y_trainval),
                                 Time_s=round(perf_counter()-t, 3)))
            print(f'  {name}: 10 semillas en {perf_counter()-t_group:.1f} s', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'runs_10_seeds.csv', index=False)
    fields = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MASE', 'R2', 'Time_s']
    summary = result.groupby(['Serie', 'Model'], sort=False)[fields].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT/'summary_mean_std.csv', index=False)
    elapsed = perf_counter()-t0
    (OUT/'total_time.txt').write_text(f'{elapsed:.2f} segundos = {elapsed/60:.2f} minutos\n')
    print(f'\nTiempo total: {elapsed/60:.2f} minutos')
    print(f'Resultados guardados en: {OUT}')

    #Guardar en la imagen en pdf con formato latex
    
        # Academic style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight"
    })

    df = result.copy()
    df = df.rename(columns={"Serie": "Series"})

    model_names = {
        "SARIMA": "SARIMA",
        "LSTM": "LSTM",
        "GRU": "GRU",
        "SARIMA-LSTM_seq": r"SL$_{\mathrm{seq}}$",
        "SARIMA-GRU_seq": r"SG$_{\mathrm{seq}}$"
    }

    # Aplicar los nombres abreviados al DataFrame
    df["Model"] = df["Model"].replace(model_names)

    # Orden de los modelos en la gráfica
    models = [
        "LSTM",
        "GRU",
        r"SL$_{\mathrm{seq}}$",
        r"SG$_{\mathrm{seq}}$"
    ]

    series = [
        "15-34 M", "15-34 F",
        "35-49 M", "35-49 F",
        "50-74 M", "50-74 F",
        "Global_Rate"
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.barplot(
        data=df,
        y="RMSE",
        x="Model",
        hue="Series",
        order=models,
        hue_order=series,
        estimator=np.mean,
        errorbar="sd",
        capsize=0.12,
        palette="Set2",
        ax=ax
    )

    ax.set_ylabel("RMSE (mean ± sd)")
    ax.set_xlabel(" ")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    # Leyenda centrada arriba de la figura
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()

    fig.legend(
        handles, labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
        columnspacing=2.5,
        handletextpad=0.6,
        fontsize=18
    )

    sns.despine(ax=ax)

    # Reservar espacio arriba para la leyenda
    fig.subplots_adjust(
        top=0.90,
        bottom=0.13,
        left=0.09,
        right=0.97
    )

    plt.savefig(
        OUT / "rmse_robustness.pdf",
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.close()


if __name__ == '__main__':
    main()
