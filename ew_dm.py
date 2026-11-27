"""Expanding-window forecasts and one-step DM-HLN; no hyperparameter search."""
from pathlib import Path
from itertools import combinations
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import t as student_t

BASE = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if DEVICE.type == 'cpu':
    torch.set_num_threads(1)

SERIES = ['15-34 M', '15-34 F', '35-49 M', '35-49 F', '50-74 M', '50-74 F', 'Global_Rate']
SARIMA = [
    ((0,1,0),(0,1,1,4)), ((0,1,0),(0,1,1,4)),
    ((0,1,0),(1,1,2,4)), ((0,1,0),(0,1,1,4)),
    ((2,1,1),(0,1,1,4)), ((2,1,0),(0,1,1,4)),
    ((0,1,0),(0,1,1,4))
]
# Cada tupla: (neurons, lr, window, epochs, batch_size, dropout)
CONFIG = {'LSTM': [((128, 128), 0.01, 4, 3, 32, 0.1),
          ((128, 128), 0.01, 4, 3, 32, 0.0),
          ((64, 32), 0.01, 4, 5, 32, 0.1),
          ((128, 64), 0.01, 4, 13, 32, 0.0),
          ((64, 64), 0.01, 4, 4, 32, 0.1),
          ((128,), 0.01, 4, 99, 16, 0.0),
          ((128, 128), 0.01, 4, 3, 32, 0.0)],
 'GRU': [((128, 64), 0.01, 4, 2, 16, 0.3),
         ((128, 128), 0.01, 4, 2, 32, 0.1),
         ((128,), 0.01, 4, 3, 32, 0.0),
         ((128, 128), 0.01, 4, 7, 32, 0.1),
         ((128, 128), 0.01, 4, 6, 16, 0.3),
         ((128, 32), 0.001, 6, 13, 32, 0.0),
         ((128, 64), 0.01, 4, 2, 16, 0.1)],
 'SARIMA-LSTM_seq': [((32,), 0.001, 8, 9, 16, 0.1),
                     ((32, 32), 0.01, 4, 76, 16, 0.0),
                     ((64,), 0.01, 4, 96, 16, 0.0),
                     ((32,), 0.001, 8, 8, 16, 0.1),
                     ((128, 128), 0.01, 8, 53, 32, 0.3),
                     ((128, 32), 0.001, 4, 10, 32, 0.3),
                     ((128, 32), 0.01, 8, 96, 32, 0.1)],
 'SARIMA-GRU_seq': [((32,), 0.001, 8, 7, 16, 0.0),
                    ((128, 32), 0.01, 8, 89, 16, 0.0),
                    ((128, 128), 0.01, 6, 84, 16, 0.0),
                    ((128,), 0.001, 8, 2, 16, 0.3),
                    ((128, 32), 0.01, 8, 99, 32, 0.1),
                    ((64, 32), 0.001, 4, 15, 16, 0.1),
                    ((128, 64), 0.01, 8, 59, 32, 0.0)]}

class Recurrent(nn.Module):
    def __init__(self, kind, neurons, dropout, vanilla=False, horizon=1):
        super().__init__()
        unit = nn.LSTM if kind == 'LSTM' else nn.GRU
        self.layers = nn.ModuleList()
        size = 1
        for n in neurons:
            self.layers.append(unit(size, n, batch_first=True))
            size = n
        self.drop = nn.Dropout(dropout)
        self.vanilla = vanilla
        self.fc = nn.Linear(size, horizon)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            if (i < len(self.layers)-1) if self.vanilla else (i == 0):
                x = self.drop(x)
        x = self.drop(x[:, -1, :]) if self.vanilla and len(self.layers) == 1 else x[:, -1, :]
        return self.fc(x)

def predict(kind, params, historical, steps, seed, vanilla, direct_horizon=14):
    neurons, lr, window, epochs, batch, dropout = params
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Vanilla: mismos tipos float32 usados en los notebooks originales.
    historical = np.asarray(historical, dtype=np.float32 if vanilla else float)
    scaler = MinMaxScaler().fit(historical.reshape(-1, 1))
    scaled = scaler.transform(historical.reshape(-1, 1)).ravel()
    H = direct_horizon if vanilla else 1
    X = np.array([scaled[i-window:i] for i in range(window, len(scaled)-H+1)], dtype=np.float32)
    y = np.array([scaled[i:i+H] for i in range(window, len(scaled)-H+1)], dtype=np.float32)
    X = torch.as_tensor(X[:, :, None], dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    # DataLoader sin shuffle en vanilla; randperm en los híbridos.
    if vanilla:
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=batch, shuffle=False)
    model = Recurrent(kind, neurons, dropout, vanilla=vanilla, horizon=H).to(DEVICE)
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

    model.eval()
    if vanilla:
        # Original notebook: direct multi-output prediction, no feedback.
        x = torch.tensor(scaled[-window:].reshape(1,window,1), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred = model(x).cpu().numpy()
        return scaler.inverse_transform(pred.reshape(-1,1)).ravel()[:steps]
    # Original sequential notebooks: recurse in unscaled residual space.
    history = list(historical)
    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            x = scaler.transform(np.array(history[-window:]).reshape(-1,1))
            x = torch.tensor(x.reshape(1,window,1), dtype=torch.float32, device=DEVICE)
            pred = scaler.inverse_transform(model(x).cpu().numpy())[0,0]
            predictions.append(pred)
            history.append(float(pred))
    return np.asarray(predictions)


def dm_hln(e1, e2, hac_lags=0):
    """Squared loss, h=1, Bartlett HAC, two-sided Student t reference."""
    d = np.asarray(e1, dtype=float)**2 - np.asarray(e2, dtype=float)**2
    T = len(d)
    if T < 2 or not np.isfinite(d).all() or not 0 <= hac_lags < T:
        raise ValueError('Invalid DM inputs')
    u = d - d.mean()
    omega = np.dot(u, u) / T
    for k in range(1, hac_lags + 1):
        omega += 2 * (1-k/(hac_lags+1)) * np.dot(u[k:], u[:-k]) / T
    if omega <= 0:
        return np.nan, np.nan, d.mean()
    stat = d.mean() / np.sqrt(omega/T) * np.sqrt((T-1)/T)
    return stat, 2 * student_t.sf(abs(stat), df=T-1), d.mean()


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    ix = valid[np.argsort(p[valid])]
    out[ix] = np.minimum(1, np.maximum.accumulate(p[ix] * np.arange(len(ix), 0, -1)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=BASE/'data.csv')
    ap.add_argument('--out', type=Path, default=BASE/'results'/'expanding_dm_original')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--initial', type=int, default=76)
    ap.add_argument('--hac-lags', type=int, default=0, help='0 is conventional h-1 for h=1; 1 for sensitivity')
    ap.add_argument('--direct-horizon', choices=['fixed','remaining'], default='fixed', help='fixed preserves the original 14-output model; remaining reduces output width at each origin')
    args = ap.parse_args()
    df = pd.read_csv(args.data, index_col='Period')
    df.index = pd.PeriodIndex(df.index.str.strip(), freq='Q')
    df = df.sort_index()
    if df.index.has_duplicates or df[SERIES].isna().any().any():
        raise ValueError('Duplicate periods or missing observations')
    if not 0 < args.initial < len(df)-1:
        raise ValueError('Invalid initial training length')
    if not 0 <= args.hac_lags < len(df)-args.initial:
        raise ValueError('Invalid HAC lag count')
    args.out.mkdir(parents=True, exist_ok=True)
    models = ['SARIMA', *CONFIG]
    rows = []
    for j, serie in enumerate(SERIES):
        values = df[serie].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite data: {serie}')
        order, seasonal = SARIMA[j]
        for n in range(args.initial, len(values)):
            history = values[:n]
            steps = len(values)-n
            fit = SARIMAX(history, order=order, seasonal_order=seasonal,
                          enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
            if not fit.mle_retvals.get('converged', True):
                print(f'WARNING: SARIMA nonconvergence: {serie}, origin {n}', flush=True)
            residuals = np.asarray(fit.resid)[fit.loglikelihood_burn:]
            baseline = np.asarray(fit.forecast(steps=steps))
            forecasts = {'SARIMA': baseline}
            for name, configs in CONFIG.items():
                hybrid = name.startswith('SARIMA-')
                kind = 'LSTM' if 'LSTM' in name else 'GRU'
                pred = predict(kind, configs[j], residuals if hybrid else history,
                               steps, args.seed, vanilla=not hybrid,
                               direct_horizon=(len(values)-args.initial if args.direct_horizon=='fixed' else steps))
                forecasts[name] = baseline + pred if hybrid else pred
            for name, pred in forecasts.items():
                for h, yhat in enumerate(pred, 1):
                    actual = values[n+h-1]
                    error = actual-float(yhat)
                    rows.append(dict(Serie=serie, Model=name, Seed=args.seed,
                        Origin=n, Origin_period=df.index[n-1], Horizon=h,
                        Target=n+h, Target_period=df.index[n+h-1], Actual=actual,
                        Forecast=yhat, Error=error, Squared_error=error**2,
                        SARIMA_converged=fit.mle_retvals.get('converged', True)))
            print(f'{serie}: origin {n}, {steps} forecasts per model', flush=True)
        pd.DataFrame(rows).to_csv(args.out/'all_predictions.csv', index=False)
    result = pd.DataFrame(rows)
    one = result[result.Horizon == 1].copy()
    one.to_csv(args.out/'h1_predictions.csv', index=False)
    for serie in SERIES:
        for name in models:
            subset = result[(result.Serie==serie)&(result.Model==name)]
            matrix = subset.pivot(index='Origin', columns='Target', values='Error')
            safe = serie.replace(' ', '_')
            matrix.to_csv(args.out/f'errors_{safe}_{name}.csv')
    summary = one.groupby(['Serie','Model'], sort=False).agg(
        T=('Error','size'), MAE=('Error',lambda e: np.mean(np.abs(e))),
        MSE=('Squared_error','mean')).reset_index()
    summary['RMSE'] = np.sqrt(summary.MSE)
    summary.to_csv(args.out/'h1_metrics.csv', index=False)
    by_h = result.groupby(['Serie','Model','Horizon'], sort=False).agg(
        T=('Error','size'), MAE=('Error',lambda e: np.mean(np.abs(e))),
        MSE=('Squared_error','mean')).reset_index()
    by_h['RMSE'] = np.sqrt(by_h.MSE)
    by_h.to_csv(args.out/'metrics_by_horizon.csv', index=False)
    tests = []
    for serie in SERIES:
        wide = one[one.Serie==serie].pivot(index='Origin',columns='Model',values='Error').sort_index()
        for a,b in combinations(models,2):
            stat,p,delta = dm_hln(wide[a],wide[b],args.hac_lags)
            tests.append(dict(Serie=serie, Model1=a, Model2=b, h=1,T=len(wide),
                HAC_lags=args.hac_lags, DM_HLN=stat, p_value=p, Mean_loss_difference=delta))
    tests = pd.DataFrame(tests)
    tests['p_Holm_within_series'] = tests.groupby('Serie',sort=False).p_value.transform(holm)
    tests.to_csv(args.out/'dm_all_pairs.csv',index=False)
    benchmark = tests[tests.Model1=='SARIMA'].copy()
    benchmark['p_Holm_SARIMA_within_series'] = benchmark.groupby('Serie',sort=False).p_value.transform(holm)
    benchmark.to_csv(args.out/'dm_vs_sarima.csv',index=False)
    original_best = dict(zip(SERIES, ['SARIMA','SARIMA','SARIMA-LSTM_seq','SARIMA-LSTM_seq','SARIMA','LSTM','SARIMA']))
    selected = []
    for serie, best in original_best.items():
        for _, row in tests[(tests.Serie==serie)&((tests.Model1==best)|(tests.Model2==best))].iterrows():
            row = row.copy()
            if row.Model2 == best:
                row.Model1, row.Model2 = row.Model2, row.Model1
                row.DM_HLN = -row.DM_HLN
                row.Mean_loss_difference = -row.Mean_loss_difference
            selected.append(row)
    pd.DataFrame(selected).to_csv(args.out/'dm_original_best.csv',index=False)
    print(f'Results: {args.out}')

if __name__ == '__main__':
    main()
