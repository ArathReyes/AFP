#!/usr/bin/env python3
"""
CNN+Transformer forecaster for US Rates spreads & flys
- Mirrors the CLI, data flow, scaling, level/delta target logic, training loop,
  early stopping, plots, and Excel export from lstm_us_rates.py.

Usage (very similar to LSTM script):
    python cnn_transformer_us_rates.py \
        --path Benchmarks/data/US_TimeSeries.xlsx \
        --sheet SpreadsAndFlys \
        --lookback 60 --horizon 1 \
        --target delta --scale robust \
        --epochs 30 --batch-size 128 --lr 1e-3 --patience 6

Outputs:
- PNG:  ./CNN_Transformer/Results/test_us_rates_forecasts.png
- XLSX: ./CNN_Transformer/Results/test_us_rates_predictions.xlsx (Actual/Predicted)

Default (no --sheet): runs all country sheets, writes the performance table to
  CNN_Transformer/Results/performance_table_countries.csv and .xlsx, and saves one
  test_rates_forecasts_<sheet>.png per sheet in the same Results folder.
With --sheet <name>: runs that single sheet only (plot only).
All results are saved under CNN_Transformer/Results (next to this script), regardless of current working directory.

Forecast metrics produced (per country):
  - RMSE, MAE: scale-dependent errors.
  - MAPE_pct, sMAPE_pct: percentage errors (%; sMAPE symmetric).
  - MAAPE: mean arctangent absolute percentage error (bounded, robust).
  - R2: coefficient of determination (1 - SS_res/SS_tot).
  - Correlation: Pearson correlation between predicted and actual.
  - Baseline_*: same metrics for a rolling-mean baseline (lookback window).
  - N_test, N_features: number of test points and series (spreads/flies).

Note: For delta target, we reconstruct level predictions with S_{t+1} = S_t + ΔS_{t+1}.
"""
import argparse, os, math
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def n_sequences(T: int, L: int, H: int) -> int:
    # from your loop: t in [L, T-H]
    return max(0, T - H - L + 1)

def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def load_spreads(path: str, sheet: str = None) -> pd.DataFrame:
    """Load precomputed spread/fly time series.

    Supports:
      - CSV/TXT with first column as date index
      - Excel with optional --sheet (sheet name or index)

    Cleans:
      - coerces index to datetime when possible
      - keeps numeric columns only
      - drops empty rows, forward-fills, drops remaining NaNs
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.lower().endswith((".csv", ".txt")):
        df = pd.read_csv(path, index_col=0)

    elif path.lower().endswith((".xlsx", ".xls")):
        # IMPORTANT: respect the --sheet argument (name or integer index)
        sheet_name = sheet if sheet is not None else 0
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
        except ValueError:
            # if user passed something like "0" as a string, try int
            try:
                df = pd.read_excel(path, sheet_name=int(sheet_name), index_col=0)
            except Exception as e:
                raise

    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    # Try to coerce index to datetime (works for Excel serials, strings, timestamps)
    if not isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, errors="coerce")
        # If most values parsed, use it; otherwise keep original
        if idx.notna().mean() >= 0.8:
            df.index = idx
    df = df.sort_index()

    # Keep numeric columns (your generated file already is numeric)
    num_df = df.select_dtypes(include=[np.number]).copy()

    # Clean rows
    num_df = num_df.replace([np.inf, -np.inf], np.nan)
    num_df = num_df.dropna(how="all").ffill().dropna()

    if num_df.empty:
        raise ValueError("No numeric spread/fly columns after cleaning.")

    return num_df



def time_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    n = len(df); n_train = int(n*train_frac); n_val = int(n*val_frac)
    return df.iloc[:n_train], df.iloc[n_train:n_train+n_val], df.iloc[n_train+n_val:]


def build_sequences(X: np.ndarray, Y: np.ndarray, L: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    T = X.shape[0]
    xs, ys = [], []
    for t in range(L, T - H + 1):
        xs.append(X[t-L:t, :])      # (L, F)
        ys.append(Y[t:t+H, :])      # (H, F)
    return np.asarray(xs), np.asarray(ys)

# ---------------- Scalers ----------------
class StandardScalerNP:
    def fit(self, x):
        self.mean_ = np.nanmean(x, axis=0); self.std_ = np.nanstd(x, axis=0); self.std_[self.std_==0]=1.0; return self
    def transform(self, x): return (x - self.mean_) / self.std_
    def inverse_transform(self, x): return x*self.std_ + self.mean_

class RobustScalerNP:
    def fit(self, x):
        self.med_ = np.nanmedian(x, axis=0); q1 = np.nanpercentile(x,25,axis=0); q3 = np.nanpercentile(x,75,axis=0)
        self.iqr_ = q3 - q1; self.iqr_[self.iqr_==0]=1.0; return self
    def transform(self, x): return (x - self.med_) / self.iqr_
    def inverse_transform(self, x): return x*self.iqr_ + self.med_

class MinMaxScalerNP:
    def fit(self, x):
        self.min_ = np.nanmin(x, axis=0); self.max_ = np.nanmax(x, axis=0); rng = self.max_ - self.min_; rng[rng==0]=1.0; self.rng_=rng; return self
    def transform(self, x): return (x - self.min_) / self.rng_
    def inverse_transform(self, x): return x*self.rng_ + self.min_

def get_scaler(name: str):
    if name == "standard": return StandardScalerNP()
    if name == "robust":   return RobustScalerNP()
    if name == "minmax":   return MinMaxScalerNP()
    raise ValueError("scale must be one of: standard, robust, minmax")


# ---------------- Forecast performance metrics ----------------
def _mse(a, b): return float(np.mean((a - b) ** 2))
def _mae(a, b): return float(np.mean(np.abs(a - b)))
def _mape(pred, actual):
    denom = np.abs(actual)
    denom[denom < 1e-12] = np.nan
    return float(np.nanmean(np.abs((actual - pred) / denom)) * 100)
def _smape(pred, actual):
    mean = (np.abs(actual) + np.abs(pred)) / 2
    mean[mean < 1e-12] = np.nan
    return float(np.nanmean(np.abs(actual - pred) / mean) * 100)
def _maape(pred, actual):
    epsilon = np.finfo(float).eps
    ape = np.abs((actual - pred) / np.maximum(np.abs(actual), epsilon))
    return float(np.mean(np.arctan(ape)))
def _r2(pred, actual):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1 - ss_res / ss_tot)
def _corr(pred, actual):
    if pred.size < 2 or np.std(pred) < 1e-12 or np.std(actual) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred.ravel(), actual.ravel())[0, 1])


def compute_forecast_metrics(level_pred: np.ndarray, level_true: np.ndarray,
                            baseline: np.ndarray) -> dict:
    """Compute model vs baseline forecast metrics (level space). All arrays (N, F)."""
    return {
        "RMSE": np.sqrt(_mse(level_pred, level_true)),
        "MAE": _mae(level_pred, level_true),
        "MAPE_pct": _mape(level_pred, level_true),
        "sMAPE_pct": _smape(level_pred, level_true),
        "MAAPE": _maape(level_pred, level_true),
        "R2": _r2(level_pred, level_true),
        "Correlation": _corr(level_pred, level_true),
        "Baseline_RMSE": np.sqrt(_mse(baseline, level_true)),
        "Baseline_MAE": _mae(baseline, level_true),
        "Baseline_MAPE_pct": _mape(baseline, level_true),
        "Baseline_sMAPE_pct": _smape(baseline, level_true),
        "Baseline_MAAPE": _maape(baseline, level_true),
        "Baseline_R2": _r2(baseline, level_true),
    }


# ---------------- Model ----------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Transformer-compatible)."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # (T, 1, D)
    def forward(self, x):
        # x: (T, N, D)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CNNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(c_in,  c_out, kernel_size, padding=padding)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, padding=padding)
        self.bn2   = nn.BatchNorm1d(c_out)
        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout)
        self.short = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x):
        # x: (N, C, T)
        res = self.short(x)
        y = self.conv1(x); y = self.bn1(y); y = self.act(y)
        y = self.conv2(y); y = self.bn2(y)
        y = y + res; y = self.act(y); y = self.drop(y)
        return y

class CNNTransformer(nn.Module):
    def __init__(self, n_features: int, cnn_channels=(32,64), kernel_size=3,
                 attn_heads=4, ff_mult=2, dropout=0.2, horizon: int = 1):
        super().__init__()
        c_list = [n_features] + list(cnn_channels)
        self.cnn = nn.ModuleList([CNNBlock(c_list[i], c_list[i+1], kernel_size, dropout)
                                   for i in range(len(c_list)-1)])
        d_model = c_list[-1]
        self.posenc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=attn_heads,
                                                   dim_feedforward=ff_mult*d_model,
                                                   dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.readout = nn.Linear(d_model, n_features)  # predict all features at once
        self.horizon = horizon
    def forward(self, x):
        # x: (N, L, F)
        x = x.permute(0,2,1)  # (N, F, L)
        for blk in self.cnn:
            x = blk(x)
        x = x.permute(2,0,1)  # (L, N, C)
        x = self.posenc(x)
        h = self.encoder(x)   # (L, N, C)
        # autoregressive one-step: use last token
        y1 = self.readout(h[-1])  # (N, F)
        # shape to (N, H=1, F) to match LSTM script conventions
        return y1.unsqueeze(1)

# ---------------- Training helpers ----------------
def huber_loss(delta=1.0):
    return nn.HuberLoss(delta=delta)


def make_loaders(X_tr, y_tr, X_va, y_va, batch_size):
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    tr_ds = TensorDataset(to_tensor(X_tr), to_tensor(y_tr))
    va_ds = TensorDataset(to_tensor(X_va), to_tensor(y_va)) if X_va is not None else None
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False) if va_ds is not None else None
    return tr_loader, va_loader


def evaluate(model, loader, device):
    mse_sum = mae_sum = 0.0; n = 0
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            preds.append(yhat.cpu().numpy()); trues.append(yb.cpu().numpy())
            d = (yhat - yb).detach().cpu().numpy().ravel()
            mse_sum += np.sum(d**2); mae_sum += np.sum(np.abs(d)); n += d.size
    rmse = float(np.sqrt(mse_sum / max(n,1))); mae = float(mae_sum / max(n,1))
    return rmse, mae, np.concatenate(preds, 0), np.concatenate(trues, 0)


def run_one_sheet(path: str, sheet: str, args) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list, int]:
    """Run full train/val/test pipeline for one sheet. Returns (metrics, level_pred, level_true, baseline, teX_df, features, L)."""
    df = load_spreads(path, sheet)
    features = df.columns.tolist()

    if args.target == "delta":
        df_y = df.diff().shift(-1).dropna()
        df_x = df.loc[df_y.index]
    else:
        df_y = df.shift(-1).dropna()
        df_x = df.loc[df_y.index]

    trX_df, vaX_df, teX_df = time_split(df_x)
    trY_df, vaY_df, teY_df = df_y.loc[trX_df.index], df_y.loc[vaX_df.index], df_y.loc[teX_df.index]

    ScalerX = get_scaler(args.scale)
    ScalerY = get_scaler(args.scale)
    X_tr = ScalerX.fit(trX_df.values.copy()).transform(trX_df.values)
    X_va = ScalerX.transform(vaX_df.values)
    X_te = ScalerX.transform(teX_df.values)
    Y_tr = ScalerY.fit(trY_df.values.copy()).transform(trY_df.values)
    Y_va = ScalerY.transform(vaY_df.values)
    Y_te = ScalerY.transform(teY_df.values)

    L, H = args.lookback, args.horizon
    trX, trY = build_sequences(X_tr, Y_tr, L, H)
    vaX, vaY = build_sequences(X_va, Y_va, L, H)
    teX, teY = build_sequences(X_te, Y_te, L, H)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNTransformer(
        n_features=len(features), cnn_channels=tuple(args.cnn_ch), kernel_size=args.kernel,
        attn_heads=args.attn_heads, ff_mult=args.ff_mult, dropout=args.dropout, horizon=H
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = huber_loss(args.huber_delta)
    train_loader, val_loader = make_loaders(trX, trY, vaX, vaY, args.batch_size)

    best_val = float("inf")
    best_state = None
    patience = args.patience
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        va_loss /= len(val_loader.dataset)
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    test_ds = TensorDataset(torch.tensor(teX, dtype=torch.float32), torch.tensor(teY, dtype=torch.float32))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    _, _, y_pred_s, y_true_s = evaluate(model, test_loader, device)

    y_pred_raw = np.vstack([ScalerY.inverse_transform(y_pred_s[i, 0, :].reshape(1, -1))
                            for i in range(y_pred_s.shape[0])])
    y_true_raw = np.vstack([ScalerY.inverse_transform(y_true_s[i, 0, :].reshape(1, -1))
                            for i in range(y_true_s.shape[0])])
    if args.target == "delta":
        last_levels = teX_df.values[-y_pred_raw.shape[0]-1:-1, :]
        level_pred = last_levels + y_pred_raw
        level_true = last_levels + y_true_raw
    else:
        level_pred = y_pred_raw
        level_true = y_true_raw
    baseline = teX_df.rolling(window=L).mean().values[L-1:-1, :]

    metrics = compute_forecast_metrics(level_pred, level_true, baseline)
    metrics["N_test"] = level_pred.shape[0]
    metrics["N_features"] = level_pred.shape[1]
    return metrics, level_pred, level_true, baseline, teX_df, features, L


def _sheet_tag(sheet_name: str) -> str:
    """Filesystem-safe tag from sheet name for plot/Excel filenames."""
    s = str(sheet_name).strip().replace(" ", "_")
    return "".join(c for c in s if c.isalnum() or c in "_-") or "forecasts"


def _save_forecast_plot(out_dir: str, level_pred: np.ndarray, level_true: np.ndarray,
                       baseline: np.ndarray, teX_df: pd.DataFrame, features: list, L: int, tag: str) -> None:
    """Save test_rates_forecasts_{tag}.png to out_dir."""
    N = level_pred.shape[0]
    idx = teX_df.index[L : L + N]
    n_feats = len(features)
    n_cols = 2
    n_rows = int(math.ceil(n_feats / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.6 * n_rows), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for j, feat in enumerate(features):
        ax = axes[j]
        ax.plot(idx, level_true[:, j], label="Actual")
        ax.plot(idx, level_pred[:, j], label="CNN+Transformer")
        ax.plot(idx, baseline[:, j], label="Baseline", alpha=0.65, linestyle="--")
        maape_j = _maape(level_pred[:, j], level_true[:, j])
        base_maape_j = _maape(baseline[:, j], level_true[:, j])
        ax.set_title(f"{feat}  MAAPE={maape_j:.3f} | Baseline MAAPE={base_maape_j:.3f}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend()
    for k in range(j + 1, len(axes)):
        fig.delaxes(axes[k])
    fig.suptitle(f"Test: Actual vs CNN+Transformer (Lookback: {L})", y=0.995, fontsize=12)
    png_path = os.path.join(out_dir, f"test_rates_forecasts_{tag}.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", png_path)


# ---------------- Main ----------------
def main():
    path_ = os.getcwd()
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default= path_ + "/data/Rates_SpreadsFlys_MR.xlsx")
    ap.add_argument("--sheet", default=None,
                    help="Single sheet (country) name; if omitted, run all sheets and write country-level table to Results")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon",  type=int, default=1)
    ap.add_argument("--epochs",   type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--scale",    choices=["standard","robust","minmax"], default="robust")
    ap.add_argument("--target",   choices=["level","delta"], default="delta")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    # CNN+Transformer specific
    ap.add_argument("--cnn-ch",   type=int, nargs="*", default=[32,64])
    ap.add_argument("--kernel",   type=int, default=3)
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--ff-mult",  type=int, default=2)
    ap.add_argument("--dropout",  type=float, default=0.2)
    args = ap.parse_args()
    set_seed(args.seed)

    # Results always in CNN_Transformer/Results (next to this script), regardless of cwd
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(_script_dir, "Results")
    ensure_dirs(out_dir)
    data_path = args.path if os.path.isabs(args.path) else os.path.normpath(os.path.join(os.getcwd(), args.path))
    print("Results will be saved to:", os.path.abspath(out_dir))

    # No --sheet: run all country sheets, save table AND one plot+Excel per sheet
    if args.sheet is None:
        if not os.path.exists(data_path):
            print(f"ERROR: File not found: {data_path}")
            return
        xls = pd.ExcelFile(data_path)
        sheets = [s for s in xls.sheet_names if s.lower() != "adf_pvalues"]
        if not sheets:
            print("No data sheets found (excluding 'ADF_pvalues').")
            return
        print(f"Running CNN+Transformer for {len(sheets)} sheets: {sheets}")
        rows = []
        for sheet in sheets:
            print(f"\n--- Sheet: {sheet} ---")
            try:
                metrics, level_pred, level_true, baseline, teX_df, features, L = run_one_sheet(data_path, sheet, args)
                metrics["Country"] = sheet
                rows.append(metrics)
                tag = _sheet_tag(sheet)
                _save_forecast_plot(out_dir, level_pred, level_true, baseline, teX_df, features, L, tag)
            except Exception as e:
                print(f"  Skip '{sheet}': {e}")
                import traceback
                traceback.print_exc()
        if not rows:
            print("No sheets completed successfully.")
            return
        df = pd.DataFrame(rows)
        # Improvement over baseline (positive = model better)
        if "Baseline_RMSE" in df.columns and (df["Baseline_RMSE"] > 0).all():
            df["RMSE_improvement_pct"] = (1 - df["RMSE"] / df["Baseline_RMSE"]) * 100
        if "Baseline_MAE" in df.columns and (df["Baseline_MAE"] > 0).all():
            df["MAE_improvement_pct"] = (1 - df["MAE"] / df["Baseline_MAE"]) * 100
        col_order = ["Country", "RMSE", "MAE", "MAPE_pct", "sMAPE_pct", "MAAPE", "R2", "Correlation",
                     "Baseline_RMSE", "Baseline_MAE", "Baseline_MAPE_pct", "Baseline_sMAPE_pct", "Baseline_MAAPE", "Baseline_R2",
                     "RMSE_improvement_pct", "MAE_improvement_pct",
                     "N_test", "N_features"]
        df = df[[c for c in col_order if c in df.columns]]
        csv_path = os.path.join(out_dir, "performance_table_countries.csv")
        xlsx_path = os.path.join(out_dir, "performance_table_countries.xlsx")
        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False)
        print(f"\nPerformance table saved: {csv_path}")
        print(f"Performance table saved: {xlsx_path}")
        return

    # Single-sheet: run and save plot + Excel
    metrics, level_pred, level_true, baseline, teX_df, features, L = run_one_sheet(data_path, args.sheet, args)
    print(f"Test (raw)  CNN+Transformer: RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} MAPE={metrics['MAPE_pct']:.4f} sMAPE={metrics['sMAPE_pct']:.4f} MAAPE={metrics['MAAPE']:.4f} R2={metrics['R2']:.4f} Corr={metrics['Correlation']:.4f} | "
          f"Baseline: RMSE={metrics['Baseline_RMSE']:.4f} MAE={metrics['Baseline_MAE']:.4f} MAPE={metrics['Baseline_MAPE_pct']:.4f} sMAPE={metrics['Baseline_sMAPE_pct']:.4f} MAAPE={metrics['Baseline_MAAPE']:.4f} R2={metrics['Baseline_R2']:.4f}")

    tag = _sheet_tag(args.sheet)
    try:
        _save_forecast_plot(out_dir, level_pred, level_true, baseline, teX_df, features, L, tag)
    except Exception as e:
        import traceback
        print(f"Plotting skipped due to error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
