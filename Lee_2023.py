#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep Learning Nelson-Siegel (DLNS) OOS Yield Forecasting
--------------------------------------------------------
Simple local version:
- expects files at ./data/target_and_features.mat and ./data/dataset.csv by default
- no automatic path search
- expanding-window OOS
- benchmark: random walk
- horizon default: 12 months
"""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DLNSConfig:
    mat_path: str = "data/target_and_features.mat"
    csv_path: str = "data/dataset.csv"
    prefer_mat: bool = True

    horizon: int = 12
    oos_start: str = "1989-01"
    min_train_obs: int = 180
    reestimate_every: int = 1

    use_maturities: Optional[List[int]] = None
    lag_window: int = 12
    lambda_ns: float = 0.0609
    trainable_lambda: bool = False

    hidden_layers: Tuple[int, ...] = (64, 32)
    activation: str = "relu"
    dropout: float = 0.10
    weight_decay: float = 1e-5
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 250
    patience: int = 20
    val_size: int = 60
    seed: int = 42

    save_dir: str = "results_dlns"
    run_tag: str = "dlns_lee2023_h12"
    model_name: str = "DLNSModel"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def yyyymm_to_timestamp(values: Sequence[int]) -> pd.DatetimeIndex:
    values = np.asarray(values).astype(int)
    years = values // 100
    months = values % 100
    return pd.DatetimeIndex([
        pd.Timestamp(year=int(y), month=int(m), day=1) + pd.offsets.MonthEnd(0)
        for y, m in zip(years, months)
    ])


def parse_oos_start(oos_start: str) -> pd.Timestamp:
    y, m = [int(x) for x in oos_start.split("-")]
    return pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)


def safe_mat_struct_get(obj, field: str):
    if not hasattr(obj, field):
        raise AttributeError(f"MAT struct missing field '{field}'.")
    return getattr(obj, field)


def select_columns_by_maturity(names: Sequence[str], maturities: Optional[Sequence[int]]) -> List[int]:
    name_to_idx = {str(n): i for i, n in enumerate(names)}
    if maturities is None:
        return list(range(len(names)))

    idx = []
    for m in maturities:
        key = f"m{int(m):03d}"
        if key not in name_to_idx:
            raise KeyError(f"Requested maturity {m} months ({key}) not found in data.")
        idx.append(name_to_idx[key])
    return idx


def load_from_mat(mat_path: str, use_maturities: Optional[Sequence[int]] = None):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "X" not in mat:
        raise KeyError("MAT file must contain top-level key 'X'.")

    X = mat["X"]
    ys = safe_mat_struct_get(X, "yields")
    time_raw = np.asarray(safe_mat_struct_get(ys, "Time")).astype(int)
    y_raw = np.asarray(safe_mat_struct_get(ys, "data"), dtype=float)
    names = [str(x) for x in np.asarray(safe_mat_struct_get(ys, "names")).tolist()]

    col_idx = select_columns_by_maturity(names, use_maturities)
    names_sel = [names[i] for i in col_idx]
    maturities = np.array([int(n[1:]) for n in names_sel], dtype=int)
    dates = yyyymm_to_timestamp(time_raw)
    yields = y_raw[:, col_idx]
    return dates, yields, maturities, names_sel


def load_from_csv(csv_path: str, use_maturities: Optional[Sequence[int]] = None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Time" not in df.columns:
        raise KeyError("CSV file must contain 'Time' column in YYYYMM format.")

    yield_cols = [c for c in df.columns if c.startswith("m") and len(c) == 4 and c[1:].isdigit()]
    yield_cols = sorted(yield_cols, key=lambda x: int(x[1:]))
    if not yield_cols:
        raise ValueError("No yield columns like m012, m024, ... found in CSV.")

    col_idx = select_columns_by_maturity(yield_cols, use_maturities)
    names_sel = [yield_cols[i] for i in col_idx]
    maturities = np.array([int(n[1:]) for n in names_sel], dtype=int)
    dates = yyyymm_to_timestamp(df["Time"].to_numpy())
    yields = df[names_sel].to_numpy(dtype=float)
    return dates, yields, maturities, names_sel


def load_yield_panel(config: DLNSConfig):
    errors = []

    if config.prefer_mat:
        try:
            return (*load_from_mat(config.mat_path, config.use_maturities), "mat")
        except Exception as e:
            errors.append(f"MAT load failed: {e}")

        try:
            return (*load_from_csv(config.csv_path, config.use_maturities), "csv")
        except Exception as e:
            errors.append(f"CSV load failed: {e}")

    else:
        try:
            return (*load_from_csv(config.csv_path, config.use_maturities), "csv")
        except Exception as e:
            errors.append(f"CSV load failed: {e}")

        try:
            return (*load_from_mat(config.mat_path, config.use_maturities), "mat")
        except Exception as e:
            errors.append(f"MAT load failed: {e}")

    raise RuntimeError(" | ".join(errors))


def build_supervised_dataset(yields: np.ndarray, horizon: int, lag_window: int):
    T, N = yields.shape
    X_list, Y_list, origin_idx = [], [], []

    for t in range(lag_window - 1, T - horizon):
        hist = yields[t - lag_window + 1:t + 1]
        target = yields[t + horizon]

        if not np.isfinite(hist).all() or not np.isfinite(target).all():
            continue

        X_list.append(hist.reshape(-1))
        Y_list.append(target)
        origin_idx.append(t)

    if not X_list:
        raise ValueError("No valid supervised samples could be built.")

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(Y_list, dtype=np.float32),
        np.asarray(origin_idx, dtype=int),
    )


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class DLNSNet(nn.Module):
    def __init__(self, input_dim: int, maturities_months: np.ndarray, cfg: DLNSConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("tau", torch.tensor(maturities_months.astype(np.float32)).view(1, -1))

        layers_ = []
        prev = input_dim
        for units in cfg.hidden_layers:
            layers_.append(nn.Linear(prev, units))
            layers_.append(get_activation(cfg.activation))
            if cfg.dropout > 0:
                layers_.append(nn.Dropout(cfg.dropout))
            prev = units

        self.backbone = nn.Sequential(*layers_)
        self.factor_head = nn.Linear(prev, 3)

        if cfg.trainable_lambda:
            raw_init = np.log(np.exp(cfg.lambda_ns) - 1.0)
            self.lambda_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        else:
            self.register_buffer("lambda_const", torch.tensor(cfg.lambda_ns, dtype=torch.float32))

    def current_lambda(self) -> torch.Tensor:
        if self.cfg.trainable_lambda:
            return torch.nn.functional.softplus(self.lambda_raw) + 1e-6
        return self.lambda_const

    def ns_loadings(self) -> torch.Tensor:
        lam = self.current_lambda()
        x = lam * self.tau
        l1 = torch.ones_like(x)
        l2 = torch.where(x == 0.0, torch.ones_like(x), (1.0 - torch.exp(-x)) / x)
        l3 = l2 - torch.exp(-x)
        return torch.cat([l1, l2, l3], dim=0)

    def forward(self, x):
        h = self.backbone(x)
        f = self.factor_head(h)
        B = self.ns_loadings()
        y = f @ B
        return y, f


def train_dlns_model(X_train: np.ndarray, Y_train: np.ndarray, maturities: np.ndarray, cfg: DLNSConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    x_scaler = StandardScaler().fit(X_train)
    Xs = x_scaler.transform(X_train).astype(np.float32)

    val_size = min(cfg.val_size, max(12, Xs.shape[0] // 5))
    split = Xs.shape[0] - val_size
    if split <= 10:
        raise ValueError("Training sample too short for validation split.")

    X_tr, X_val = Xs[:split], Xs[split:]
    Y_tr, Y_val = Y_train[:split].astype(np.float32), Y_train[split:].astype(np.float32)

    model = DLNSNet(Xs.shape[1], maturities, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr))
    tr_loader = DataLoader(tr_ds, batch_size=min(cfg.batch_size, len(tr_ds)), shuffle=False)
    X_val_t = torch.tensor(X_val)
    Y_val_t = torch.tensor(Y_val)

    best_state = None
    best_val = np.inf
    patience_left = cfg.patience

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            yhat, _ = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            yv, _ = model(X_val_t)
            val_loss = float(loss_fn(yv, Y_val_t).item())

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, x_scaler


def predict_dlns(model: DLNSNet, x_scaler: StandardScaler, X: np.ndarray):
    model.eval()
    Xs = x_scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        yhat, fhat = model(torch.tensor(Xs))
    return yhat.numpy(), fhat.numpy(), float(model.current_lambda().detach().cpu().item())


def aggregate_metrics(y_true: np.ndarray, y_hat: np.ndarray, y_bench: np.ndarray) -> Dict[str, np.ndarray | float]:
    err_model = y_true - y_hat
    err_bench = y_true - y_bench

    mse_model = np.nanmean(err_model ** 2, axis=0)
    mse_bench = np.nanmean(err_bench ** 2, axis=0)

    return {
        "MSE_model": mse_model,
        "MSE_bench": mse_bench,
        "RMSE_model": np.sqrt(mse_model),
        "RMSE_bench": np.sqrt(mse_bench),
        "MAE_model": np.nanmean(np.abs(err_model), axis=0),
        "MAE_bench": np.nanmean(np.abs(err_bench), axis=0),
        "R2OOS": 1.0 - mse_model / mse_bench,
        "R2OOS_AGG": float(1.0 - np.nansum(err_model ** 2) / np.nansum(err_bench ** 2)),
    }


def create_summary_table(metrics: Dict[str, np.ndarray | float], maturity_names: Sequence[str], maturities_months: np.ndarray,
                         cfg: DLNSConfig, source_used: str, n_oos: int) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(maturity_names):
        rows.append({
            "model": cfg.model_name,
            "benchmark": "RandomWalk",
            "target_group": "yield",
            "target_name": name,
            "maturity_months": int(maturities_months[i]),
            "horizon": cfg.horizon,
            "n_oos": n_oos,
            "rmse_model": float(metrics["RMSE_model"][i]),
            "rmse_benchmark": float(metrics["RMSE_bench"][i]),
            "mae_model": float(metrics["MAE_model"][i]),
            "mae_benchmark": float(metrics["MAE_bench"][i]),
            "mse_model": float(metrics["MSE_model"][i]),
            "mse_benchmark": float(metrics["MSE_bench"][i]),
            "oos_r2": float(metrics["R2OOS"][i]),
            "source_used": source_used,
            "run_tag": cfg.run_tag,
        })

    rows.append({
        "model": cfg.model_name,
        "benchmark": "RandomWalk",
        "target_group": "yield_agg",
        "target_name": "ALL",
        "maturity_months": -1,
        "horizon": cfg.horizon,
        "n_oos": n_oos,
        "rmse_model": float(np.nanmean(metrics["RMSE_model"])),
        "rmse_benchmark": float(np.nanmean(metrics["RMSE_bench"])),
        "mae_model": float(np.nanmean(metrics["MAE_model"])),
        "mae_benchmark": float(np.nanmean(metrics["MAE_bench"])),
        "mse_model": float(np.nanmean(metrics["MSE_model"])),
        "mse_benchmark": float(np.nanmean(metrics["MSE_bench"])),
        "oos_r2": float(metrics["R2OOS_AGG"]),
        "source_used": source_used,
        "run_tag": cfg.run_tag,
    })

    return pd.DataFrame(rows)


def run_oos_forecast(cfg: DLNSConfig) -> Dict[str, object]:
    dates, yields, maturities, names, source_used = load_yield_panel(cfg)
    oos_start_ts = parse_oos_start(cfg.oos_start)

    X_all, Y_all, origin_idx = build_supervised_dataset(yields, cfg.horizon, cfg.lag_window)
    origin_dates = dates[origin_idx]
    target_dates = dates[origin_idx + cfg.horizon]

    oos_mask = origin_dates >= oos_start_ts
    oos_positions = np.where(oos_mask)[0]
    if len(oos_positions) == 0:
        raise ValueError("No OOS observations after oos_start.")

    Y_pred = np.full_like(Y_all, np.nan)
    Y_rw = np.full_like(Y_all, np.nan)
    F_pred = np.full((Y_all.shape[0], 3), np.nan, dtype=np.float32)
    lambda_path = np.full(Y_all.shape[0], np.nan)

    last_fit_pos = -10**9
    model = None
    scaler = None

    for pos in oos_positions:
        if pos < cfg.min_train_obs:
            continue

        if model is None or (pos - last_fit_pos) >= cfg.reestimate_every:
            model, scaler = train_dlns_model(X_all[:pos], Y_all[:pos], maturities, cfg)
            last_fit_pos = pos

        yhat, fhat, lam = predict_dlns(model, scaler, X_all[pos:pos + 1])
        Y_pred[pos] = yhat[0]
        Y_rw[pos] = X_all[pos, -len(maturities):]
        F_pred[pos] = fhat[0]
        lambda_path[pos] = lam

    keep = oos_mask & np.isfinite(Y_pred).all(axis=1)
    if keep.sum() == 0:
        raise ValueError("No valid OOS forecasts were produced.")

    Y_true_oos = Y_all[keep]
    Y_pred_oos = Y_pred[keep]
    Y_rw_oos = Y_rw[keep]
    F_pred_oos = F_pred[keep]
    origin_dates_oos = origin_dates[keep]
    target_dates_oos = target_dates[keep]
    lambda_oos = lambda_path[keep]

    metrics = aggregate_metrics(Y_true_oos, Y_pred_oos, Y_rw_oos)
    summary = create_summary_table(metrics, names, maturities, cfg, source_used, int(keep.sum()))

    annual_prev = [m for m in maturities if m % 12 == 0 and m >= 12 and (m + 12) in set(maturities)]
    dy_names, dy_true_list, dy_hat_list, dy_rw_list = [], [], [], []
    maturity_to_idx = {m: i for i, m in enumerate(maturities.tolist())}
    origin_raw = yields[origin_idx[keep]]

    for m in annual_prev:
        j = maturity_to_idx[m]
        dy_names.append(f"dy_{m // 12}")
        dy_true_list.append(Y_true_oos[:, j] - origin_raw[:, j])
        dy_hat_list.append(Y_pred_oos[:, j] - origin_raw[:, j])
        dy_rw_list.append(Y_rw_oos[:, j] - origin_raw[:, j])

    compat = None
    if dy_names:
        compat = {
            "Dates": np.array([d.strftime("%Y-%m-%d") for d in target_dates_oos], dtype=object),
            "Y_Columns": np.array(dy_names, dtype=object),
            "Y_True": np.column_stack(dy_true_list),
            f"Y_forecast_agg_{cfg.model_name}": np.column_stack(dy_hat_list),
            "Y_zero_benchmark": np.column_stack(dy_rw_list),
        }

    return {
        "config": cfg,
        "yield_names": names,
        "maturities": maturities,
        "source_used": source_used,
        "origin_dates_oos": origin_dates_oos,
        "target_dates_oos": target_dates_oos,
        "Y_actual": Y_true_oos,
        "Y_forecast": Y_pred_oos,
        "Y_rw": Y_rw_oos,
        "Factor_forecast": F_pred_oos,
        "lambda_path": lambda_oos,
        "metrics": metrics,
        "summary": summary,
        "compat": compat,
    }


def save_results(results: Dict[str, object]) -> None:
    cfg: DLNSConfig = results["config"]
    ensure_dir(cfg.save_dir)

    summary_path = os.path.join(cfg.save_dir, f"summary_{cfg.run_tag}.csv")
    mat_path = os.path.join(cfg.save_dir, f"dlns_oos_{cfg.run_tag}.mat")
    compat_path = os.path.join(cfg.save_dir, f"dlns_oos_compat_{cfg.run_tag}.mat")

    results["summary"].to_csv(summary_path, index=False)

    metrics = results["metrics"]
    mdict = {
        "model_name": cfg.model_name,
        "benchmark_name": "RandomWalk",
        "target_group": "yield",
        "run_tag": cfg.run_tag,
        "source_used": results["source_used"],
        "maturity_names": np.array(results["yield_names"], dtype=object),
        "maturity_months": results["maturities"].astype(np.int32),
        "origin_yyyymm": np.array([d.year * 100 + d.month for d in results["origin_dates_oos"]], dtype=np.int32),
        "target_yyyymm": np.array([d.year * 100 + d.month for d in results["target_dates_oos"]], dtype=np.int32),
        "lambda_ns": np.asarray(cfg.lambda_ns, dtype=float),
        "lambda_path": np.asarray(results["lambda_path"], dtype=float),
        "horizon": np.asarray(cfg.horizon, dtype=np.int32),
        "oos_start": np.array(cfg.oos_start, dtype=object),
        "lag_window": np.asarray(cfg.lag_window, dtype=np.int32),
        "Y_actual": results["Y_actual"],
        "Y_zero_benchmark": results["Y_rw"],
        f"Y_forecast_agg_{cfg.model_name}": results["Y_forecast"],
        f"Factor_forecast_{cfg.model_name}": results["Factor_forecast"],
        f"MSE_{cfg.model_name}": metrics["MSE_model"],
        "MSE_RW": metrics["MSE_bench"],
        f"RMSE_{cfg.model_name}": metrics["RMSE_model"],
        "RMSE_RW": metrics["RMSE_bench"],
        f"MAE_{cfg.model_name}": metrics["MAE_model"],
        "MAE_RW": metrics["MAE_bench"],
        f"R2OOS_{cfg.model_name}": metrics["R2OOS"],
        f"R2OOS_{cfg.model_name}_AGG": np.asarray(metrics["R2OOS_AGG"], dtype=float),
    }
    sio.savemat(mat_path, mdict, do_compression=True)

    if results.get("compat") is not None:
        sio.savemat(compat_path, results["compat"], do_compression=True)

    print(f"Saved summary: {summary_path}")
    print(f"Saved mat: {mat_path}")
    if results.get("compat") is not None:
        print(f"Saved compat mat: {compat_path}")


def parse_hidden_layers(s: str) -> Tuple[int, ...]:
    vals = [int(x) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("hidden_layers cannot be empty")
    return tuple(vals)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DLNS OOS forecasting, Lee (2023)-style")
    p.add_argument("--mat_path", type=str, default="data/target_and_features.mat")
    p.add_argument("--csv_path", type=str, default="data/dataset.csv")
    p.add_argument("--prefer_mat", action="store_true", default=True)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--oos_start", type=str, default="1989-01")
    p.add_argument("--min_train_obs", type=int, default=180)
    p.add_argument("--reestimate_every", type=int, default=1)
    p.add_argument("--maturities", type=str, default=None)
    p.add_argument("--lag_window", type=int, default=12)
    p.add_argument("--lambda_ns", type=float, default=0.0609)
    p.add_argument("--trainable_lambda", action="store_true")
    p.add_argument("--hidden_layers", type=str, default="64,32")
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val_size", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="results_dlns")
    p.add_argument("--run_tag", type=str, default="dlns_lee2023_h12")
    p.add_argument("--model_name", type=str, default="DLNSModel")
    return p


def main():
    args = build_arg_parser().parse_args()

    maturities = None
    if args.maturities:
        maturities = [int(x) for x in args.maturities.split(",") if x.strip()]

    cfg = DLNSConfig(
        mat_path=args.mat_path,
        csv_path=args.csv_path,
        prefer_mat=args.prefer_mat,
        horizon=args.horizon,
        oos_start=args.oos_start,
        min_train_obs=args.min_train_obs,
        reestimate_every=args.reestimate_every,
        use_maturities=maturities,
        lag_window=args.lag_window,
        lambda_ns=args.lambda_ns,
        trainable_lambda=args.trainable_lambda,
        hidden_layers=parse_hidden_layers(args.hidden_layers),
        activation=args.activation,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        val_size=args.val_size,
        seed=args.seed,
        save_dir=args.save_dir,
        run_tag=args.run_tag,
        model_name=args.model_name,
    )

    results = run_oos_forecast(cfg)
    save_results(results)
    print(f"Aggregate OOS R2 vs RW: {float(results['metrics']['R2OOS_AGG']):.6f}")


if __name__ == "__main__":
    main()
