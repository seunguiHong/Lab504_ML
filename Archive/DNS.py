#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DNS OOS Yield Forecasting (12-month horizon by default)
-------------------------------------------------------
Single-file implementation for expanding-window out-of-sample forecasting
with a 2-step Dynamic Nelson-Siegel (DNS) model.

Pipeline
1) Load yields from target_and_features.mat (preferred) or dataset.csv.
2) For each OOS origin t:
   a. Estimate Nelson-Siegel factors (level, slope, curvature) on each
      in-sample date by cross-sectional OLS with fixed lambda.
   b. Fit VAR(1) to the factor time series.
   c. Produce h-step-ahead factor forecasts.
   d. Map factor forecasts back into yields.
   e. Compare against Random Walk benchmark.
3) Save forecasts, errors, and summary metrics.

Notes
- Default maturities are all available columns in X.yields from the .mat file.
- Default horizon is 12 months.
- Default OOS start is 1989-01.
- Default lambda is 0.0609 when maturities are measured in months,
  which implies the curvature loading peaks around 30 months.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io.matlab import mat_struct


# =========================================================
# Configuration
# =========================================================

@dataclass
class DNSConfig:
    mat_path: str = "data/target_and_features.mat"
    csv_path: str = "data/dataset.csv"
    prefer_mat: bool = True
    horizon: int = 12
    oos_start: str = "1989-01"
    lambda_ns: float = 0.0609
    reestimate_every: int = 1
    min_train_obs: int = 120
    use_maturities: Optional[List[int]] = None  # in months, e.g. [12,24,36,...,120]
    save_dir: str = "results_dns"
    run_tag: str = "dns_yield_h12"
    model_name: str = "DNSModel"


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def yyyymm_to_timestamp(values: Sequence[int]) -> pd.DatetimeIndex:
    values = np.asarray(values).astype(int)
    years = values // 100
    months = values % 100
    dates = [pd.Timestamp(year=int(y), month=int(m), day=1) + pd.offsets.MonthEnd(0)
             for y, m in zip(years, months)]
    return pd.DatetimeIndex(dates)


def timestamp_to_yyyymm(ts: pd.Timestamp) -> int:
    return ts.year * 100 + ts.month


def parse_oos_start(oos_start: str) -> pd.Timestamp:
    parts = oos_start.split("-")
    if len(parts) != 2:
        raise ValueError(f"oos_start must be YYYY-MM, got: {oos_start}")
    y, m = int(parts[0]), int(parts[1])
    return pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)


def is_mat_struct(x) -> bool:
    return isinstance(x, mat_struct)


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


def safe_mat_struct_get(obj, field: str):
    if not hasattr(obj, field):
        raise AttributeError(f"MAT struct missing field '{field}'.")
    return getattr(obj, field)


# =========================================================
# Data loading
# =========================================================

def load_from_mat(mat_path: str,
                  use_maturities: Optional[Sequence[int]] = None) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, List[str]]:
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "X" not in mat:
        raise KeyError("MAT file must contain 'X'.")

    X = mat["X"]
    yields_struct = safe_mat_struct_get(X, "yields")
    time_raw = np.asarray(safe_mat_struct_get(yields_struct, "Time")).astype(int)
    y_raw = np.asarray(safe_mat_struct_get(yields_struct, "data"), dtype=float)
    names = [str(x) for x in np.asarray(safe_mat_struct_get(yields_struct, "names")).tolist()]

    col_idx = select_columns_by_maturity(names, use_maturities)
    dates = yyyymm_to_timestamp(time_raw)
    yields = y_raw[:, col_idx]
    names_sel = [names[i] for i in col_idx]
    maturities_months = np.array([int(name[1:]) for name in names_sel], dtype=int)
    return dates, yields, maturities_months, names_sel


def load_from_csv(csv_path: str,
                  use_maturities: Optional[Sequence[int]] = None) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if "Time" not in df.columns:
        raise KeyError("CSV file must contain 'Time' column in YYYYMM format.")

    yield_cols = [c for c in df.columns if c.startswith("m") and len(c) == 4 and c[1:].isdigit()]
    yield_cols = sorted(yield_cols, key=lambda x: int(x[1:]))
    if not yield_cols:
        raise ValueError("No yield columns like m001, m012, ..., m360 found in CSV.")

    col_idx = select_columns_by_maturity(yield_cols, use_maturities)
    names_sel = [yield_cols[i] for i in col_idx]
    maturities_months = np.array([int(name[1:]) for name in names_sel], dtype=int)

    dates = yyyymm_to_timestamp(df["Time"].to_numpy())
    yields = df[names_sel].to_numpy(dtype=float)
    return dates, yields, maturities_months, names_sel


def load_yield_panel(config: DNSConfig) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, List[str], str]:
    errors = []

    if config.prefer_mat:
        try:
            dates, yields, maturities, names = load_from_mat(config.mat_path, config.use_maturities)
            return dates, yields, maturities, names, "mat"
        except Exception as e:
            errors.append(f"MAT load failed: {e}")

    try:
        dates, yields, maturities, names = load_from_csv(config.csv_path, config.use_maturities)
        return dates, yields, maturities, names, "csv"
    except Exception as e:
        errors.append(f"CSV load failed: {e}")

    if not config.prefer_mat:
        try:
            dates, yields, maturities, names = load_from_mat(config.mat_path, config.use_maturities)
            return dates, yields, maturities, names, "mat"
        except Exception as e:
            errors.append(f"MAT load failed after CSV fallback: {e}")

    msg = " | ".join(errors)
    raise RuntimeError(f"Unable to load yield panel. {msg}")


# =========================================================
# DNS model components
# =========================================================

def ns_loadings(maturities_months: np.ndarray, lambda_ns: float) -> np.ndarray:
    tau = np.asarray(maturities_months, dtype=float)
    x = lambda_ns * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        l2 = np.where(x == 0.0, 1.0, (1.0 - np.exp(-x)) / x)
    l3 = l2 - np.exp(-x)
    return np.column_stack([np.ones_like(tau), l2, l3])


def estimate_dns_factors_ols(yields: np.ndarray,
                             maturities_months: np.ndarray,
                             lambda_ns: float) -> np.ndarray:
    """
    Cross-sectional OLS at each date using available maturities only:
    y_t = B_t * beta_t + e_t
    returns factors with shape (T, 3)
    """
    B_full = ns_loadings(maturities_months, lambda_ns)
    T = yields.shape[0]
    betas = np.full((T, 3), np.nan, dtype=float)

    for t in range(T):
        y_t = yields[t]
        mask = np.isfinite(y_t)
        if mask.sum() < 3:
            continue
        B_t = B_full[mask]
        y_obs = y_t[mask]
        coef, *_ = np.linalg.lstsq(B_t, y_obs, rcond=None)
        betas[t] = coef

    return betas


def fit_var1(factors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    factors_t = c + A factors_{t-1} + u_t
    returns intercept c shape (3,), A shape (3,3)
    Uses only adjacent rows with finite factors.
    """
    X = factors[:-1]
    Y = factors[1:]
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[mask]
    Y = Y[mask]
    if len(X) < 10:
        raise ValueError("Too few valid factor observations to estimate VAR(1).")
    Z = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    c = coef[0]
    A = coef[1:].T
    return c, A


def var1_forecast(last_factor: np.ndarray,
                  c: np.ndarray,
                  A: np.ndarray,
                  horizon: int) -> np.ndarray:
    f = np.asarray(last_factor, dtype=float).copy()
    for _ in range(horizon):
        f = c + A @ f
    return f


def reconstruct_yields_from_factors(factor: np.ndarray,
                                    maturities_months: np.ndarray,
                                    lambda_ns: float) -> np.ndarray:
    B = ns_loadings(maturities_months, lambda_ns)
    return B @ factor


# =========================================================
# Metrics
# =========================================================

def rmse(actual: np.ndarray, forecast: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sqrt(np.nanmean((actual - forecast) ** 2, axis=axis))


def mae(actual: np.ndarray, forecast: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.nanmean(np.abs(actual - forecast), axis=axis)


def oos_r2(actual: np.ndarray, model_fcst: np.ndarray, benchmark_fcst: np.ndarray, axis: int = 0) -> np.ndarray:
    sse_model = np.nansum((actual - model_fcst) ** 2, axis=axis)
    sse_bench = np.nansum((actual - benchmark_fcst) ** 2, axis=axis)
    return 1.0 - (sse_model / sse_bench)


def mse_by_col(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return np.nanmean((actual - forecast) ** 2, axis=0)


def overall_oos_r2(actual: np.ndarray, model_fcst: np.ndarray, benchmark_fcst: np.ndarray) -> float:
    sse_model = np.nansum((actual - model_fcst) ** 2)
    sse_bench = np.nansum((actual - benchmark_fcst) ** 2)
    return float(1.0 - sse_model / sse_bench)


# =========================================================
# OOS pipeline
# =========================================================

def run_dns_oos(config: DNSConfig) -> Dict[str, object]:
    dates, yields, maturities_months, maturity_names, source_used = load_yield_panel(config)

    # Basic checks
    if yields.ndim != 2:
        raise ValueError("Yield panel must be 2-dimensional.")
    if len(dates) != yields.shape[0]:
        raise ValueError("Dates length must match number of rows in yields.")
    if yields.shape[1] < 3:
        raise ValueError("Need at least 3 maturities to estimate DNS factors.")

    oos_start_ts = parse_oos_start(config.oos_start)
    valid_origins = np.where(dates >= oos_start_ts)[0]
    valid_origins = valid_origins[valid_origins + config.horizon < len(dates)]

    if len(valid_origins) == 0:
        raise ValueError("No valid OOS origins. Check oos_start and horizon.")

    # Need enough initial training observations for factor VAR estimation.
    valid_origins = valid_origins[valid_origins >= config.min_train_obs]
    if len(valid_origins) == 0:
        raise ValueError("No valid OOS origins after applying min_train_obs.")

    forecast_dns = np.full((len(valid_origins), yields.shape[1]), np.nan)
    forecast_rw = np.full((len(valid_origins), yields.shape[1]), np.nan)
    actual_y = np.full((len(valid_origins), yields.shape[1]), np.nan)
    factor_forecasts = np.full((len(valid_origins), 3), np.nan)
    origin_y = np.full((len(valid_origins), yields.shape[1]), np.nan)
    implied_dy = np.full((len(valid_origins), yields.shape[1]), np.nan)
    actual_dy = np.full((len(valid_origins), yields.shape[1]), np.nan)
    rw_dy = np.full((len(valid_origins), yields.shape[1]), np.nan)
    origin_dates = []
    target_dates = []

    cached = {}

    for j, t0 in enumerate(valid_origins):
        train_end = t0  # inclusive origin uses data up to t0
        retrain = (j == 0) or (j % config.reestimate_every == 0)

        if retrain:
            y_train = yields[:train_end + 1]
            betas_train = estimate_dns_factors_ols(y_train, maturities_months, config.lambda_ns)
            c, A = fit_var1(betas_train)
            cached = {
                "betas_train": betas_train,
                "c": c,
                "A": A,
            }

        betas_train = cached["betas_train"]
        c = cached["c"]
        A = cached["A"]

        last_factor = betas_train[-1]
        beta_fcst = var1_forecast(last_factor, c, A, config.horizon)
        y_fcst_dns = reconstruct_yields_from_factors(beta_fcst, maturities_months, config.lambda_ns)
        y_fcst_rw = yields[t0].copy()
        y_true = yields[t0 + config.horizon].copy()

        forecast_dns[j] = y_fcst_dns
        forecast_rw[j] = y_fcst_rw
        actual_y[j] = y_true
        origin_y[j] = yields[t0].copy()
        implied_dy[j] = y_fcst_dns - yields[t0].copy()
        actual_dy[j] = y_true - yields[t0].copy()
        rw_dy[j] = y_fcst_rw - yields[t0].copy()
        factor_forecasts[j] = beta_fcst
        origin_dates.append(dates[t0])
        target_dates.append(dates[t0 + config.horizon])

        if (j + 1) % 25 == 0 or j == len(valid_origins) - 1:
            print(f"Processed {j + 1}/{len(valid_origins)} OOS forecasts. "
                  f"Origin={dates[t0].strftime('%Y-%m')}, Target={dates[t0 + config.horizon].strftime('%Y-%m')}")

    dns_rmse = rmse(actual_y, forecast_dns, axis=0)
    rw_rmse = rmse(actual_y, forecast_rw, axis=0)
    dns_mae = mae(actual_y, forecast_dns, axis=0)
    rw_mae = mae(actual_y, forecast_rw, axis=0)
    dns_oos_r2 = oos_r2(actual_y, forecast_dns, forecast_rw, axis=0)

    mse_dns = mse_by_col(actual_y, forecast_dns)
    mse_rw = mse_by_col(actual_y, forecast_rw)
    overall_r2 = overall_oos_r2(actual_y, forecast_dns, forecast_rw)

    summary_by_maturity = pd.DataFrame({
        "model": ["DNS"] * len(maturity_names),
        "benchmark": ["RW"] * len(maturity_names),
        "target_group": ["yields"] * len(maturity_names),
        "target_name": maturity_names,
        "maturity_months": maturities_months,
        "horizon": [config.horizon] * len(maturity_names),
        "n_oos": [actual_y.shape[0]] * len(maturity_names),
        "rmse_model": dns_rmse,
        "rmse_benchmark": rw_rmse,
        "mae_model": dns_mae,
        "mae_benchmark": rw_mae,
        "mse_model": mse_dns,
        "mse_benchmark": mse_rw,
        "oos_r2": dns_oos_r2,
        "source_used": [source_used] * len(maturity_names),
        "run_tag": [config.run_tag] * len(maturity_names),
    })

    summary_agg = pd.DataFrame({
        "model": ["DNS"],
        "benchmark": ["RW"],
        "target_group": ["yields"],
        "target_name": ["ALL_MATURITIES"],
        "maturity_months": [np.nan],
        "horizon": [config.horizon],
        "n_oos": [actual_y.shape[0]],
        "rmse_model": [np.nanmean(dns_rmse)],
        "rmse_benchmark": [np.nanmean(rw_rmse)],
        "mae_model": [np.nanmean(dns_mae)],
        "mae_benchmark": [np.nanmean(rw_mae)],
        "mse_model": [np.nanmean(mse_dns)],
        "mse_benchmark": [np.nanmean(mse_rw)],
        "oos_r2": [overall_r2],
        "source_used": [source_used],
        "run_tag": [config.run_tag],
    })

    summary_full = pd.concat([summary_by_maturity, summary_agg], ignore_index=True)

    forecasts_df = pd.DataFrame({
        "origin_date": origin_dates,
        "target_date": target_dates,
    })
    for i, name in enumerate(maturity_names):
        forecasts_df[f"actual_{name}"] = actual_y[:, i]
        forecasts_df[f"dns_{name}"] = forecast_dns[:, i]
        forecasts_df[f"rw_{name}"] = forecast_rw[:, i]

    factor_df = pd.DataFrame({
        "origin_date": origin_dates,
        "target_date": target_dates,
        "beta1_fcst": factor_forecasts[:, 0],
        "beta2_fcst": factor_forecasts[:, 1],
        "beta3_fcst": factor_forecasts[:, 2],
    })

    results = {
        "config": config,
        "source_used": source_used,
        "dates_all": dates,
        "maturity_names": maturity_names,
        "maturity_months": maturities_months,
        "origin_dates": np.array(origin_dates, dtype="datetime64[ns]"),
        "target_dates": np.array(target_dates, dtype="datetime64[ns]"),
        "actual_y": actual_y,
        "forecast_dns": forecast_dns,
        "forecast_rw": forecast_rw,
        "factor_forecasts": factor_forecasts,
        "origin_y": origin_y,
        "implied_dy": implied_dy,
        "actual_dy": actual_dy,
        "rw_dy": rw_dy,
        "mse_dns": mse_dns,
        "mse_rw": mse_rw,
        "rmse_dns": dns_rmse,
        "rmse_rw": rw_rmse,
        "mae_dns": dns_mae,
        "mae_rw": rw_mae,
        "r2oos_dns": dns_oos_r2,
        "r2oos_dns_agg": overall_r2,
        "summary": summary_full,
        "forecasts_df": forecasts_df,
        "factor_df": factor_df,
    }
    return results


# =========================================================
# Saving
# =========================================================

def _annual_dy_compat_arrays(results: Dict[str, object]) -> Tuple[np.ndarray, List[str]]:
    """
    Build annual-maturity dy arrays compatible with plot_rx_dy.m.
    For maturity m=12*k months, create column dy_k = y_{t+12}^{(kY)} - y_t^{(kY)}.
    """
    mats = np.asarray(results["maturity_months"], dtype=int)
    annual_idx = [i for i, m in enumerate(mats) if m % 12 == 0 and m >= 12]
    if not annual_idx:
        return np.empty((results["actual_dy"].shape[0], 0)), []
    annual_idx = sorted(annual_idx, key=lambda i: mats[i])
    colnames = [f"dy_{mats[i] // 12}" for i in annual_idx]
    return np.asarray(results["actual_dy"][:, annual_idx], dtype=float), colnames


def save_results(results: Dict[str, object], config: DNSConfig) -> Dict[str, str]:
    ensure_dir(config.save_dir)

    summary_path = os.path.join(config.save_dir, f"summary_{config.run_tag}.csv")
    forecasts_path = os.path.join(config.save_dir, f"forecasts_{config.run_tag}.csv")
    factors_path = os.path.join(config.save_dir, f"factor_forecasts_{config.run_tag}.csv")
    mat_path = os.path.join(config.save_dir, f"dns_oos_{config.run_tag}.mat")
    compat_mat_path = os.path.join(config.save_dir, f"dns_oos_compat_{config.run_tag}.mat")

    results["summary"].to_csv(summary_path, index=False)
    results["forecasts_df"].to_csv(forecasts_path, index=False)
    results["factor_df"].to_csv(factors_path, index=False)

    maturity_names_obj = np.asarray(results["maturity_names"], dtype=object)
    save_dict = {
        "model_name": np.asarray([config.model_name], dtype=object),
        "benchmark_name": np.asarray(["RW"], dtype=object),
        "target_group": np.asarray(["yields"], dtype=object),
        "run_tag": np.asarray([config.run_tag], dtype=object),
        "source_used": np.asarray([results["source_used"]], dtype=object),
        "maturity_names": maturity_names_obj,
        "maturity_months": np.asarray(results["maturity_months"], dtype=int),
        "origin_yyyymm": np.array([timestamp_to_yyyymm(pd.Timestamp(x)) for x in results["origin_dates"]], dtype=int),
        "target_yyyymm": np.array([timestamp_to_yyyymm(pd.Timestamp(x)) for x in results["target_dates"]], dtype=int),
        "Y_actual": np.asarray(results["actual_y"], dtype=float),
        "Y_zero_benchmark": np.asarray(results["forecast_rw"], dtype=float),
        "Y_forecast_agg_DNS": np.asarray(results["forecast_dns"], dtype=float),
        "Factor_forecast_DNS": np.asarray(results["factor_forecasts"], dtype=float),
        "MSE_DNS": np.asarray(results["mse_dns"], dtype=float).reshape(1, -1),
        "MSE_RW": np.asarray(results["mse_rw"], dtype=float).reshape(1, -1),
        "RMSE_DNS": np.asarray(results["rmse_dns"], dtype=float).reshape(1, -1),
        "RMSE_RW": np.asarray(results["rmse_rw"], dtype=float).reshape(1, -1),
        "MAE_DNS": np.asarray(results["mae_dns"], dtype=float).reshape(1, -1),
        "MAE_RW": np.asarray(results["mae_rw"], dtype=float).reshape(1, -1),
        "R2OOS_DNS": np.asarray(results["r2oos_dns"], dtype=float).reshape(1, -1),
        "R2OOS_DNS_AGG": np.array([[results["r2oos_dns_agg"]]], dtype=float),
        "lambda_ns": np.array([[config.lambda_ns]], dtype=float),
        "horizon": np.array([[config.horizon]], dtype=int),
        "oos_start": np.asarray([config.oos_start], dtype=object),
        "min_train_obs": np.array([[config.min_train_obs]], dtype=int),
        "reestimate_every": np.array([[config.reestimate_every]], dtype=int),
    }
    sio.savemat(mat_path, save_dict, do_compression=True)

    # Compatibility MAT for downstream MATLAB analysis like plot_rx_dy.m
    dy_true_annual, dy_cols = _annual_dy_compat_arrays(results)
    mats = np.asarray(results["maturity_months"], dtype=int)
    annual_idx = [i for i, m in enumerate(mats) if m % 12 == 0 and m >= 12]
    dy_hat_annual = np.asarray(results["implied_dy"][:, annual_idx], dtype=float) if annual_idx else np.empty((results["implied_dy"].shape[0], 0))
    dy_rw_annual = np.asarray(results["rw_dy"][:, annual_idx], dtype=float) if annual_idx else np.empty((results["rw_dy"].shape[0], 0))
    compat_save = {
        "Dates": np.asarray(pd.to_datetime(results["target_dates"]).strftime("%Y-%m-%d"), dtype=object),
        "Y_Columns": np.asarray(dy_cols, dtype=object),
        "Y_True": dy_true_annual,
        f"Y_forecast_agg_{config.model_name}": dy_hat_annual,
        "Y_zero_benchmark": dy_rw_annual,
        "Origin_Dates": np.asarray(pd.to_datetime(results["origin_dates"]).strftime("%Y-%m-%d"), dtype=object),
        "Yield_Columns": np.asarray(results["maturity_names"], dtype=object),
        "Yield_Actual": np.asarray(results["actual_y"], dtype=float),
        f"Yield_forecast_agg_{config.model_name}": np.asarray(results["forecast_dns"], dtype=float),
        "Yield_zero_benchmark": np.asarray(results["forecast_rw"], dtype=float),
        "model_name": np.asarray([config.model_name], dtype=object),
        "horizon": np.array([[config.horizon]], dtype=int),
        "run_tag": np.asarray([config.run_tag], dtype=object),
    }
    sio.savemat(compat_mat_path, compat_save, do_compression=True)

    return {
        "summary_csv": summary_path,
        "forecasts_csv": forecasts_path,
        "factors_csv": factors_path,
        "mat": mat_path,
        "compat_mat": compat_mat_path,
    }


# =========================================================
# CLI
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DNS OOS yield forecasting with random-walk benchmark.")
    parser.add_argument("--mat_path", type=str, default="data/target_and_features.mat")
    parser.add_argument("--csv_path", type=str, default="data/dataset.csv")
    parser.add_argument("--prefer_mat", type=int, default=1, help="1: use .mat first, 0: use .csv first")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--oos_start", type=str, default="1989-01")
    parser.add_argument("--lambda_ns", type=float, default=0.0609)
    parser.add_argument("--reestimate_every", type=int, default=1)
    parser.add_argument("--min_train_obs", type=int, default=120)
    parser.add_argument("--save_dir", type=str, default="results_dns")
    parser.add_argument("--run_tag", type=str, default="dns_yield_h12")
    parser.add_argument("--model_name", type=str, default="DNSModel")
    parser.add_argument(
        "--maturities",
        type=str,
        default="",
        help="Comma-separated maturities in months, e.g. '12,24,36,60,84,120'. Leave empty for all maturities.",
    )
    return parser


def parse_args_to_config(args: argparse.Namespace) -> DNSConfig:
    maturities = None
    if args.maturities.strip():
        maturities = [int(x.strip()) for x in args.maturities.split(",") if x.strip()]
    return DNSConfig(
        mat_path=args.mat_path,
        csv_path=args.csv_path,
        prefer_mat=bool(args.prefer_mat),
        horizon=int(args.horizon),
        oos_start=args.oos_start,
        lambda_ns=float(args.lambda_ns),
        reestimate_every=int(args.reestimate_every),
        min_train_obs=int(args.min_train_obs),
        use_maturities=maturities,
        save_dir=args.save_dir,
        run_tag=args.run_tag,
        model_name=args.model_name,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = parse_args_to_config(args)

    print("=" * 80)
    print("Running DNS OOS yield forecasting")
    print(config)
    print("=" * 80)

    results = run_dns_oos(config)
    saved = save_results(results, config)

    print("\nSource used:", results["source_used"])
    print("\nSummary (last row is aggregate across maturities):")
    print(results["summary"].tail(10).to_string(index=False))
    print("\nSaved files:")
    for k, v in saved.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
