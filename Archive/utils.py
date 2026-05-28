#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import multiprocessing
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm


def get_cpu_count() -> int:
    return max(1, multiprocessing.cpu_count())


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_dump_dir(prefix: str = "trainingDumps_") -> str:
    return tempfile.mkdtemp(prefix=prefix)


def safe_rmtree(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def matlab_names_to_list(x) -> List[str]:
    arr = np.asarray(x, dtype=object).reshape(-1)

    names = []
    for item in arr:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        elif isinstance(item, np.ndarray):
            if item.size == 1:
                names.append(str(item.item()))
            else:
                names.append("".join(item.astype(str).reshape(-1)).strip())
        else:
            names.append(str(item))

    return names


def yyyymm_to_month_end(yyyymm) -> pd.DatetimeIndex:
    yyyymm = np.asarray(yyyymm).reshape(-1).astype(int)
    year = yyyymm // 100
    month = yyyymm % 100

    dates = pd.to_datetime(
        {"year": year, "month": month, "day": np.ones_like(year)},
        errors="raise",
    )

    return pd.DatetimeIndex(dates) + pd.offsets.MonthEnd(0)


def load_mat_group_frame(data_path: str, root_name: str, group_name: str) -> pd.DataFrame:
    mat = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)

    root = mat[root_name]
    block = getattr(root, group_name)

    data = np.asarray(block.data, dtype=float)
    dates = yyyymm_to_month_end(block.Time)
    names = matlab_names_to_list(block.names)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    df = pd.DataFrame(data, index=dates, columns=names)
    df.index.name = "Date"

    return df


def subset_target_columns(
    Y_df: pd.DataFrame,
    target_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    if target_indices is None:
        return Y_df.copy()

    return Y_df.iloc[:, list(target_indices)].copy()


def align_feature_target_frames(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = X_df.join(Y_df, how="inner")

    if dropna:
        merged = merged.dropna(axis=0, how="any")

    return merged[X_df.columns].copy(), merged[Y_df.columns].copy()


def load_dataset(
    data_path: str,
    feature_groups: Sequence[str],
    target_group: str,
    target_indices: Optional[Sequence[int]] = None,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_df = pd.concat(
        [load_mat_group_frame(data_path, "X", group) for group in feature_groups],
        axis=1,
    )

    Y_df = load_mat_group_frame(data_path, "y", target_group)
    Y_df = subset_target_columns(Y_df, target_indices)

    return align_feature_target_frames(X_df, Y_df, dropna=dropna)


def locate_oos_start_index(dates, oos_start: str) -> int:
    dates = pd.DatetimeIndex(dates)
    idx = np.where(dates >= pd.Timestamp(oos_start))[0]

    if idx.size == 0:
        raise ValueError("No available sample date on or after oos_start.")

    return int(idx[0])


def enumerate_oos_forecast_indices(dates, oos_start: str) -> List[int]:
    dates = pd.DatetimeIndex(dates)
    start_idx = locate_oos_start_index(dates, oos_start)

    return list(range(start_idx, len(dates)))


def compute_training_end_index(forecast_index: int, horizon: int) -> int:
    return int(forecast_index - horizon)


def build_purged_validation_slices(
    n_observations: int,
    validation_fraction: float,
    purge_size: int,
) -> Tuple[slice, slice]:
    val_len = int(np.ceil(n_observations * float(validation_fraction)))
    val_len = max(val_len, 1)

    fit_end = n_observations - int(purge_size) - val_len
    val_start = fit_end + int(purge_size)

    if fit_end < 2 or val_start >= n_observations:
        raise ValueError("Invalid purged validation split.")

    return slice(0, fit_end), slice(val_start, n_observations)


def fit_feature_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)

    if X.ndim == 2:
        X_flat = X
    else:
        X_flat = X.reshape(-1, X.shape[-1])

    mean_ = np.nanmean(X_flat, axis=0)
    std_ = np.nanstd(X_flat, axis=0, ddof=0)
    std_[std_ < 1e-12] = 1.0

    return mean_, std_


def apply_feature_standardizer(
    X: np.ndarray,
    mean_: np.ndarray,
    std_: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)

    if X.ndim == 2:
        return (X - mean_[None, :]) / std_[None, :]

    return (X - mean_[None, None, :]) / std_[None, None, :]


def prepare_validation_matrices(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    validation_fraction: float,
    purge_size: int,
    standardize_features: bool = False,
):
    fit_slice, val_slice = build_purged_validation_slices(
        n_observations=X_train.shape[0],
        validation_fraction=validation_fraction,
        purge_size=purge_size,
    )

    X_fit = np.asarray(X_train, dtype=float)[fit_slice]
    Y_fit = np.asarray(Y_train, dtype=float)[fit_slice]
    X_val = np.asarray(X_train, dtype=float)[val_slice]
    Y_val = np.asarray(Y_train, dtype=float)[val_slice]

    scaler = None

    if standardize_features:
        mean_, std_ = fit_feature_standardizer(X_fit)
        X_fit = apply_feature_standardizer(X_fit, mean_, std_)
        X_val = apply_feature_standardizer(X_val, mean_, std_)
        scaler = {"mean": mean_, "std": std_}

    return X_fit, Y_fit, X_val, Y_val, scaler


def prepare_final_training_matrices(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    standardize_features: bool = False,
):
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    scaler = None

    if standardize_features:
        mean_, std_ = fit_feature_standardizer(X_train)
        X_train = apply_feature_standardizer(X_train, mean_, std_)
        X_test = apply_feature_standardizer(X_test, mean_, std_)
        scaler = {"mean": mean_, "std": std_}

    return X_train, Y_train, X_test, scaler


def run_one_seed_job(args):
    model_func, seed, X_model, Y_model, params, refit, dumploc = args

    return model_func(
        X=X_model,
        Y=Y_model,
        no=seed,
        params=params,
        refit=refit,
        dumploc=dumploc,
    )


def run_seed_ensemble(
    model_func,
    ncpus: int,
    nmc: int,
    X_model: np.ndarray,
    Y_model: np.ndarray,
    params: Dict[str, Any],
    dumploc: str,
    refit: bool,
) -> Dict[int, Any]:
    nmc = int(nmc)
    n_workers = min(int(ncpus), nmc)

    jobs = [
        (model_func, seed, X_model, Y_model, params, bool(refit), dumploc)
        for seed in range(nmc)
    ]

    if n_workers <= 1:
        results = [run_one_seed_job(job) for job in jobs]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(run_one_seed_job, jobs)

    return {seed: results[seed] for seed in range(nmc)}


def normalize_l1l2(value) -> List[float]:
    arr = np.asarray(value, dtype=float).ravel()

    if arr.size == 0:
        return [0.0, 0.0]

    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]

    return [float(arr[0]), float(arr[1])]


def build_dropout_l1l2_candidates(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    dropout_raw = params.get("Dropout", 0.0)
    l1l2_raw = params.get("l1l2", [0.0, 0.0])

    if np.isscalar(dropout_raw):
        dropout_candidates = [float(dropout_raw)]
    else:
        dropout_candidates = [float(v) for v in np.asarray(dropout_raw, dtype=float).ravel()]

    l1l2_arr = np.asarray(l1l2_raw, dtype=float)

    if l1l2_arr.ndim == 1:
        l1l2_candidates = [normalize_l1l2(l1l2_arr)]
    else:
        l1l2_candidates = [normalize_l1l2(row) for row in l1l2_arr]

    candidates = []

    for dropout in dropout_candidates:
        for l1l2 in l1l2_candidates:
            candidate = copy.deepcopy(params)
            candidate["Dropout"] = float(dropout)
            candidate["l1l2"] = l1l2
            candidates.append(candidate)

    return candidates


def extract_seed_forecasts(outputs: Dict[int, Any], nmc: int) -> np.ndarray:
    forecasts = []

    for seed in range(int(nmc)):
        yhat = np.asarray(outputs[seed][0], dtype=float)

        if yhat.ndim == 1:
            yhat = yhat.reshape(1, -1)

        forecasts.append(yhat)

    return np.concatenate(forecasts, axis=0)


def extract_seed_validation_losses(outputs: Dict[int, Any], nmc: int) -> np.ndarray:
    return np.array([outputs[seed][1] for seed in range(int(nmc))], dtype=float)


def top_validation_seed_mean(
    seed_forecasts: np.ndarray,
    seed_val_loss: np.ndarray,
    navg: int,
) -> np.ndarray:
    seed_forecasts = np.asarray(seed_forecasts, dtype=float)
    seed_val_loss = np.asarray(seed_val_loss, dtype=float).reshape(-1)

    best_seed_idx = np.argsort(seed_val_loss)[: int(navg)]

    return np.mean(seed_forecasts[best_seed_idx], axis=0)


def mean_squared_error_by_target(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)

    return np.nanmean((Y_true - Y_pred) ** 2, axis=0)


def out_of_sample_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if benchmark is None:
        benchmark = np.zeros_like(y_true)
    else:
        benchmark = np.asarray(benchmark, dtype=float).reshape(-1)

    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(benchmark)

    if ok.sum() == 0:
        return np.nan

    sse_model = np.sum((y_true[ok] - y_pred[ok]) ** 2)
    sse_benchmark = np.sum((y_true[ok] - benchmark[ok]) ** 2)

    if sse_benchmark <= 0:
        return np.nan

    return 1.0 - sse_model / sse_benchmark


def out_of_sample_r2_vector(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
) -> np.ndarray:
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)

    if benchmark is None:
        benchmark = np.zeros_like(Y_true)
    else:
        benchmark = np.asarray(benchmark, dtype=float)

    return np.array(
        [
            out_of_sample_r2(Y_true[:, j], Y_pred[:, j], benchmark[:, j])
            for j in range(Y_true.shape[1])
        ]
    )


def newey_west_se_of_mean(x: np.ndarray, hac_lags: int) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    n = x.size

    if n < 2:
        return np.nan

    u = x - np.mean(x)
    lrv = np.dot(u, u) / n
    max_lag = min(int(hac_lags), n - 1)

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        lrv += 2.0 * weight * np.dot(u[lag:], u[:-lag]) / n

    if lrv < 0:
        return np.nan

    return np.sqrt(lrv / n)


def clark_west_pvalue(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    hac_lags: int = 12,
) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if benchmark is None:
        benchmark = np.zeros_like(y_true)
    else:
        benchmark = np.asarray(benchmark, dtype=float).reshape(-1)

    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(benchmark)

    if ok.sum() < 5:
        return np.nan

    yt = y_true[ok]
    yf = y_pred[ok]
    y0 = benchmark[ok]

    f = (yt - y0) ** 2 - (yt - yf) ** 2 + (y0 - yf) ** 2
    se = newey_west_se_of_mean(f, hac_lags)

    if not np.isfinite(se) or se <= 0:
        return np.nan

    return float(1.0 - norm.cdf(np.mean(f) / se))


def clark_west_pvalue_vector(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    hac_lags: int = 12,
) -> np.ndarray:
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)

    if benchmark is None:
        benchmark = np.zeros_like(Y_true)
    else:
        benchmark = np.asarray(benchmark, dtype=float)

    return np.array(
        [
            clark_west_pvalue(Y_true[:, j], Y_pred[:, j], benchmark[:, j], hac_lags)
            for j in range(Y_true.shape[1])
        ]
    )


def summarize_oos_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    hac_lags: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)

    if benchmark is None:
        benchmark = np.zeros_like(Y_true)
    else:
        benchmark = np.asarray(benchmark, dtype=float)

    mse = mean_squared_error_by_target(Y_true, Y_pred)
    r2 = out_of_sample_r2_vector(Y_true, Y_pred, benchmark)
    pval = clark_west_pvalue_vector(Y_true, Y_pred, benchmark, hac_lags)

    return mse, r2, pval


def build_save_dict(
    cfg: Dict[str, Any],
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    Y_true: np.ndarray,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    save_dict = {
        "Y_True": np.asarray(Y_true, dtype=float),
        "Dates": np.array(pd.DatetimeIndex(X_df.index).strftime("%Y-%m-%d"), dtype=object),
        "X_Columns": np.array(X_df.columns.astype(str), dtype=object),
        "Y_Columns": np.array(Y_df.columns.astype(str), dtype=object),
        "DataPath": cfg.get("data_path", ""),
        "FeatureGroupsJSON": json.dumps(cfg.get("feature_groups", [])),
        "TargetGroup": cfg.get("target_group", ""),
        "RunTag": cfg.get("run_tag", ""),
        "ParamsJSON": json.dumps(cfg.get("params", {})),
        "Horizon": cfg.get("horizon", np.nan),
        "OOSStart": cfg.get("oos_start", ""),
        "HyperFreq": cfg.get("hyper_freq", np.nan),
        "NMC": cfg.get("nmc", np.nan),
        "NAVG": cfg.get("navg", np.nan),
    }

    if extra_metadata:
        save_dict.update(extra_metadata)

    return save_dict


def save_results_mat(file_path: str, save_dict: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(file_path) or ".")
    sio.savemat(file_path, save_dict)