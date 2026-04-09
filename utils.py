#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import tempfile
import multiprocessing
from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm


# ============================================================================
# Basic utilities
# ============================================================================

def get_cpu_count() -> int:
    """Return available CPU count with a safe fallback."""
    try:
        return max(1, multiprocessing.cpu_count())
    except Exception:
        return 1


def ensure_dir(path: str) -> str:
    """Create a directory if needed and return the same path."""
    os.makedirs(path, exist_ok=True)
    return path


def make_dump_dir(prefix: str = "trainingDumps_") -> str:
    """Create a temporary working directory."""
    return tempfile.mkdtemp(prefix=prefix)


def safe_rmtree(path: Optional[str]) -> None:
    """Remove a directory if it exists."""
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def flatten_to_1d(x) -> np.ndarray:
    """Convert MATLAB-loaded content to a flat 1D NumPy array."""
    return np.asarray(x).reshape(-1)


def matlab_names_to_list(x) -> List[str]:
    """Convert MATLAB name fields to a Python list of strings."""
    if x is None:
        return []

    arr = np.asarray(x, dtype=object)

    if arr.ndim == 0:
        return [str(arr.item())]

    out = []
    for item in arr.reshape(-1):
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        elif isinstance(item, np.ndarray):
            if item.size == 1:
                out.append(str(item.item()))
            else:
                out.append("".join(np.asarray(item).astype(str).reshape(-1)).strip())
        else:
            out.append(str(item))
    return out


def yyyymm_to_month_end(yyyymm: Sequence[int]) -> pd.DatetimeIndex:
    """Convert YYYYMM integers to month-end timestamps."""
    yyyymm = flatten_to_1d(yyyymm).astype(int)
    year = yyyymm // 100
    month = yyyymm % 100

    dt = pd.to_datetime(
        {"year": year, "month": month, "day": np.ones_like(year)},
        errors="raise",
    )
    return pd.DatetimeIndex(dt) + pd.offsets.MonthEnd(0)


# ============================================================================
# Data loading
# ============================================================================

def load_mat_group_frame(mat_path: str, root_name: str, group_name: str) -> pd.DataFrame:
    """
    Load one group from target_and_features.mat into a DataFrame.

    Parameters
    ----------
    mat_path : str
        Path to .mat file.
    root_name : str
        'X' or 'y'.
    group_name : str
        Group name stored under the root.

    Returns
    -------
    pd.DataFrame
    """
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if root_name not in mat:
        raise KeyError(f"Missing root '{root_name}' in {mat_path}.")

    root = mat[root_name]
    if not hasattr(root, group_name):
        raise KeyError(f"Missing group '{root_name}.{group_name}' in {mat_path}.")

    block = getattr(root, group_name)

    if not hasattr(block, "Time") or not hasattr(block, "data") or not hasattr(block, "names"):
        raise ValueError(f"Group '{root_name}.{group_name}' must contain Time, data, and names.")

    dates = yyyymm_to_month_end(block.Time)
    data = np.asarray(block.data, dtype=float)
    names = matlab_names_to_list(block.names)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if len(names) != data.shape[1]:
        raise ValueError(
            f"Column mismatch in '{root_name}.{group_name}': "
            f"{len(names)} names vs {data.shape[1]} columns."
        )

    df = pd.DataFrame(data, index=dates, columns=names)
    df.index.name = "Date"
    return df


def subset_target_columns(Y_df: pd.DataFrame, target_indices: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """
    Subset target columns by zero-based indices.

    Parameters
    ----------
    Y_df : pd.DataFrame
    target_indices : list[int] or None
        None keeps all target columns.

    Returns
    -------
    pd.DataFrame
    """
    if target_indices is None:
        return Y_df.copy()

    target_indices = list(target_indices)
    if len(target_indices) == 0:
        raise ValueError("target_indices must be None or a non-empty list.")

    n_cols = Y_df.shape[1]
    if min(target_indices) < 0 or max(target_indices) >= n_cols:
        raise IndexError(f"target_indices out of bounds for {n_cols} target columns.")

    return Y_df.iloc[:, target_indices].copy()


def align_feature_target_frames(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align feature and target frames on their common date index.
    """
    merged = X_df.join(Y_df, how="inner")
    if dropna:
        merged = merged.dropna(axis=0, how="any").copy()

    X_out = merged[X_df.columns].copy()
    Y_out = merged[Y_df.columns].copy()
    return X_out, Y_out


def load_dataset(
    mat_path: str,
    feature_groups: Sequence[str],
    target_group: str,
    target_indices: Optional[Sequence[int]] = None,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load aligned features and targets from the canonical MAT dataset.

    Parameters
    ----------
    mat_path : str
    feature_groups : list[str]
        Example: ['dy_pc1', 'dy_pc2'] or ['dy_pc'].
    target_group : str
        Example: 'dy'.
    target_indices : list[int] or None
        Zero-based target column indices.
    dropna : bool
        Drop rows with missing values after alignment.

    Returns
    -------
    X_df, Y_df : pd.DataFrame, pd.DataFrame
    """
    if not feature_groups:
        raise ValueError("feature_groups must be a non-empty list.")

    X_blocks = [load_mat_group_frame(mat_path, "X", g) for g in feature_groups]
    X_df = pd.concat(X_blocks, axis=1)

    Y_df = load_mat_group_frame(mat_path, "y", target_group)
    Y_df = subset_target_columns(Y_df, target_indices=target_indices)

    return align_feature_target_frames(X_df, Y_df, dropna=dropna)


# ============================================================================
# OOS indexing and validation splits
# ============================================================================

def locate_oos_start_index(dates: Sequence[pd.Timestamp], oos_start: str) -> int:
    """Return the first index i such that dates[i] >= oos_start."""
    dates = pd.DatetimeIndex(dates)
    oos_start_ts = pd.Timestamp(oos_start)

    idx = np.where(dates >= oos_start_ts)[0]
    if idx.size == 0:
        raise ValueError(
            f"No sample date on or after {oos_start_ts.date()}. "
            f"Available range: {dates.min().date()} to {dates.max().date()}."
        )
    return int(idx[0])


def enumerate_oos_forecast_indices(dates: Sequence[pd.Timestamp], oos_start: str) -> List[int]:
    """Return all OOS forecast indices from oos_start onward."""
    dates = pd.DatetimeIndex(dates)
    start_idx = locate_oos_start_index(dates, oos_start)
    return list(range(start_idx, len(dates)))


def compute_training_end_index(forecast_index: int, horizon: int) -> int:
    """
    Return the last index available for training when forecasting at forecast_index.

    If horizon=12, forecast at i can only use data through i-12.
    """
    return int(forecast_index - horizon)


def build_purged_validation_slices(
    n_observations: int,
    validation_fraction: float,
    purge_size: int,
) -> Tuple[slice, slice]:
    """
    Construct fit and validation slices with an embargo/purge gap.

    Returns
    -------
    fit_slice : slice
    validation_slice : slice
    """
    if n_observations < 4:
        raise ValueError("Not enough observations for purged validation.")

    validation_length = int(np.ceil(n_observations * float(validation_fraction)))
    validation_length = max(validation_length, 1)

    fit_end = n_observations - purge_size - validation_length
    validation_start = fit_end + purge_size

    if fit_end < 2 or validation_start >= n_observations:
        raise ValueError(
            f"Invalid purged split: n={n_observations}, "
            f"validation_fraction={validation_fraction}, purge_size={purge_size}."
        )

    return slice(0, fit_end), slice(validation_start, n_observations)


# ============================================================================
# Standardization
# ============================================================================

def fit_feature_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit mean/std on training data only.

    Supports
    --------
    X.ndim == 2 : (N, p)
    X.ndim == 3 : (N, L, p)

    Returns
    -------
    mean_ : ndarray
    std_  : ndarray
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 2:
        flat = X
    elif X.ndim == 3:
        flat = X.reshape(-1, X.shape[-1])
    else:
        raise ValueError("X must be 2D or 3D.")

    mean_ = np.nanmean(flat, axis=0)
    std_ = np.nanstd(flat, axis=0, ddof=0)
    std_[std_ < 1e-12] = 1.0
    return mean_, std_


def apply_feature_standardizer(X: np.ndarray, mean_: np.ndarray, std_: np.ndarray) -> np.ndarray:
    """
    Apply a previously fitted standardizer to 2D or 3D feature arrays.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 2:
        return (X - mean_[None, :]) / std_[None, :]
    if X.ndim == 3:
        return (X - mean_[None, None, :]) / std_[None, None, :]

    raise ValueError("X must be 2D or 3D.")


def prepare_validation_matrices(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    validation_fraction: float,
    purge_size: int,
    standardize_features: bool = False,
):
    """
    Prepare fit/validation matrices for hyperparameter evaluation.

    If standardize_features is True:
    - fit the scaler on fit data only
    - apply it to fit and validation data

    Returns
    -------
    X_fit, Y_fit, X_val, Y_val, scaler_info
    """
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)

    fit_slice, val_slice = build_purged_validation_slices(
        n_observations=X_train.shape[0],
        validation_fraction=validation_fraction,
        purge_size=purge_size,
    )

    X_fit = X_train[fit_slice]
    Y_fit = Y_train[fit_slice]
    X_val = X_train[val_slice]
    Y_val = Y_train[val_slice]

    scaler_info = None
    if standardize_features:
        mean_, std_ = fit_feature_standardizer(X_fit)
        X_fit = apply_feature_standardizer(X_fit, mean_, std_)
        X_val = apply_feature_standardizer(X_val, mean_, std_)
        scaler_info = {"mean": mean_, "std": std_}

    return X_fit, Y_fit, X_val, Y_val, scaler_info


def prepare_final_training_matrices(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    standardize_features: bool = False,
):
    """
    Prepare final train/test matrices after hyperparameter selection.

    If standardize_features is True:
    - fit the scaler on the full training sample
    - apply it to the full training sample and the test point

    Returns
    -------
    X_train_final, Y_train_final, X_test_final, scaler_info
    """
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    scaler_info = None
    if standardize_features:
        mean_, std_ = fit_feature_standardizer(X_train)
        X_train = apply_feature_standardizer(X_train, mean_, std_)
        X_test = apply_feature_standardizer(X_test, mean_, std_)
        scaler_info = {"mean": mean_, "std": std_}

    return X_train, Y_train, X_test, scaler_info


# ============================================================================
# Forecast evaluation
# ============================================================================

def mean_squared_error_by_target(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Return per-target MSE."""
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)
    return np.nanmean((Y_true - Y_pred) ** 2, axis=0)


def out_of_sample_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
) -> float:
    """
    Compute out-of-sample R^2 for a single target.

    If benchmark is None, the zero benchmark is used.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if benchmark is None:
        benchmark = np.zeros_like(y_true, dtype=float)
    else:
        benchmark = np.asarray(benchmark, dtype=float).reshape(-1)

    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(benchmark)
    if valid.sum() == 0:
        return np.nan

    sse_model = np.sum((y_true[valid] - y_pred[valid]) ** 2)
    sse_bench = np.sum((y_true[valid] - benchmark[valid]) ** 2)

    if sse_bench <= 0:
        return np.nan

    return 1.0 - sse_model / sse_bench


def out_of_sample_r2_vector(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute OOS R^2 target by target."""
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)

    if benchmark is None:
        benchmark = np.zeros_like(Y_true, dtype=float)
    else:
        benchmark = np.asarray(benchmark, dtype=float)

    return np.array(
        [out_of_sample_r2(Y_true[:, j], Y_pred[:, j], benchmark[:, j]) for j in range(Y_true.shape[1])]
    )


def _newey_west_standard_error_of_mean(loss_diff: np.ndarray, hac_lags: int) -> float:
    """
    Newey-West standard error for the sample mean of a time series.
    """
    x = np.asarray(loss_diff, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = x.size

    if n < 2:
        return np.nan

    u = x - np.mean(x)
    gamma0 = np.dot(u, u) / n
    lrv = gamma0

    max_lag = min(int(hac_lags), n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = np.dot(u[lag:], u[:-lag]) / n
        lrv += 2.0 * weight * gamma

    if lrv < 0:
        return np.nan

    return np.sqrt(lrv / n)


def clark_west_pvalue(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    hac_lags: int = 12,
) -> float:
    """
    Compute the Clark-West one-sided p-value for one target.

    The null is that the candidate model does not improve on the benchmark.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if benchmark is None:
        benchmark = np.zeros_like(y_true, dtype=float)
    else:
        benchmark = np.asarray(benchmark, dtype=float).reshape(-1)

    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(benchmark)
    if valid.sum() < 5:
        return np.nan

    yt = y_true[valid]
    yf = y_pred[valid]
    y0 = benchmark[valid]

    f_t = (yt - y0) ** 2 - (yt - yf) ** 2 + (y0 - yf) ** 2
    f_bar = np.mean(f_t)
    se_bar = _newey_west_standard_error_of_mean(f_t, hac_lags=hac_lags)

    if not np.isfinite(se_bar) or se_bar <= 0:
        return np.nan

    t_stat = f_bar / se_bar
    return float(1.0 - norm.cdf(t_stat))


def clark_west_pvalue_vector(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    hac_lags: int = 12,
) -> np.ndarray:
    """Compute Clark-West p-values target by target."""
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)

    if benchmark is None:
        benchmark = np.zeros_like(Y_true, dtype=float)
    else:
        benchmark = np.asarray(benchmark, dtype=float)

    return np.array(
        [clark_west_pvalue(Y_true[:, j], Y_pred[:, j], benchmark[:, j], hac_lags=hac_lags)
         for j in range(Y_true.shape[1])]
    )


def summarize_oos_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    hac_lags: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-target MSE, OOS R^2, and Clark-West p-values.
    """
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)

    if benchmark is None:
        benchmark = np.zeros_like(Y_true, dtype=float)
    else:
        benchmark = np.asarray(benchmark, dtype=float)

    mse_vec = mean_squared_error_by_target(Y_true, Y_pred)
    r2_vec = out_of_sample_r2_vector(Y_true, Y_pred, benchmark=benchmark)
    pval_vec = clark_west_pvalue_vector(Y_true, Y_pred, benchmark=benchmark, hac_lags=hac_lags)

    return mse_vec, r2_vec, pval_vec


# ============================================================================
# Result saving and summary extraction
# ============================================================================

def result_name(cfg: Dict[str, Any]) -> str:
    """
    Build a deterministic experiment filename.

    This function assumes there is no outer sweep. It only encodes the
    current experiment configuration.
    """
    target_group = str(cfg.get("target_group", "target"))
    run_tag = str(cfg.get("run_tag", "run"))
    model_name = str(cfg.get("model_name", "Model")).replace("Model", "")
    horizon = f"h{int(cfg.get('horizon', 0))}"

    parts = [target_group, run_tag, model_name, horizon]

    params = cfg.get("params", {})

    if "archi" in params:
        arch = np.asarray(params["archi"]).ravel()
        arch_tag = "x".join(str(int(v)) for v in arch) if arch.size > 0 else "none"
        parts.append(f"a{arch_tag}")

    if "lstm_units" in params:
        arch = np.asarray(params["lstm_units"]).ravel()
        arch_tag = "x".join(str(int(v)) for v in arch) if arch.size > 0 else "none"
        parts.append(f"lstm{arch_tag}")

    if "dense_archi" in params:
        arch = np.asarray(params["dense_archi"]).ravel()
        arch_tag = "x".join(str(int(v)) for v in arch) if arch.size > 0 else "none"
        parts.append(f"dense{arch_tag}")

    if "seq_len" in params:
        parts.append(f"seq{int(params['seq_len'])}")

    if "Dropout" in params:
        do_arr = np.asarray(params["Dropout"], dtype=float).ravel()
        do_tag = "|".join(f"{v:g}" for v in do_arr)
        parts.append(f"do{do_tag}")

    if "l1l2" in params:
        reg_arr = np.asarray(params["l1l2"], dtype=float).ravel()
        if reg_arr.size == 1:
            reg_tag = f"{reg_arr[0]:g}-{reg_arr[0]:g}"
        elif reg_arr.size >= 2:
            reg_tag = f"{reg_arr[0]:g}-{reg_arr[1]:g}"
        else:
            reg_tag = "0-0"
        parts.append(f"reg{reg_tag}")

    if "alpha" in params:
        parts.append(f"alpha{float(params['alpha']):g}")

    if "learning_rate" in params:
        parts.append(f"lr{float(params['learning_rate']):g}")

    if "model_name" in params:
        parts.append(str(params["model_name"]))

    return "__".join(parts)


def build_save_dict(
    cfg: Dict[str, Any],
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    Y_true: np.ndarray,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a standard metadata dictionary for MAT saving.
    """
    save_dict = {
        "MatPath": cfg.get("mat_path", ""),
        "FeatureGroupsJSON": json.dumps(cfg.get("feature_groups", [])),
        "TargetGroup": cfg.get("target_group", ""),
        "RunTag": cfg.get("run_tag", ""),
        "ModelName": cfg.get("model_name", ""),
        "ParamsJSON": json.dumps(cfg.get("params", {})),
        "RunConfigJSON": json.dumps(
            {
                "horizon": cfg.get("horizon", np.nan),
                "oos_start": cfg.get("oos_start", ""),
                "hyper_freq": cfg.get("hyper_freq", np.nan),
                "nmc": cfg.get("nmc", np.nan),
                "navg": cfg.get("navg", np.nan),
                "run_tag": cfg.get("run_tag", ""),
            }
        ),
        "Horizon": cfg.get("horizon", np.nan),
        "Y_True": np.asarray(Y_true, dtype=float),
        "Dates": np.array(pd.DatetimeIndex(X_df.index).strftime("%Y-%m-%d"), dtype=object),
        "X_Columns": np.array(X_df.columns.astype(str), dtype=object),
        "Y_Columns": np.array(Y_df.columns.astype(str), dtype=object),
    }

    if extra_metadata:
        save_dict.update(extra_metadata)

    return save_dict


def save_results_mat(file_path: str, save_dict: Dict[str, Any]) -> None:
    """Save a result dictionary to a MAT file."""
    ensure_dir(os.path.dirname(file_path) or ".")
    sio.savemat(file_path, save_dict)


def extract_summary_rows(
    save_dict: Dict[str, Any],
    cfg: Dict[str, Any],
    mat_file: str,
    y_columns: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert saved experiment output into row-wise summary records.

    Returns
    -------
    summary_rows, r2_rows, pval_rows, mse_rows
    """
    model_name = str(cfg.get("model_name", save_dict.get("ModelName", "")))

    r2_key = f"R2OOS_{model_name}"
    pval_key = f"R2OOS_pval_{model_name}"
    mse_key = f"MSE_{model_name}"

    if r2_key not in save_dict or pval_key not in save_dict or mse_key not in save_dict:
        raise KeyError(f"Missing metric keys for model '{model_name}' in save_dict.")

    r2_vec = np.asarray(save_dict[r2_key], dtype=float).reshape(-1)
    pval_vec = np.asarray(save_dict[pval_key], dtype=float).reshape(-1)
    mse_vec = np.asarray(save_dict[mse_key], dtype=float).reshape(-1)

    if len(y_columns) != len(r2_vec):
        raise ValueError("Length mismatch between y_columns and saved metrics.")

    params_json = json.dumps(cfg.get("params", {}), ensure_ascii=False)

    summary_rows = []
    r2_rows = []
    pval_rows = []
    mse_rows = []

    for j, target_name in enumerate(y_columns):
        base_row = {
            "mat_file": mat_file,
            "model_name": model_name,
            "target_name": str(target_name),
            "target_group": cfg.get("target_group", ""),
            "feature_groups": json.dumps(cfg.get("feature_groups", []), ensure_ascii=False),
            "run_tag": cfg.get("run_tag", ""),
            "params_json": params_json,
            "horizon": cfg.get("horizon", np.nan),
            "oos_start": cfg.get("oos_start", ""),
        }

        summary_rows.append(
            {
                **base_row,
                "R2OOS": float(r2_vec[j]) if np.isfinite(r2_vec[j]) else np.nan,
                "p_value": float(pval_vec[j]) if np.isfinite(pval_vec[j]) else np.nan,
                "MSE": float(mse_vec[j]) if np.isfinite(mse_vec[j]) else np.nan,
            }
        )

        r2_rows.append({**base_row, "value": float(r2_vec[j]) if np.isfinite(r2_vec[j]) else np.nan})
        pval_rows.append({**base_row, "value": float(pval_vec[j]) if np.isfinite(pval_vec[j]) else np.nan})
        mse_rows.append({**base_row, "value": float(mse_vec[j]) if np.isfinite(mse_vec[j]) else np.nan})

    return summary_rows, r2_rows, pval_rows, mse_rows


def save_experiment_summary_to_excel(
    summary_rows: Sequence[Dict[str, Any]],
    r2_rows: Sequence[Dict[str, Any]],
    pval_rows: Sequence[Dict[str, Any]],
    mse_rows: Sequence[Dict[str, Any]],
    out_xlsx: str,
) -> None:
    """
    Save experiment summaries into one Excel workbook.
    """
    ensure_dir(os.path.dirname(out_xlsx) or ".")

    summary_df = pd.DataFrame(summary_rows)
    r2_df = pd.DataFrame(r2_rows)
    pval_df = pd.DataFrame(pval_rows)
    mse_df = pd.DataFrame(mse_rows)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        r2_df.to_excel(writer, sheet_name="r2_by_target", index=False)
        pval_df.to_excel(writer, sheet_name="pval_by_target", index=False)
        mse_df.to_excel(writer, sheet_name="mse_by_target", index=False)