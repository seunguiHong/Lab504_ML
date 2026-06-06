#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import multiprocessing
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm

import NNBib as NNB


def run(C):
    dumploc = make_dump_dir()

    try:
        X_df, Y_df = load_dataset(
            data_path=C.data_path,
            feature_groups=C.feature_groups,
            target_group=C.target_group,
            target_indices=C.target_indices,
        )

        X = X_df.to_numpy(dtype=float)
        Y = Y_df.to_numpy(dtype=float)
        dates = pd.DatetimeIndex(X_df.index)

        ncpus = resolve_ncpus(C)

        print_experiment_header(C, X, Y, ncpus)

        if C.model == "MacroNN":
            group_sizes = compute_feature_group_sizes(
                C.data_path, C.feature_groups,
            )
            C.params = dict(C.params)
            C.params["group_sizes"] = group_sizes

        result = run_oos_forecast(
            X=X,
            Y=Y,
            dates=dates,
            C=C,
            dumploc=dumploc,
            ncpus=ncpus,
        )

        save_dict = build_save_dict(C, X_df, Y_df, Y)
        save_dict.update(result)

        save_results_mat(C.out_file, save_dict)

        print_experiment_footer(C, save_dict)

        return save_dict, C.out_file

    finally:
        safe_rmtree(dumploc)


def print_experiment_header(C, X, Y, ncpus):
    print("\n" + "=" * 72)
    print("Experiment settings")
    print("=" * 72)
    print(f"Data path       : {C.data_path}")
    print(f"Feature groups  : {C.feature_groups}")
    print(f"Target group    : {C.target_group}")
    print(f"Target indices  : {C.target_indices}")
    print(f"Model           : {C.model}")
    print(f"Run tag         : {C.run_tag}")
    print(f"Output file     : {C.out_file}")
    print(f"X shape         : {X.shape}")
    print(f"Y shape         : {Y.shape}")
    print(f"OOS start       : {C.oos_start}")
    print(f"Horizon         : {C.horizon}")
    print(f"Hyper frequency : {C.hyper_freq}")
    print(f"NMC / NAVG      : {C.nmc} / {C.navg}")
    print(f"CPU workers     : {ncpus}")
    print("-" * 72)
    print("Model parameters")
    print(json.dumps(C.params, indent=2))
    print("=" * 72 + "\n")


def print_experiment_footer(C, save_dict):
    print("\n" + "=" * 72)
    print("Finished")
    print("=" * 72)
    print(f"Saved to : {C.out_file}")
    print(f"R2OOS    : {np.round(save_dict['R2OOS'], 4)}")
    print(f"p-values : {np.round(save_dict['R2OOS_pval'], 4)}")
    print("=" * 72)


def run_oos_forecast(X, Y, dates, C, dumploc, ncpus):
    dates = pd.DatetimeIndex(dates)
    oos_indices = enumerate_oos_forecast_indices(dates, C.oos_start)

    T, M = Y.shape
    nmc = int(C.nmc)
    navg = int(C.navg)
    horizon = int(C.horizon)
    hyper_freq = int(C.hyper_freq)
    log_freq = int(getattr(C, "log_freq", 12))

    if navg > nmc:
        raise ValueError("navg cannot exceed nmc.")

    Y_forecast = np.full((T, M), np.nan)
    Y_forecast_all = np.full((T, nmc, M), np.nan)
    val_loss = np.full((T, nmc), np.nan)

    active_params = None
    active_dir = os.path.join(dumploc, "active")
    os.makedirs(active_dir, exist_ok=True)

    oos_count = 0
    total_oos = len(oos_indices)

    prev_val_score = np.nan
    best_val_since_refit = np.inf

    print(f"Total OOS steps: {total_oos}")

    for step, forecast_idx in enumerate(oos_indices, start=1):
        X_model, Y_model = build_nn_input(
            X=X,
            Y=Y,
            forecast_idx=forecast_idx,
            horizon=horizon,
        )

        if X_model is None:
            continue

        oos_count += 1

        refit = compute_refit_flag(
            oos_count=oos_count,
            hyper_freq=hyper_freq,
            prev_val_score=prev_val_score,
            best_val_since_refit=best_val_since_refit,
        )

        if refit or active_params is None:
            outputs, active_params, seed_val_loss = refit_select_candidate(
                X_model=X_model,
                Y_model=Y_model,
                C=C,
                dumploc=dumploc,
                active_dir=active_dir,
                ncpus=ncpus,
            )

        else:
            outputs = run_seed_ensemble(
                model_name=C.model,
                ncpus=ncpus,
                nmc=nmc,
                X_model=X_model,
                Y_model=Y_model,
                params=active_params,
                dumploc=active_dir,
                refit=False,
            )

            seed_val_loss = extract_seed_validation_losses(outputs, nmc)

        seed_forecasts = extract_seed_forecasts(outputs, nmc)

        Y_forecast_all[forecast_idx] = seed_forecasts

        Y_forecast[forecast_idx] = top_validation_seed_mean(
            seed_forecasts=seed_forecasts,
            seed_val_loss=seed_val_loss,
            navg=navg,
        )

        val_loss[forecast_idx] = seed_val_loss

        current_val_score = ensemble_validation_score(
            seed_val_loss=seed_val_loss,
            navg=navg,
        )

        if refit:
            best_val_since_refit = current_val_score
        else:
            best_val_since_refit = min(best_val_since_refit, current_val_score)

        prev_val_score = current_val_score

        if step == 1 or step % log_freq == 0 or step == total_oos:
            r2_now = summarize_oos_metrics(
                Y_true=Y,
                Y_pred=Y_forecast,
                hac_lags=horizon,
            )[1]

            print(
                f"[{step:4d}/{total_oos}] "
                f"{dates[forecast_idx].strftime('%Y-%m-%d')} | "
                f"oos={oos_count:4d} | "
                f"refit={str(refit):5s} | "
                f"val={current_val_score:10.6f} | "
                f"best={best_val_since_refit:10.6f} | "
                f"R2={np.round(r2_now, 4)}"
            )

    mse, r2, pval = summarize_oos_metrics(
        Y_true=Y,
        Y_pred=Y_forecast,
        hac_lags=horizon,
    )

    return {
        "Y_Forecast": Y_forecast,
        "Y_Forecast_All": Y_forecast_all,
        "ValLoss": val_loss,
        "MSE": mse,
        "R2OOS": r2,
        "R2OOS_pval": pval,
    }


def compute_refit_flag(oos_count, hyper_freq, prev_val_score, best_val_since_refit):
    if int(oos_count) == 1:
        return True

    if int(oos_count) % int(hyper_freq) != 0:
        return False

    if not np.isfinite(prev_val_score):
        return True

    if not np.isfinite(best_val_since_refit):
        return True

    return bool(prev_val_score > best_val_since_refit)


def refit_select_candidate(X_model, Y_model, C, dumploc, active_dir, ncpus):
    candidates = build_dropout_l1l2_candidates(C.params)

    round_dir = tempfile.mkdtemp(prefix="candidateRound_", dir=dumploc)

    best_outputs = None
    best_params = None
    best_score = np.inf
    best_candidate_dir = None
    best_seed_val_loss = None

    try:
        for k, params in enumerate(candidates):
            candidate_dir = os.path.join(round_dir, f"candidate_{k:03d}")
            os.makedirs(candidate_dir, exist_ok=True)

            outputs = run_seed_ensemble(
                model_name=C.model,
                ncpus=ncpus,
                nmc=C.nmc,
                X_model=X_model,
                Y_model=Y_model,
                params=params,
                dumploc=candidate_dir,
                refit=True,
            )

            seed_val_loss = extract_seed_validation_losses(outputs, C.nmc)

            score = ensemble_validation_score(
                seed_val_loss=seed_val_loss,
                navg=C.navg,
            )

            if score < best_score:
                best_outputs = outputs
                best_params = copy.deepcopy(params)
                best_score = score
                best_candidate_dir = candidate_dir
                best_seed_val_loss = seed_val_loss

        replace_active_dir(
            source_dir=best_candidate_dir,
            active_dir=active_dir,
        )

        return best_outputs, best_params, best_seed_val_loss

    finally:
        safe_rmtree(round_dir)


def run_seed_ensemble(model_name, ncpus, nmc, X_model, Y_model, params, dumploc, refit):
    nmc = int(nmc)
    n_workers = min(int(ncpus), nmc)

    jobs = [
        (model_name, seed, X_model, Y_model, params, bool(refit), dumploc)
        for seed in range(nmc)
    ]

    if n_workers <= 1:
        results = [run_one_seed_job(job) for job in jobs]

    else:
        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(run_one_seed_job, jobs)

    return {seed: results[seed] for seed in range(nmc)}


def run_one_seed_job(args):
    model_name, seed, X_model, Y_model, params, refit, dumploc = args

    if model_name == "NN":
        model_func = NNB.NN

    elif model_name == "pcNN":
        model_func = NNB.pcNN

    elif model_name == "MacroNN":
        model_func = NNB.MacroNN

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model_func(
        X=X_model,
        Y=Y_model,
        no=seed,
        params=params,
        refit=refit,
        dumploc=dumploc,
    )


def build_nn_input(X, Y, forecast_idx, horizon):
    train_end = int(forecast_idx) - int(horizon)

    if train_end < 1:
        return None, None

    X_train = X[: train_end + 1]
    Y_train = Y[: train_end + 1]
    X_test = X[forecast_idx : forecast_idx + 1]
    Y_test = Y[forecast_idx : forecast_idx + 1]

    if X_test.shape[0] != 1:
        return None, None

    if not np.all(np.isfinite(X_test)):
        return None, None

    ok = np.all(np.isfinite(X_train), axis=1) & np.all(np.isfinite(Y_train), axis=1)

    X_train = X_train[ok]
    Y_train = Y_train[ok]

    if X_train.shape[0] < 30:
        return None, None

    X_model = np.vstack([X_train, X_test])
    Y_model = np.vstack([Y_train, Y_test])

    return X_model, Y_model


def build_dropout_l1l2_candidates(params):
    dropout_raw = params.get("Dropout", 0.0)
    l1l2_raw = params.get("l1l2", [0.0, 0.0])

    if np.isscalar(dropout_raw):
        dropout_candidates = [float(dropout_raw)]
    else:
        dropout_candidates = [
            float(v) for v in np.asarray(dropout_raw, dtype=float).ravel()
        ]

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


def normalize_l1l2(value):
    arr = np.asarray(value, dtype=float).ravel()

    if arr.size == 0:
        return [0.0, 0.0]

    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]

    return [float(arr[0]), float(arr[1])]


def extract_seed_forecasts(outputs, nmc):
    forecasts = []

    for seed in range(int(nmc)):
        yhat = np.asarray(outputs[seed][0], dtype=float)

        if yhat.ndim == 1:
            yhat = yhat.reshape(1, -1)

        forecasts.append(yhat)

    return np.concatenate(forecasts, axis=0)


def extract_seed_validation_losses(outputs, nmc):
    return np.array([outputs[seed][1] for seed in range(int(nmc))], dtype=float)


def top_validation_seed_mean(seed_forecasts, seed_val_loss, navg):
    seed_forecasts = np.asarray(seed_forecasts, dtype=float)
    seed_val_loss = np.asarray(seed_val_loss, dtype=float).reshape(-1)

    ok = np.isfinite(seed_val_loss)

    if not np.any(ok):
        return np.full(seed_forecasts.shape[1], np.nan)

    valid_idx = np.where(ok)[0]
    order = valid_idx[np.argsort(seed_val_loss[valid_idx])]
    keep = order[: int(navg)]

    return np.mean(seed_forecasts[keep], axis=0)


def ensemble_validation_score(seed_val_loss, navg):
    seed_val_loss = np.asarray(seed_val_loss, dtype=float).reshape(-1)
    vals = seed_val_loss[np.isfinite(seed_val_loss)]

    if vals.size == 0:
        return np.nan

    vals = np.sort(vals)
    k = min(int(navg), vals.size)

    return float(np.mean(vals[:k]))


def replace_active_dir(source_dir, active_dir):
    if source_dir is None or not os.path.exists(source_dir):
        raise ValueError("source_dir for active model replacement is invalid.")

    if os.path.exists(active_dir):
        shutil.rmtree(active_dir)

    shutil.copytree(source_dir, active_dir)


def resolve_ncpus(C):
    requested = getattr(C, "ncpus", None)

    if requested is None:
        return max(1, multiprocessing.cpu_count())

    return max(1, min(int(requested), multiprocessing.cpu_count()))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def make_dump_dir(prefix="trainingDumps_"):
    return tempfile.mkdtemp(prefix=prefix)


def safe_rmtree(path):
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def matlab_names_to_list(x):
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


def yyyymm_to_month_end(yyyymm):
    yyyymm = np.asarray(yyyymm).reshape(-1).astype(int)

    year = yyyymm // 100
    month = yyyymm % 100

    dates = pd.to_datetime(
        {"year": year, "month": month, "day": np.ones_like(year)},
        errors="raise",
    )

    return pd.DatetimeIndex(dates) + pd.offsets.MonthEnd(0)


def load_mat_group_frame(data_path, root_name, group_name):
    mat = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)

    if root_name not in mat:
        raise KeyError(f"Missing root {root_name} in {data_path}.")

    root = mat[root_name]

    if not hasattr(root, group_name):
        raise KeyError(f"Missing group {root_name}.{group_name} in {data_path}.")

    block = getattr(root, group_name)

    data = np.asarray(block.data, dtype=float)
    dates = yyyymm_to_month_end(block.Time)
    names = matlab_names_to_list(block.names)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if len(names) != data.shape[1]:
        raise ValueError(f"Column-name mismatch in {root_name}.{group_name}.")

    df = pd.DataFrame(data, index=dates, columns=names)
    df.index.name = "Date"

    return df


def subset_target_columns(Y_df, target_indices=None):
    if target_indices is None:
        return Y_df.copy()

    return Y_df.iloc[:, list(target_indices)].copy()


def align_feature_target_frames(X_df, Y_df, dropna=True):
    merged = X_df.join(Y_df, how="inner")

    if dropna:
        merged = merged.dropna(axis=0, how="any")

    return merged[X_df.columns].copy(), merged[Y_df.columns].copy()


def load_dataset(data_path, feature_groups, target_group, target_indices=None, dropna=True):
    X_df = pd.concat(
        [load_mat_group_frame(data_path, "X", group) for group in feature_groups],
        axis=1,
    )

    Y_df = load_mat_group_frame(data_path, "y", target_group)
    Y_df = subset_target_columns(Y_df, target_indices)

    return align_feature_target_frames(X_df, Y_df, dropna=dropna)


def compute_feature_group_sizes(data_path, feature_groups):
    """Return the number of columns in each feature group."""
    return [
        load_mat_group_frame(data_path, "X", group).shape[1]
        for group in feature_groups
    ]


def locate_oos_start_index(dates, oos_start):
    dates = pd.DatetimeIndex(dates)
    idx = np.where(dates >= pd.Timestamp(oos_start))[0]

    if idx.size == 0:
        raise ValueError("No available sample date on or after oos_start.")

    return int(idx[0])


def enumerate_oos_forecast_indices(dates, oos_start):
    dates = pd.DatetimeIndex(dates)
    start_idx = locate_oos_start_index(dates, oos_start)

    return list(range(start_idx, len(dates)))


def mean_squared_error_by_target(Y_true, Y_pred):
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)

    return np.nanmean((Y_true - Y_pred) ** 2, axis=0)


def out_of_sample_r2(y_true, y_pred, benchmark=None):
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


def out_of_sample_r2_vector(Y_true, Y_pred, benchmark=None):
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


def newey_west_se_of_mean(x, hac_lags):
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


def clark_west_pvalue(y_true, y_pred, benchmark=None, hac_lags=12):
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


def clark_west_pvalue_vector(Y_true, Y_pred, benchmark=None, hac_lags=12):
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


def summarize_oos_metrics(Y_true, Y_pred, benchmark=None, hac_lags=12):
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


def build_save_dict(C, X_df, Y_df, Y_true):
    return {
        "Y_True": np.asarray(Y_true, dtype=float),
        "Dates": np.array(pd.DatetimeIndex(X_df.index).strftime("%Y-%m-%d"), dtype=object),
        "X_Columns": np.array(X_df.columns.astype(str), dtype=object),
        "Y_Columns": np.array(Y_df.columns.astype(str), dtype=object),
        "DataPath": C.data_path,
        "FeatureGroupsJSON": json.dumps(C.feature_groups),
        "TargetGroup": C.target_group,
        "TargetIndicesJSON": json.dumps(C.target_indices),
        "Model": C.model,
        "RunTag": C.run_tag,
        "ParamsJSON": json.dumps(C.params),
        "Horizon": C.horizon,
        "OOSStart": C.oos_start,
        "HyperFreq": C.hyper_freq,
        "NMC": C.nmc,
        "NAVG": C.navg,
    }


def save_results_mat(file_path, save_dict):
    ensure_dir(os.path.dirname(file_path) or ".")
    sio.savemat(file_path, save_dict)