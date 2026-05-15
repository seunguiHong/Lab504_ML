#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm

import config as C
import NNBib as NNB


_BASE_SEED = 1234


def main():
    X_df, Y_df = load_dataset(
        data_path=C.data_path,
        feature_groups=C.feature_groups,
        target_group=C.target_group,
        target_indices=C.target_indices,
    )

    result = run_oos(X_df, Y_df)

    save_result(
        out_file=C.out_file,
        result=result,
        X_df=X_df,
        Y_df=Y_df,
    )

    print("Saved to", C.out_file)
    print("R2OOS:", np.round(result["R2OOS"], 4))


def run_oos(X_df, Y_df):
    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(X_df.index)

    oos_idx = oos_indices(dates, C.oos_start)

    T, M = Y.shape
    nmc = int(C.nmc)
    navg = int(C.navg)

    if navg > nmc:
        raise ValueError("navg cannot exceed nmc.")

    Y_forecast = np.full((T, M), np.nan)
    Y_forecast_all = np.full((T, nmc, M), np.nan)
    Y_benchmark = np.full((T, M), np.nan)
    val_loss = np.full((T, nmc), np.nan)

    candidates = NNB.build_candidates(C)

    current_candidate = None
    oos_count = 0

    print(f"Model: {C.model}")
    print(f"Benchmark: {C.benchmark}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"OOS steps: {len(oos_idx)}")

    for step, forecast_idx in enumerate(oos_idx, start=1):
        sample = make_forecast_sample(
            X=X,
            Y=Y,
            forecast_idx=forecast_idx,
            purge_gap=C.purge_gap,
        )

        if sample is None:
            continue

        X_hist_raw, Y_hist, X_test_raw = sample

        split = purged_validation_split(
            X=X_hist_raw,
            Y=Y_hist,
            validation_frac=C.validation_frac,
            purge_gap=C.purge_gap,
        )

        if split is None:
            continue

        X_fit_raw, Y_fit, X_val_raw, Y_val = split

        if not finite_design(X_fit_raw, Y_fit, X_val_raw, Y_val, X_test_raw):
            continue

        oos_count += 1
        retune = (current_candidate is None) or ((oos_count - 1) % int(C.hyper_freq) == 0)

        if retune:
            current_candidate, loss_vec, epoch_vec = select_candidate(
                candidates=candidates,
                X_fit_raw=X_fit_raw,
                Y_fit=Y_fit,
                X_val_raw=X_val_raw,
                Y_val=Y_val,
                X_test_raw=X_test_raw,
            )
        else:
            loss_vec, epoch_vec = evaluate_candidate(
                candidate=current_candidate,
                X_fit_raw=X_fit_raw,
                Y_fit=Y_fit,
                X_val_raw=X_val_raw,
                Y_val=Y_val,
                X_test_raw=X_test_raw,
            )

        seed_forecasts = final_seed_forecasts(
            candidate=current_candidate,
            X_hist_raw=X_hist_raw,
            Y_hist=Y_hist,
            X_test_raw=X_test_raw,
            epoch_vec=epoch_vec,
        )

        Y_forecast_all[forecast_idx, :, :] = seed_forecasts
        Y_forecast[forecast_idx, :] = top_navg_mean(seed_forecasts, loss_vec, navg)
        Y_benchmark[forecast_idx, :] = benchmark_forecast(
            Y=Y,
            forecast_idx=forecast_idx,
            purge_gap=C.purge_gap,
            benchmark=C.benchmark,
        )
        val_loss[forecast_idx, :] = loss_vec

        if step == 1 or step % 12 == 0 or step == len(oos_idx):
            r2_now = summarize_oos_metrics(
                Y_true=Y,
                Y_pred=Y_forecast,
                Y_benchmark=Y_benchmark,
                hac_lags=int(C.purge_gap),
            )[1]

            print(
                f"[{step:4d}/{len(oos_idx)}] "
                f"date={dates[forecast_idx].strftime('%Y-%m-%d')} | "
                f"retune={retune} | "
                f"val={float(np.nanmean(loss_vec)):10.6f}"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    mse, r2, pval = summarize_oos_metrics(
        Y_true=Y,
        Y_pred=Y_forecast,
        Y_benchmark=Y_benchmark,
        hac_lags=int(C.purge_gap),
    )

    return {
        "Y_Forecast": Y_forecast,
        "Y_Forecast_All": Y_forecast_all,
        "Y_Benchmark": Y_benchmark,
        "ValLoss": val_loss,
        "MSE": mse,
        "R2OOS": r2,
        "R2OOS_pval": pval,
    }


def select_candidate(candidates, X_fit_raw, Y_fit, X_val_raw, Y_val, X_test_raw):
    best_candidate = None
    best_loss_vec = None
    best_epoch_vec = None
    best_score = np.inf

    for candidate in candidates:
        loss_vec, epoch_vec = evaluate_candidate(
            candidate=candidate,
            X_fit_raw=X_fit_raw,
            Y_fit=Y_fit,
            X_val_raw=X_val_raw,
            Y_val=Y_val,
            X_test_raw=X_test_raw,
        )

        score = float(np.nanmean(loss_vec))

        if score < best_score:
            best_candidate = candidate
            best_loss_vec = loss_vec
            best_epoch_vec = epoch_vec
            best_score = score

    return best_candidate, best_loss_vec, best_epoch_vec


def evaluate_candidate(candidate, X_fit_raw, Y_fit, X_val_raw, Y_val, X_test_raw):
    X_fit, X_val, _ = NNB.prepare_validation(
        model=C.model,
        X_fit_raw=X_fit_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        C=C,
    )

    nmc = int(C.nmc)

    loss_vec = np.full(nmc, np.nan)
    epoch_vec = np.full(nmc, 1, dtype=int)

    if len(candidate["archi"]) == 0:
        loss, epoch = NNB.train_validation(
            X_fit=X_fit,
            Y_fit=Y_fit,
            X_val=X_val,
            Y_val=Y_val,
            candidate=candidate,
            seed=_seed(0),
        )

        loss_vec[:] = loss
        epoch_vec[:] = epoch

        return loss_vec, epoch_vec

    for seed_id in range(nmc):
        loss, epoch = NNB.train_validation(
            X_fit=X_fit,
            Y_fit=Y_fit,
            X_val=X_val,
            Y_val=Y_val,
            candidate=candidate,
            seed=_seed(seed_id),
        )

        loss_vec[seed_id] = loss
        epoch_vec[seed_id] = epoch

    return loss_vec, epoch_vec


def final_seed_forecasts(candidate, X_hist_raw, Y_hist, X_test_raw, epoch_vec):
    X_hist, X_test = NNB.prepare_final(
        model=C.model,
        X_hist_raw=X_hist_raw,
        X_test_raw=X_test_raw,
        C=C,
    )

    nmc = int(C.nmc)
    M = Y_hist.shape[1]

    forecasts = np.full((nmc, M), np.nan)

    if len(candidate["archi"]) == 0:
        pred = NNB.train_final(
            X_train=X_hist,
            Y_train=Y_hist,
            X_test=X_test,
            candidate=candidate,
            seed=_seed(0),
            epochs=1,
        )

        forecasts[:, :] = pred.reshape(1, -1)

        return forecasts

    for seed_id in range(nmc):
        forecasts[seed_id, :] = NNB.train_final(
            X_train=X_hist,
            Y_train=Y_hist,
            X_test=X_test,
            candidate=candidate,
            seed=_seed(seed_id),
            epochs=int(epoch_vec[seed_id]),
        )

    return forecasts


def make_forecast_sample(X, Y, forecast_idx, purge_gap):
    train_end = int(forecast_idx) - int(purge_gap)

    if train_end < 1:
        return None

    X_hist = X[: train_end + 1]
    Y_hist = Y[: train_end + 1]
    X_test = X[forecast_idx:forecast_idx + 1]

    if X_test.shape[0] != 1:
        return None

    return X_hist, Y_hist, X_test


def purged_validation_split(X, Y, validation_frac, purge_gap):
    n = X.shape[0]
    n_val = int(np.ceil(n * float(validation_frac)))
    n_val = max(1, n_val)

    fit_end = n - n_val - int(purge_gap)
    val_start = fit_end + int(purge_gap)

    if fit_end < 30 or val_start >= n:
        return None

    return (
        X[:fit_end],
        Y[:fit_end],
        X[val_start:],
        Y[val_start:],
    )


def finite_design(X_fit, Y_fit, X_val, Y_val, X_test):
    return (
        np.all(np.isfinite(X_fit))
        and np.all(np.isfinite(Y_fit))
        and np.all(np.isfinite(X_val))
        and np.all(np.isfinite(Y_val))
        and np.all(np.isfinite(X_test))
    )


def benchmark_forecast(Y, forecast_idx, purge_gap, benchmark):
    Y = np.asarray(Y, dtype=float)

    if benchmark == "zero":
        return np.zeros(Y.shape[1], dtype=float)

    if benchmark == "historical_mean":
        train_end = int(forecast_idx) - int(purge_gap)

        if train_end < 0:
            return np.full(Y.shape[1], np.nan)

        hist = Y[: train_end + 1, :]

        return np.nanmean(hist, axis=0)

    raise ValueError(f"Unknown benchmark: {benchmark}")


def top_navg_mean(seed_forecasts, seed_loss, navg):
    order = np.argsort(seed_loss)
    keep = order[:int(navg)]

    return np.mean(seed_forecasts[keep, :], axis=0)


def oos_indices(dates, oos_start):
    dates = pd.DatetimeIndex(dates)
    idx = np.where(dates >= pd.Timestamp(oos_start))[0]

    if idx.size == 0:
        raise ValueError("No available date on or after oos_start.")

    return list(range(int(idx[0]), len(dates)))


def load_dataset(data_path, feature_groups, target_group, target_indices=None):
    X_blocks = [
        load_mat_group_frame(data_path, "X", group)
        for group in feature_groups
    ]

    X_df = pd.concat(X_blocks, axis=1)

    Y_df = load_mat_group_frame(data_path, "y", target_group)

    if target_indices is not None:
        Y_df = Y_df.iloc[:, list(target_indices)].copy()

    merged = X_df.join(Y_df, how="inner").dropna(axis=0, how="any")

    X_df = merged[X_df.columns].copy()
    Y_df = merged[Y_df.columns].copy()

    return X_df, Y_df


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

    dt = pd.to_datetime(
        {"year": year, "month": month, "day": np.ones_like(year)},
        errors="raise",
    )

    return pd.DatetimeIndex(dt) + pd.offsets.MonthEnd(0)


def summarize_oos_metrics(Y_true, Y_pred, Y_benchmark, hac_lags):
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)
    Y_benchmark = np.asarray(Y_benchmark, dtype=float)

    mse = np.nanmean((Y_true - Y_pred) ** 2, axis=0)

    r2 = np.array(
        [
            r2_oos(Y_true[:, j], Y_pred[:, j], Y_benchmark[:, j])
            for j in range(Y_true.shape[1])
        ]
    )

    pval = np.array(
        [
            clark_west_pvalue(Y_true[:, j], Y_pred[:, j], Y_benchmark[:, j], hac_lags)
            for j in range(Y_true.shape[1])
        ]
    )

    return mse, r2, pval


def r2_oos(y, f, b):
    y = np.asarray(y, dtype=float).reshape(-1)
    f = np.asarray(f, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    ok = np.isfinite(y) & np.isfinite(f) & np.isfinite(b)

    if ok.sum() == 0:
        return np.nan

    sse_model = np.sum((y[ok] - f[ok]) ** 2)
    sse_bench = np.sum((y[ok] - b[ok]) ** 2)

    if sse_bench <= 0:
        return np.nan

    return 1.0 - sse_model / sse_bench


def clark_west_pvalue(y, f, b, hac_lags):
    y = np.asarray(y, dtype=float).reshape(-1)
    f = np.asarray(f, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    ok = np.isfinite(y) & np.isfinite(f) & np.isfinite(b)

    if ok.sum() < 5:
        return np.nan

    y = y[ok]
    f = f[ok]
    b = b[ok]

    cw = (y - b) ** 2 - ((y - f) ** 2 - (b - f) ** 2)

    se = nw_se_mean(cw, hac_lags)

    if not np.isfinite(se) or se <= 0:
        return np.nan

    tstat = np.mean(cw) / se

    return float(1.0 - norm.cdf(tstat))


def nw_se_mean(x, lags):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    n = x.size

    if n < 2:
        return np.nan

    u = x - np.mean(x)
    omega = np.dot(u, u) / n

    max_lag = min(int(lags), n - 1)

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = np.dot(u[lag:], u[:-lag]) / n
        omega += 2.0 * weight * gamma

    if omega < 0:
        return np.nan

    return np.sqrt(omega / n)


def save_result(out_file, result, X_df, Y_df):
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    save_dict = {
        "Y_True": Y_df.to_numpy(dtype=float),
        "Y_Forecast": result["Y_Forecast"],
        "Y_Forecast_All": result["Y_Forecast_All"],
        "Y_Benchmark": result["Y_Benchmark"],
        "ValLoss": result["ValLoss"],
        "MSE": result["MSE"],
        "R2OOS": result["R2OOS"],
        "R2OOS_pval": result["R2OOS_pval"],
        "Dates": np.array(pd.DatetimeIndex(X_df.index).strftime("%Y-%m-%d"), dtype=object),
        "X_Columns": np.array(X_df.columns.astype(str), dtype=object),
        "Y_Columns": np.array(Y_df.columns.astype(str), dtype=object),
        "DataPath": C.data_path,
        "FeatureGroupsJSON": json.dumps(C.feature_groups),
        "TargetGroup": C.target_group,
        "TargetIndicesJSON": json.dumps(C.target_indices),
        "Model": C.model,
        "Benchmark": C.benchmark,
        "RunTag": C.run_tag,
        "ConfigJSON": json.dumps(public_config(), default=str),
    }

    sio.savemat(out_file, save_dict)


def public_config():
    return {
        key: value
        for key, value in C.__dict__.items()
        if not key.startswith("_")
        and isinstance(value, (str, int, float, bool, list, tuple, type(None)))
    }


def _seed(seed_id):
    return _BASE_SEED + int(seed_id)


if __name__ == "__main__":
    main()