#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import shutil
import hashlib
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.api as sm

from scipy.stats import t as tstat


def to_1d(x):
    return np.asarray(x).squeeze()


def to_name_list(x):
    x = np.asarray(x).squeeze()
    if x.ndim == 0:
        return [str(x.item())]
    return [str(v) for v in x.tolist()]


def yyyymm_to_month_end(x):
    x = np.asarray(x).astype(int).ravel()
    return pd.to_datetime(x.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)


def load_dataset(mat_path, feature_groups, target_group):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    Xmat = mat["X"]
    ymat = mat["y"]

    x_parts = []
    for group_name in feature_groups:
        block = getattr(Xmat, group_name)
        time_vec = to_1d(block.Time).astype(int)
        data = np.asarray(block.data, dtype=float)
        names = to_name_list(block.names)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        group_dates = yyyymm_to_month_end(time_vec)

        if group_name != "yields":
            names = [f"{group_name}__{name}" for name in names]

        x_parts.append(pd.DataFrame(data, index=group_dates, columns=names))

    X_df = pd.concat(x_parts, axis=1)
    X_df.index.name = "Date"

    yblock = getattr(ymat, target_group)
    y_time = to_1d(yblock.Time).astype(int)
    y_data = np.asarray(yblock.data, dtype=float)
    y_names = to_name_list(yblock.names)

    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)

    y_dates = yyyymm_to_month_end(y_time)
    Y_df = pd.DataFrame(y_data, index=y_dates, columns=y_names)
    Y_df.index.name = "Date"

    data = X_df.join(Y_df, how="inner").dropna().copy()
    return data[X_df.columns], data[Y_df.columns]


def zero_benchmark(y_true):
    return np.zeros_like(y_true, dtype=float)


def r2_oos(y_true, y_forecast):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_forecast = np.asarray(y_forecast, dtype=float).ravel()
    y_benchmark = zero_benchmark(y_true)

    valid = (~np.isnan(y_true)) & (~np.isnan(y_forecast))
    if valid.sum() == 0:
        return np.nan

    ss_res = np.sum((y_true[valid] - y_forecast[valid]) ** 2)
    ss_bmk = np.sum((y_true[valid] - y_benchmark[valid]) ** 2)

    if not np.isfinite(ss_bmk) or ss_bmk == 0:
        return np.nan

    return 1.0 - ss_res / ss_bmk


def r2_oos_pvalue(y_true, y_forecast, hac_lags=12):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_forecast = np.asarray(y_forecast, dtype=float).ravel()
    y_benchmark = zero_benchmark(y_true)

    valid = (~np.isnan(y_true)) & (~np.isnan(y_forecast))
    if valid.sum() < 2:
        return np.nan

    yt = y_true[valid]
    yf = y_forecast[valid]
    y0 = y_benchmark[valid]

    f = np.square(yt - y0) - np.square(yt - yf) + np.square(y0 - yf)

    try:
        x = np.ones_like(f)
        model = sm.OLS(f, x, hasconst=True)
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        return 1.0 - tstat.cdf(results.tvalues[0], results.nobs - 1)
    except Exception:
        return np.nan


def summarize_oos_metrics(Y_true, Y_pred):
    mse_vec = np.nanmean(np.square(Y_true - Y_pred), axis=0)
    r2_vec = np.array([r2_oos(Y_true[:, k], Y_pred[:, k]) for k in range(Y_true.shape[1])])
    pval_vec = np.array([r2_oos_pvalue(Y_true[:, k], Y_pred[:, k]) for k in range(Y_true.shape[1])])
    return mse_vec, r2_vec, pval_vec


def get_cpu_count():
    raw = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if raw is not None:
        token = str(raw).split(",")[0].split("(")[0]
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            return max(1, int(digits))
    return max(1, mp.cpu_count())


def make_dump_dir(base="./trainingDumps_"):
    idx = 0
    while True:
        path = os.path.abspath(f"{base}{idx}")
        if not os.path.exists(path):
            os.mkdir(path)
            return path
        idx += 1


def safe_rmtree(path):
    try:
        shutil.rmtree(path)
        print("Removed dir:", path)
    except Exception:
        print("Directory:", path, "could not be removed")


def build_model_input(X, Y, test_idx, horizon):
    train_end = test_idx - horizon
    if train_end < 0:
        return None, None

    X_train = X[: train_end + 1, :]
    Y_train = Y[: train_end + 1, :]
    X_test = X[test_idx : test_idx + 1, :]
    Y_test = Y[test_idx : test_idx + 1, :]

    X_model = np.vstack([X_train, X_test])
    Y_model = np.vstack([Y_train, Y_test])

    return X_model, Y_model


def run_parallel_seeds(model_func, ncpus, nmc, X, Y, **kwargs):
    outputs = None

    while outputs is None:
        pool = None
        try:
            pool = mp.Pool(processes=ncpus)
            jobs = [
                pool.apply_async(model_func, args=(X, Y, seed), kwds=kwargs)
                for seed in range(nmc)
            ]
            outputs = [job.get(timeout=500) for job in jobs]
            pool.close()
            pool.join()
            time.sleep(1)
        except Exception as e:
            print(str(e))
            print("Timed out, shutting pool down")

            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    pass
                try:
                    pool.terminate()
                except Exception:
                    pass

            time.sleep(1)

    return outputs


def build_save_dict(cfg, X_df, Y_df, Y_true):
    save_dict = {
        "Note": json.dumps(cfg["params"]),
        "MatPath": cfg["mat_path"],
        "FeatureGroupsJSON": json.dumps(cfg["feature_groups"]),
        "TargetGroup": cfg["target_group"],
        "ModelName": cfg["model_name"],
        "ParamsJSON": json.dumps(cfg["params"]),
        "RunConfigJSON": json.dumps({
            "horizon": cfg["horizon"],
            "oos_start": cfg["oos_start"],
            "hyper_freq": cfg["hyper_freq"],
            "nmc": cfg["nmc"],
            "navg": cfg["navg"],
            "run_tag": cfg["run_tag"],
        }),
        "Horizon": cfg["horizon"],
        "Y_True": Y_true,
        "Dates": np.array(X_df.index.strftime("%Y-%m-%d"), dtype=object),
        "X_Columns": np.array(X_df.columns, dtype=object),
        "Y_Columns": np.array(Y_df.columns, dtype=object),
    }

    if X_df.shape[1] >= 1:
        save_dict["RF"] = X_df.to_numpy(dtype=float)[:, 0]

    return save_dict


def flatten_params(params):
    row = {}
    for key, value in params.items():
        row[key] = json.dumps(value) if isinstance(value, (list, tuple, dict)) else value
    return row


def run_id(params):
    txt = json.dumps(
        params,
        sort_keys=True,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
    )
    return hashlib.md5(txt.encode("utf-8")).hexdigest()[:8]


def result_name(cfg, sweep=False):
    p = cfg["params"]
    model = str(cfg["model_name"]).replace("Model", "")
    arch = "x".join(map(str, p["archi"]))
    reg = "-".join(f"{v:g}" for v in np.asarray(p["l1l2"], dtype=float).ravel()[:2])

    if sweep:
        parts = [
            "sweep",
            cfg["target_group"],
            cfg["run_tag"],
            model,
            f"h{cfg['horizon']}",
        ]
        return "__".join(parts)

    parts = [
        cfg["target_group"],
        cfg["run_tag"],
        model,
        f"h{cfg['horizon']}",
        f"a{arch}",
        f"do{p['Dropout']:g}",
        f"reg{reg}",
        f"lr{p['learning_rate']:g}",
        run_id(p),
    ]
    return "__".join(parts)


def extract_summary_rows(save_dict, cfg, mat_file, y_columns):
    model_name = cfg["model_name"]

    r2_vec = np.asarray(save_dict[f"R2OOS_{model_name}"]).ravel()
    pval_vec = np.asarray(save_dict[f"R2OOS_pval_{model_name}"]).ravel()
    mse_vec = np.asarray(save_dict[f"MSE_{model_name}"]).ravel()

    summary_row = {
        "mat_file": os.path.basename(mat_file),
        "run_tag": cfg["run_tag"],
        "model_name": model_name,
        "target_group": cfg["target_group"],
        "horizon": cfg["horizon"],
        "oos_start": cfg["oos_start"],
        "hyper_freq": cfg["hyper_freq"],
        "nmc": cfg["nmc"],
        "navg": cfg["navg"],
        "mean_R2OOS": np.nanmean(r2_vec),
        "median_R2OOS": np.nanmedian(r2_vec),
        "mean_pval": np.nanmean(pval_vec),
        "mean_MSE": np.nanmean(mse_vec),
    }
    summary_row.update(flatten_params(cfg["params"]))

    rows_r2 = []
    rows_pval = []
    rows_mse = []

    for target, r2_val, p_val, mse_val in zip(y_columns, r2_vec, pval_vec, mse_vec):
        rows_r2.append({"mat_file": os.path.basename(mat_file), "target": target, "R2OOS": r2_val})
        rows_pval.append({"mat_file": os.path.basename(mat_file), "target": target, "p_value": p_val})
        rows_mse.append({"mat_file": os.path.basename(mat_file), "target": target, "MSE": mse_val})

    return [summary_row], rows_r2, rows_pval, rows_mse


def save_sweep_summary_to_excel(summary_rows, r2_rows, pval_rows, mse_rows, out_xlsx):
    df_summary = pd.DataFrame(summary_rows)
    df_r2 = pd.DataFrame(r2_rows)
    df_pval = pd.DataFrame(pval_rows)
    df_mse = pd.DataFrame(mse_rows)

    if not df_summary.empty:
        df_summary = df_summary.sort_values(["mean_R2OOS", "mean_MSE"], ascending=[False, True])

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_r2.to_excel(writer, sheet_name="r2_by_target", index=False)
        df_pval.to_excel(writer, sheet_name="pval_by_target", index=False)
        df_mse.to_excel(writer, sheet_name="mse_by_target", index=False)

        if not df_r2.empty:
            df_r2.pivot_table(index="mat_file", columns="target", values="R2OOS").to_excel(writer, sheet_name="pivot_r2")
        if not df_pval.empty:
            df_pval.pivot_table(index="mat_file", columns="target", values="p_value").to_excel(writer, sheet_name="pivot_pval")
        if not df_mse.empty:
            df_mse.pivot_table(index="mat_file", columns="target", values="MSE").to_excel(writer, sheet_name="pivot_mse")