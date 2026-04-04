#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import copy
import json

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Utils import (
    load_dataset,
    r2_oos,
    r2_oos_pvalue,
    extract_summary_rows,
    save_sweep_summary_to_excel,
)

RUN_MODE = "sweep"
PURGE_SIZE = 12

BASE_CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["dy_pc1", "dy_pc2", "dy_pc3"],
    "target_group": "dy",
    "horizon": 12,
    "oos_start": "1989-01-31",
    "run_tag": "pc123",
    "params": {
        "model_name": "Ridge",
        "alpha": 0.1,
        "l1_ratio": 0.5,
        "standardize": True,
        "max_iter": 10000,
        "validation_split": 0.15,
    },
}

SWEEP_GRID = {
    "model_name": ["Ridge", "Lasso", "ElasticNet"],
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "l1_ratio": [0.2, 0.5, 0.8],
}


def _glm_result_name(cfg, sweep=False):
    p = cfg["params"]
    target = cfg["target_group"]
    feat = cfg["run_tag"]
    horizon = f"h{cfg['horizon']}"

    if sweep:
        return "__".join(["sweep", target, feat, "GLM", horizon])

    model = p["model_name"]
    alpha = f"a{float(p['alpha']):g}"
    std = "std1" if bool(p["standardize"]) else "std0"

    parts = [target, feat, model, horizon, alpha, std]

    if model == "ElasticNet":
        parts.append(f"l1r{float(p['l1_ratio']):g}")

    return "__".join(parts)


def _build_estimator(params):
    model_name = params["model_name"]
    alpha = float(params["alpha"])
    l1_ratio = float(params.get("l1_ratio", 0.5))
    standardize = bool(params.get("standardize", True))
    max_iter = int(params.get("max_iter", 10000))

    if model_name == "Ridge":
        model = Ridge(alpha=alpha, fit_intercept=True)
    elif model_name == "Lasso":
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter, random_state=0)
    elif model_name == "ElasticNet":
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=max_iter,
            random_state=0,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if standardize:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    return model


def fit_predict_glm(X_train, Y_train, X_test, params):
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    if Y_train.ndim == 1:
        Y_train = Y_train.reshape(-1, 1)

    n_targets = Y_train.shape[1]
    y_hat = np.full((X_test.shape[0], n_targets), np.nan)

    for k in range(n_targets):
        estimator = _build_estimator(params)
        estimator.fit(X_train, Y_train[:, k])
        y_hat[:, k] = estimator.predict(X_test)

    return y_hat


def _purged_split(X_train, Y_train, validation_split, purge_size=PURGE_SIZE):
    n = X_train.shape[0]
    val_len = int(np.ceil(n * float(validation_split)))
    fit_end = n - purge_size - val_len
    val_start = fit_end + purge_size

    X_fit = X_train[:fit_end, :]
    Y_fit = Y_train[:fit_end, :]
    X_val = X_train[val_start:, :]
    Y_val = Y_train[val_start:, :]

    return X_fit, Y_fit, X_val, Y_val


def _validation_mse(X_train, Y_train, params, validation_split):
    X_fit, Y_fit, X_val, Y_val = _purged_split(
        X_train,
        Y_train,
        validation_split=validation_split,
    )
    Y_val_hat = fit_predict_glm(X_fit, Y_fit, X_val, params)
    return float(np.mean((Y_val - Y_val_hat) ** 2))


def run_oos_forecast(X, Y, dates, cfg):
    oos_start_ts = pd.Timestamp(cfg["oos_start"])
    start_candidates = np.where(dates >= oos_start_ts)[0]

    if start_candidates.size == 0:
        raise ValueError(
            f"No available sample date on or after oos_start={oos_start_ts.date()}. "
            f"Sample runs from {dates.min().date()} to {dates.max().date()}."
        )

    start_idx = int(start_candidates[0])
    T, M = Y.shape
    Y_forecast = np.full((T, M), np.nan)
    val_loss = np.full(T, np.nan)

    total_oos = T - start_idx
    done = 0

    model_name = cfg["params"]["model_name"] if RUN_MODE == "single" else "GLM"
    print(f"Starting {model_name}: {total_oos} OOS steps")

    candidates = None
    if RUN_MODE == "sweep":
        candidates = []
        for combo in ParameterGrid(SWEEP_GRID):
            params = copy.deepcopy(cfg["params"])
            params.update(combo)
            if params["model_name"] != "ElasticNet":
                params["l1_ratio"] = 0.5
            candidates.append(params)

    for test_idx in range(start_idx, T):
        train_end = test_idx - cfg["horizon"]
        if train_end < 0:
            continue

        X_train = X[: train_end + 1, :]
        Y_train = Y[: train_end + 1, :]
        X_test = X[test_idx : test_idx + 1, :]

        if RUN_MODE == "single":
            best_params = cfg["params"]
        else:
            losses = [
                _validation_mse(
                    X_train,
                    Y_train,
                    params=p,
                    validation_split=cfg["params"].get("validation_split", 0.15),
                )
                for p in candidates
            ]
            best_idx = int(np.argmin(losses))
            best_params = candidates[best_idx]
            val_loss[test_idx] = losses[best_idx]

        Y_forecast[test_idx, :] = fit_predict_glm(X_train, Y_train, X_test, best_params)[0, :]

        done += 1
        if (done % 10 == 0) or (done == total_oos):
            pct = 100.0 * done / total_oos
            msg = (
                f"[{model_name}] "
                f"{done:4d}/{total_oos} "
                f"({pct:5.1f}%)  "
                f"date={dates[test_idx].strftime('%Y-%m-%d')}"
            )
            if RUN_MODE == "sweep":
                msg += (
                    f"  best={best_params['model_name']}"
                    f"  alpha={best_params['alpha']:g}"
                )
                if best_params["model_name"] == "ElasticNet":
                    msg += f"  l1r={best_params['l1_ratio']:g}"
                msg += f"  val={val_loss[test_idx]:.6f}"
            print(msg)

    model_key = cfg["params"]["model_name"] if RUN_MODE == "single" else "GLM"

    out = {
        "Y_zero_benchmark": np.zeros_like(Y),
        f"Y_forecast_agg_{model_key}": Y_forecast,
        f"MSE_{model_key}": np.nanmean((Y - Y_forecast) ** 2, axis=0),
        f"R2OOS_{model_key}": np.array(
            [r2_oos(Y[:, k], Y_forecast[:, k]) for k in range(M)]
        ),
        f"R2OOS_pval_{model_key}": np.array(
            [r2_oos_pvalue(Y[:, k], Y_forecast[:, k], hac_lags=cfg["horizon"]) for k in range(M)]
        ),
    }

    if RUN_MODE == "sweep":
        out["ValLoss_GLM"] = val_loss

    return out


def run_experiment(custom_config=None):
    cfg = copy.deepcopy(custom_config if custom_config is not None else BASE_CONFIG)

    X_df, Y_df = load_dataset(
        mat_path=cfg["mat_path"],
        feature_groups=cfg["feature_groups"],
        target_group=cfg["target_group"],
    )

    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(X_df.index)

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"OOS start: {cfg['oos_start']}")
    print(f"Feature groups: {cfg['feature_groups']}")
    print(f"Params: {json.dumps(cfg['params'])}")

    model_name = cfg["params"]["model_name"] if RUN_MODE == "single" else "GLM"

    save_dict = {
        "Note": (
            "Expanding-window OOS penalized linear model with horizon-consistent embargo. "
            "Sweep mode uses purged validation for hyperparameter choice."
        ),
        "MatPath": cfg["mat_path"],
        "FeatureGroupsJSON": json.dumps(cfg["feature_groups"]),
        "TargetGroup": cfg["target_group"],
        "ModelName": model_name,
        "ParamsJSON": json.dumps(cfg["params"]),
        "RunConfigJSON": json.dumps(
            {
                "horizon": cfg["horizon"],
                "oos_start": cfg["oos_start"],
                "run_tag": cfg["run_tag"],
                "run_mode": RUN_MODE,
            }
        ),
        "Horizon": cfg["horizon"],
        "Y_True": Y,
        "Dates": np.array(dates.strftime("%Y-%m-%d"), dtype=object),
        "X_Columns": np.array(X_df.columns, dtype=object),
        "Y_Columns": np.array(Y_df.columns, dtype=object),
    }

    if RUN_MODE == "sweep":
        save_dict["SweepGridJSON"] = json.dumps(SWEEP_GRID)

    save_dict.update(run_oos_forecast(X, Y, dates, cfg))

    os.makedirs("results", exist_ok=True)
    out_mat = os.path.join("results", _glm_result_name(cfg, sweep=(RUN_MODE == "sweep")) + ".mat")
    sio.savemat(out_mat, save_dict)

    print("Saved:", out_mat)
    print("R2OOS:", save_dict[f"R2OOS_{model_name}"])

    return save_dict, out_mat, X_df.columns.tolist(), Y_df.columns.tolist()


def build_sweep_configs(base_config, sweep_grid):
    cfg_list = []

    for combo in ParameterGrid(sweep_grid):
        cfg = copy.deepcopy(base_config)
        cfg["params"].update(combo)

        if cfg["params"]["model_name"] != "ElasticNet":
            cfg["params"]["l1_ratio"] = 0.5

        cfg_list.append(cfg)

    return cfg_list


def run_hyperparameter_sweep(base_config=None, sweep_grid=None):
    base_cfg = copy.deepcopy(base_config if base_config is not None else BASE_CONFIG)
    grid = copy.deepcopy(sweep_grid if sweep_grid is not None else SWEEP_GRID)

    os.makedirs("results", exist_ok=True)

    cfg_list = build_sweep_configs(base_cfg, grid)

    all_summary_rows = []
    all_r2_rows = []
    all_pval_rows = []
    all_mse_rows = []

    total_runs = len(cfg_list)
    print(f"Total runs: {total_runs}")

    for run_no, cfg in enumerate(cfg_list, start=1):
        print("=" * 100)
        print(f"Run {run_no}/{total_runs}")
        print("params:", json.dumps(cfg["params"], ensure_ascii=False))

        save_dict, mat_file, _, y_columns = run_experiment(cfg)

        model_name = cfg["params"]["model_name"] if RUN_MODE == "single" else "GLM"
        cfg_for_summary = copy.deepcopy(cfg)
        cfg_for_summary["model_name"] = model_name

        summary_rows, r2_rows, pval_rows, mse_rows = extract_summary_rows(
            save_dict=save_dict,
            cfg=cfg_for_summary,
            mat_file=mat_file,
            y_columns=y_columns,
        )

        all_summary_rows.extend(summary_rows)
        all_r2_rows.extend(r2_rows)
        all_pval_rows.extend(pval_rows)
        all_mse_rows.extend(mse_rows)

    out_xlsx = os.path.join("results", _glm_result_name(base_cfg, sweep=True) + ".xlsx")

    save_sweep_summary_to_excel(
        summary_rows=all_summary_rows,
        r2_rows=all_r2_rows,
        pval_rows=all_pval_rows,
        mse_rows=all_mse_rows,
        out_xlsx=out_xlsx,
    )

    print("Saved Excel summary to", out_xlsx)

    return {
        "summary": pd.DataFrame(all_summary_rows),
        "r2_by_target": pd.DataFrame(all_r2_rows),
        "pval_by_target": pd.DataFrame(all_pval_rows),
        "mse_by_target": pd.DataFrame(all_mse_rows),
        "xlsx_file": out_xlsx,
    }


if __name__ == "__main__":
    if RUN_MODE == "single":
        run_experiment(BASE_CONFIG)
    elif RUN_MODE == "sweep":
        run_hyperparameter_sweep(BASE_CONFIG, SWEEP_GRID)
    else:
        raise ValueError("RUN_MODE must be 'single' or 'sweep'.")