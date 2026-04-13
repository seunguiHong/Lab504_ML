#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import json

import numpy as np
import pandas as pd

from utils import (
    load_dataset,
    summarize_oos_metrics,
    build_save_dict,
    result_name,
    save_results_mat,
)


BASE_CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["d12m_y_pc"],
    "target_group": "dy",
    "target_indices": None,
    "horizon": 12,
    "oos_start": "1989-01-31",
    "run_tag": "OLSOLSOLSOLSOLSOLSOLSOLSOLSOLSOLSOLS",
    "model_name": "OLS",
    "results_dir": "results",
    "params": {
        "add_intercept": True,
    },
}


def _add_intercept(X):
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])


def _fit_predict_ols(X_train, Y_train, X_test, add_intercept=True):
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    if Y_train.ndim == 1:
        Y_train = Y_train.reshape(-1, 1)

    if add_intercept:
        X_train_reg = _add_intercept(X_train)
        X_test_reg = _add_intercept(X_test)
    else:
        X_train_reg = X_train
        X_test_reg = X_test

    beta, _, _, _ = np.linalg.lstsq(X_train_reg, Y_train, rcond=None)
    Y_hat = X_test_reg @ beta
    return np.asarray(Y_hat, dtype=float)


def run_oos_forecast(X, Y, dates, cfg):
    dates = pd.DatetimeIndex(dates)
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
    total_oos = T - start_idx
    done = 0

    print(f"Starting OLS: {total_oos} OOS steps")

    for test_idx in range(start_idx, T):
        train_end = test_idx - int(cfg["horizon"])
        if train_end < 1:
            continue

        X_train = X[: train_end + 1, :]
        Y_train = Y[: train_end + 1, :]
        X_test = X[test_idx : test_idx + 1, :]

        valid_train = np.all(np.isfinite(X_train), axis=1) & np.all(np.isfinite(Y_train), axis=1)
        X_train = X_train[valid_train]
        Y_train = Y_train[valid_train]

        if X_train.shape[0] <= X_train.shape[1]:
            continue

        if not np.all(np.isfinite(X_test)):
            continue

        Y_forecast[test_idx, :] = _fit_predict_ols(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            add_intercept=bool(cfg["params"].get("add_intercept", True)),
        )[0, :]

        done += 1
        if (done % 10 == 0) or (done == total_oos):
            pct = 100.0 * done / total_oos
            r2_now = summarize_oos_metrics(
                Y_true=Y,
                Y_pred=Y_forecast,
                hac_lags=int(cfg["horizon"]),
            )[1]
            print(
                f"[OLS] "
                f"{done:4d}/{total_oos} "
                f"({pct:5.1f}%)  "
                f"date={dates[test_idx].strftime('%Y-%m-%d')}"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    mse_vec, r2_vec, pval_vec = summarize_oos_metrics(
        Y_true=Y,
        Y_pred=Y_forecast,
        hac_lags=int(cfg["horizon"]),
    )

    return {
        "Y_zero_benchmark": np.zeros_like(Y),
        "Y_forecast_agg_OLS": Y_forecast,
        "MSE_OLS": mse_vec,
        "R2OOS_OLS": r2_vec,
        "R2OOS_pval_OLS": pval_vec,
    }


def run_experiment(custom_config=None):
    cfg = copy.deepcopy(custom_config if custom_config is not None else BASE_CONFIG)

    X_df, Y_df = load_dataset(
        mat_path=cfg["mat_path"],
        feature_groups=cfg["feature_groups"],
        target_group=cfg["target_group"],
        target_indices=cfg.get("target_indices", None),
    )

    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(X_df.index)

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"OOS start: {cfg['oos_start']}")
    print(f"Feature groups: {cfg['feature_groups']}")
    print(f"Params: {json.dumps(cfg['params'])}")

    save_dict = build_save_dict(
        cfg=cfg,
        X_df=X_df,
        Y_df=Y_df,
        Y_true=Y,
        extra_metadata={
            "Note": (
                "Expanding-window OOS ordinary least squares predictive regression "
                "with horizon h=12 and optional intercept."
            )
        },
    )
    save_dict.update(run_oos_forecast(X, Y, dates, cfg))

    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    out_mat = os.path.join(results_dir, result_name(cfg) + ".mat")
    save_results_mat(out_mat, save_dict)

    print("Saved:", out_mat)
    print("R2OOS:", np.round(save_dict["R2OOS_OLS"], 4))

    return save_dict, out_mat, X_df.columns.tolist(), Y_df.columns.tolist()


if __name__ == "__main__":
    run_experiment(BASE_CONFIG)