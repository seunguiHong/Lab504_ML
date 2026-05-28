#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json

import numpy as np
import pandas as pd

from utils import (
    load_dataset,
    enumerate_oos_forecast_indices,
    compute_training_end_index,
    summarize_oos_metrics,
    build_save_dict,
    save_results_mat,
)


CONFIG = {
    "data_path": "data/target_and_features.mat",
    "feature_groups": ["fwd"],
    "target_group": "rx",
    "target_indices": None,

    "horizon": 12,
    "oos_start": "1989-01-31",
    "run_tag": "panelB_ols_fwd",
    "out_file": "results/tab1_yc/panelA_rxols_fwd.mat",

    "params": {
        "add_intercept": True,
    },
}


def with_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_predict_ols(X_train, Y_train, X_test, add_intercept=True):
    if add_intercept:
        X_train = with_intercept(X_train)
        X_test = with_intercept(X_test)

    beta, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)
    return X_test @ beta


def run_oos_forecast(X, Y, dates, cfg):
    dates = pd.DatetimeIndex(dates)
    oos_indices = enumerate_oos_forecast_indices(dates, cfg["oos_start"])

    T, M = Y.shape
    horizon = int(cfg["horizon"])
    add_intercept = bool(cfg["params"].get("add_intercept", True))

    Y_forecast = np.full((T, M), np.nan)

    print(f"Total OOS steps: {len(oos_indices)}")

    for step, forecast_idx in enumerate(oos_indices, start=1):
        train_end = compute_training_end_index(forecast_idx, horizon)

        if train_end < 1:
            continue

        X_train = X[: train_end + 1]
        Y_train = Y[: train_end + 1]
        X_test = X[forecast_idx : forecast_idx + 1]

        if not np.all(np.isfinite(X_test)):
            continue

        ok = np.all(np.isfinite(X_train), axis=1) & np.all(np.isfinite(Y_train), axis=1)
        X_train = X_train[ok]
        Y_train = Y_train[ok]

        min_obs = X_train.shape[1] + 2 if add_intercept else X_train.shape[1] + 1

        if X_train.shape[0] < min_obs:
            continue

        Y_forecast[forecast_idx] = fit_predict_ols(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            add_intercept=add_intercept,
        )[0]

        if step == 1 or step % 12 == 0 or step == len(oos_indices):
            r2_now = summarize_oos_metrics(
                Y_true=Y,
                Y_pred=Y_forecast,
                hac_lags=horizon,
            )[1]

            print(
                f"[{step:4d}/{len(oos_indices)}] "
                f"date={dates[forecast_idx].strftime('%Y-%m-%d')}"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    mse, r2, pval = summarize_oos_metrics(
        Y_true=Y,
        Y_pred=Y_forecast,
        hac_lags=horizon,
    )

    return {
        "Y_Forecast": Y_forecast,
        "MSE": mse,
        "R2OOS": r2,
        "R2OOS_pval": pval,
    }


def run_experiment(cfg=None):
    cfg = copy.deepcopy(CONFIG if cfg is None else cfg)

    X_df, Y_df = load_dataset(
        data_path=cfg["data_path"],
        feature_groups=cfg["feature_groups"],
        target_group=cfg["target_group"],
        target_indices=cfg.get("target_indices"),
    )

    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(X_df.index)

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"OOS start: {cfg['oos_start']}")
    print(f"Feature groups: {cfg['feature_groups']}")
    print(f"Run tag: {cfg['run_tag']}")
    print(f"Params: {json.dumps(cfg['params'])}")

    result = run_oos_forecast(X, Y, dates, cfg)

    save_dict = build_save_dict(cfg, X_df, Y_df, Y)
    save_dict.update(result)
    save_results_mat(cfg["out_file"], save_dict)

    print("Saved to", cfg["out_file"])
    print("R2OOS:", np.round(save_dict["R2OOS"], 4))

    return save_dict, cfg["out_file"]


if __name__ == "__main__":
    run_experiment(CONFIG)