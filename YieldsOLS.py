#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import scipy.io as sio

from Utils import load_dataset, r2_oos, r2_oos_pvalue


CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["dy_pc2"],
    "target_group": "dy",
    "horizon": 12,
    "oos_start": "1989-01-31",
    "run_tag": "OLS_Xdy_pc1__Y_dy",
    "model_name": "OLS",
}


def fit_predict_ols(X_train, Y_train, X_test):
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    X_design = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_test_design = np.column_stack([np.ones(X_test.shape[0]), X_test])

    beta, _, _, _ = np.linalg.lstsq(X_design, Y_train, rcond=None)
    return X_test_design @ beta


def run_oos_forecast(X, Y, dates, cfg):
    oos_start_ts = pd.Timestamp(cfg["oos_start"])
    start_candidates = np.where(dates >= oos_start_ts)[0]

    if start_candidates.size == 0:
        raise ValueError(
            f"No available sample date on or after oos_start={oos_start_ts.date()}. "
            f"Sample runs from {dates.min().date()} to {dates.max().date()}."
        )

    start_idx = int(start_candidates[0])

    if dates[start_idx] != oos_start_ts:
        print(
            f"Requested OOS start {oos_start_ts.date()} is not in the merged sample. "
            f"Using first available date {dates[start_idx].date()} instead."
        )

    T, M = Y.shape
    Y_forecast = np.full((T, M), np.nan)

    total_oos = T - start_idx
    done = 0

    print(f"Starting {cfg['model_name']}: {total_oos} OOS steps")

    for test_idx in range(start_idx, T):
        train_end = test_idx - cfg["horizon"]
        if train_end < 0:
            continue

        X_train = X[: train_end + 1, :]
        Y_train = Y[: train_end + 1, :]
        X_test = X[test_idx : test_idx + 1, :]

        Y_forecast[test_idx, :] = fit_predict_ols(X_train, Y_train, X_test)[0, :]

        done += 1
        if (done % 10 == 0) or (done == total_oos):
            pct = 100.0 * done / total_oos
            print(
                f"[{cfg['model_name']}] "
                f"{done:4d}/{total_oos} "
                f"({pct:5.1f}%)  "
                f"date={dates[test_idx].strftime('%Y-%m-%d')}"
            )

    return {
        "Y_zero_benchmark": np.zeros_like(Y),
        f"Y_forecast_agg_{cfg['model_name']}": Y_forecast,
        f"MSE_{cfg['model_name']}": np.nanmean((Y - Y_forecast) ** 2, axis=0),
        f"R2OOS_{cfg['model_name']}": np.array(
            [r2_oos(Y[:, k], Y_forecast[:, k]) for k in range(Y.shape[1])]
        ),
        f"R2OOS_pval_{cfg['model_name']}": np.array(
            [r2_oos_pvalue(Y[:, k], Y_forecast[:, k], hac_lags=cfg["horizon"]) for k in range(Y.shape[1])]
        ),
    }


def main():
    cfg = CONFIG

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

    save_dict = {
        "Note": (
            "Expanding-window OOS linear regression with horizon-consistent embargo. "
            "Target dy denotes yield change. Benchmark is zero."
        ),
        "Horizon": cfg["horizon"],
        "Y_True": Y,
        "Dates": np.array(dates.strftime("%Y-%m-%d"), dtype=object),
        "X_Columns": np.array(X_df.columns, dtype=object),
        "Y_Columns": np.array(Y_df.columns, dtype=object),
    }

    save_dict.update(run_oos_forecast(X, Y, dates, cfg))

    os.makedirs("results", exist_ok=True)
    out_file = f"results/{cfg['target_group']}__{cfg['model_name']}__h{cfg['horizon']}__{cfg['run_tag']}.mat"
    sio.savemat(out_file, save_dict)

    print("Saved:", out_file)
    print("R2OOS:", save_dict[f"R2OOS_{cfg['model_name']}"])


if __name__ == "__main__":
    main()