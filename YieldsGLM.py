#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import copy
import json

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Utils import (
    load_dataset,
    r2_oos,
    r2_oos_pvalue,
)

PURGE_SIZE = 12

CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["dy_pc","macro"],
    "target_group": "dy",
    "horizon": 12,
    "oos_start": "1989-01-31",
    "run_tag": "pc123+macro",
    "params": {
        "model_name": "Ridge",                 # "Ridge" or "Lasso"
        "alpha_grid": [1e-2,1,1e2],
        "standardize": True,
        "max_iter": 10000,
        "validation_split": 0.15,
    },
}


def _glm_result_name(cfg):
    p = cfg["params"]
    target = cfg["target_group"]
    feat = cfg["run_tag"]
    model = p["model_name"]
    horizon = f"h{cfg['horizon']}"
    std = "std1" if bool(p["standardize"]) else "std0"
    return "__".join([target, feat, model, horizon, std, "alpha_search"])


def _build_estimator(params):
    model_name = params["model_name"]
    alpha = float(params["alpha"])
    standardize = bool(params.get("standardize", True))
    max_iter = int(params.get("max_iter", 10000))

    if model_name == "Ridge":
        model = Ridge(alpha=alpha, fit_intercept=True)
    elif model_name == "Lasso":
        model = Lasso(
            alpha=alpha,
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


def _build_alpha_candidates(base_params):
    alpha_grid = base_params.get("alpha_grid", None)
    if alpha_grid is None or len(alpha_grid) == 0:
        raise ValueError("params['alpha_grid'] must be a non-empty list.")

    candidates = []
    for alpha in alpha_grid:
        cand = copy.deepcopy(base_params)
        cand["alpha"] = float(alpha)
        candidates.append(cand)

    return candidates


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
    best_alpha = np.full(T, np.nan)
    val_loss = np.full(T, np.nan)

    alpha_candidates = _build_alpha_candidates(cfg["params"])
    alpha_grid = np.array([float(p["alpha"]) for p in alpha_candidates], dtype=float)
    alpha_loss_path = np.full((T, len(alpha_grid)), np.nan)

    total_oos = T - start_idx
    done = 0
    model_name = cfg["params"]["model_name"]

    print(f"Starting {model_name}: {total_oos} OOS steps")

    for test_idx in range(start_idx, T):
        train_end = test_idx - cfg["horizon"]
        if train_end < 0:
            continue

        X_train = X[: train_end + 1, :]
        Y_train = Y[: train_end + 1, :]
        X_test = X[test_idx : test_idx + 1, :]

        losses = []
        for j, cand_params in enumerate(alpha_candidates):
            loss_j = _validation_mse(
                X_train,
                Y_train,
                params=cand_params,
                validation_split=cfg["params"].get("validation_split", 0.15),
            )
            losses.append(loss_j)
            alpha_loss_path[test_idx, j] = loss_j

        best_idx = int(np.argmin(losses))
        best_params = alpha_candidates[best_idx]

        val_loss[test_idx] = losses[best_idx]
        best_alpha[test_idx] = float(best_params["alpha"])
        Y_forecast[test_idx, :] = fit_predict_glm(X_train, Y_train, X_test, best_params)[0, :]

        done += 1
        if (done % 10 == 0) or (done == total_oos):
            pct = 100.0 * done / total_oos
            r2_now = np.array([r2_oos(Y[:, k], Y_forecast[:, k]) for k in range(M)])
            print(
                f"[{model_name}] "
                f"{done:4d}/{total_oos} "
                f"({pct:5.1f}%)  "
                f"date={dates[test_idx].strftime('%Y-%m-%d')}  "
                f"alpha={best_alpha[test_idx]:g}  "
                f"val={val_loss[test_idx]:.6f}"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    return {
        "Y_zero_benchmark": np.zeros_like(Y),
        f"Y_forecast_agg_{model_name}": Y_forecast,
        f"MSE_{model_name}": np.nanmean((Y - Y_forecast) ** 2, axis=0),
        f"R2OOS_{model_name}": np.array([r2_oos(Y[:, k], Y_forecast[:, k]) for k in range(M)]),
        f"R2OOS_pval_{model_name}": np.array(
            [r2_oos_pvalue(Y[:, k], Y_forecast[:, k], hac_lags=cfg["horizon"]) for k in range(M)]
        ),
        f"ValLoss_{model_name}": val_loss,
        f"BestAlpha_{model_name}": best_alpha,
        f"AlphaGrid_{model_name}": alpha_grid,
        f"AlphaLossPath_{model_name}": alpha_loss_path,
    }


def run_experiment(custom_config=None):
    cfg = copy.deepcopy(custom_config if custom_config is not None else CONFIG)

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

    model_name = cfg["params"]["model_name"]

    save_dict = {
        "Note": (
            "Expanding-window OOS penalized linear model with horizon-consistent embargo. "
            "At each OOS date, alpha is selected by purged validation."
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
            }
        ),
        "Horizon": cfg["horizon"],
        "Y_True": Y,
        "Dates": np.array(dates.strftime("%Y-%m-%d"), dtype=object),
        "X_Columns": np.array(X_df.columns, dtype=object),
        "Y_Columns": np.array(Y_df.columns, dtype=object),
    }

    save_dict.update(run_oos_forecast(X, Y, dates, cfg))

    os.makedirs("results", exist_ok=True)
    out_mat = os.path.join("results", _glm_result_name(cfg) + ".mat")
    sio.savemat(out_mat, save_dict)

    print("Saved:", out_mat)
    print("R2OOS:", save_dict[f"R2OOS_{model_name}"])

    return save_dict, out_mat


if __name__ == "__main__":
    run_experiment(CONFIG)