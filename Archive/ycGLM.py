#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso

from utils import (
    load_dataset,
    enumerate_oos_forecast_indices,
    compute_training_end_index,
    prepare_validation_matrices,
    prepare_final_training_matrices,
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
    "run_tag": "ridge_d12m_fwdchg",
    "out_file": "results/tab1_rx/panelA_rxglm(Ridge)_fwd.mat",

    "params": {
        "estimator": "Ridge",
        "alpha_grid": [1e-2, 1.0, 1e2],
        "standardize": True,
        "max_iter": 10000,
        "validation_split": 0.15,
        "purge_size": 12,
    },
}


def build_estimator(params):
    alpha = float(params["alpha"])

    if params["estimator"] == "Ridge":
        return Ridge(alpha=alpha, fit_intercept=True)

    if params["estimator"] == "Lasso":
        return Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=int(params.get("max_iter", 10000)),
            random_state=0,
        )

    raise ValueError(f"Unknown estimator: {params['estimator']}")


def fit_predict(X_train, Y_train, X_test, params):
    Y_train = np.asarray(Y_train, dtype=float)
    if Y_train.ndim == 1:
        Y_train = Y_train.reshape(-1, 1)

    Y_hat = np.full((X_test.shape[0], Y_train.shape[1]), np.nan)

    for j in range(Y_train.shape[1]):
        model = build_estimator(params)
        model.fit(X_train, Y_train[:, j])
        Y_hat[:, j] = model.predict(X_test)

    return Y_hat


def alpha_candidates(params):
    out = []

    for alpha in params["alpha_grid"]:
        candidate = copy.deepcopy(params)
        candidate["alpha"] = float(alpha)
        out.append(candidate)

    return out


def validation_loss(X_train, Y_train, params):
    X_fit, Y_fit, X_val, Y_val, _ = prepare_validation_matrices(
        X_train=X_train,
        Y_train=Y_train,
        validation_fraction=float(params["validation_split"]),
        purge_size=int(params["purge_size"]),
        standardize_features=bool(params["standardize"]),
    )

    Y_val_hat = fit_predict(X_fit, Y_fit, X_val, params)

    return float(np.mean((Y_val - Y_val_hat) ** 2))


def select_alpha(X_train, Y_train, params):
    best_params = None
    best_loss = np.inf

    for candidate in alpha_candidates(params):
        loss = validation_loss(X_train, Y_train, candidate)

        if loss < best_loss:
            best_loss = loss
            best_params = candidate

    return best_params, best_loss


def final_forecast(X_train, Y_train, X_test, params):
    X_train, Y_train, X_test, _ = prepare_final_training_matrices(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        standardize_features=bool(params["standardize"]),
    )

    return fit_predict(X_train, Y_train, X_test, params)[0]


def run_oos_forecast(X, Y, dates, cfg):
    dates = pd.DatetimeIndex(dates)
    oos_indices = enumerate_oos_forecast_indices(dates, cfg["oos_start"])

    T, M = Y.shape

    Y_forecast = np.full((T, M), np.nan)
    val_loss = np.full(T, np.nan)

    print(f"Total OOS steps: {len(oos_indices)}")

    for step, forecast_idx in enumerate(oos_indices, start=1):
        train_end = compute_training_end_index(forecast_idx, cfg["horizon"])

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

        if X_train.shape[0] < 30:
            continue

        best_params, loss = select_alpha(X_train, Y_train, cfg["params"])

        Y_forecast[forecast_idx] = final_forecast(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            params=best_params,
        )

        val_loss[forecast_idx] = loss

        if step == 1 or step % 12 == 0 or step == len(oos_indices):
            r2_now = summarize_oos_metrics(
                Y_true=Y,
                Y_pred=Y_forecast,
                hac_lags=int(cfg["horizon"]),
            )[1]

            print(
                f"[{step:4d}/{len(oos_indices)}] "
                f"date={dates[forecast_idx].strftime('%Y-%m-%d')} | "
                f"alpha={best_params['alpha']:g} | "
                f"val={loss:10.6f}"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    mse, r2, pval = summarize_oos_metrics(
        Y_true=Y,
        Y_pred=Y_forecast,
        hac_lags=int(cfg["horizon"]),
    )

    return {
        "Y_Forecast": Y_forecast,
        "ValLoss": val_loss,
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

    result = run_oos_forecast(
        X=X,
        Y=Y,
        dates=dates,
        cfg=cfg,
    )

    save_dict = build_save_dict(cfg, X_df, Y_df, Y)
    save_dict.update(result)

    save_results_mat(cfg["out_file"], save_dict)

    print("Saved to", cfg["out_file"])
    print("R2OOS:", np.round(save_dict["R2OOS"], 4))

    return save_dict, cfg["out_file"]


if __name__ == "__main__":
    run_experiment(CONFIG)