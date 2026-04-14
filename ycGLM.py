#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso

from utils import (
    load_dataset,
    prepare_validation_matrices,
    prepare_final_training_matrices,
    summarize_oos_metrics,
    build_save_dict,
    save_results_mat,
)

CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["d12m_fwd"],
    "target_group": "dy",
    "target_indices": None,
    "horizon": 12,
    "oos_start": "1989-01-31",
    "run_tag": "RIDGERDIGERIDGE",
    "model_name": "GLM",
    "results_dir": "results",
    "params": {
        "model_name": "Ridge",          # "Ridge" or "Lasso"
        "alpha_grid": [1e-2, 1.0, 1e2],
        "standardize": True,
        "max_iter": 10000,
        "validation_split": 0.15,
        "purge_size": 12,
    },
}


def _glm_result_name(cfg):
    p = cfg["params"]
    target = str(cfg["target_group"])
    feat = str(cfg["run_tag"])
    model = str(p["model_name"])
    horizon = f"h{int(cfg['horizon'])}"
    std = "std1" if bool(p["standardize"]) else "std0"
    return "__".join([target, feat, model, horizon, std, "alpha_search"])


def _build_estimator(params):
    model_name = str(params["model_name"])
    alpha = float(params["alpha"])
    max_iter = int(params.get("max_iter", 10000))

    if model_name == "Ridge":
        return Ridge(alpha=alpha, fit_intercept=True)

    if model_name == "Lasso":
        return Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=max_iter,
            random_state=0,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


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


def _validation_mse(X_train, Y_train, params):
    X_fit, Y_fit, X_val, Y_val, _ = prepare_validation_matrices(
        X_train=X_train,
        Y_train=Y_train,
        validation_fraction=float(params.get("validation_split", 0.15)),
        purge_size=int(params.get("purge_size", 12)),
        standardize_features=bool(params.get("standardize", True)),
    )

    Y_val_hat = fit_predict_glm(X_fit, Y_fit, X_val, params)
    return float(np.mean((Y_val - Y_val_hat) ** 2))


def _final_forecast(X_train, Y_train, X_test, params):
    X_train_final, Y_train_final, X_test_final, _ = prepare_final_training_matrices(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        standardize_features=bool(params.get("standardize", True)),
    )

    Y_test_hat = fit_predict_glm(X_train_final, Y_train_final, X_test_final, params)
    return Y_test_hat[0, :]


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
    best_alpha = np.full(T, np.nan)
    val_loss = np.full(T, np.nan)

    alpha_candidates = _build_alpha_candidates(cfg["params"])
    alpha_grid = np.array([float(p["alpha"]) for p in alpha_candidates], dtype=float)
    alpha_loss_path = np.full((T, len(alpha_grid)), np.nan)

    total_oos = T - start_idx
    done = 0
    model_name = str(cfg["params"]["model_name"])

    print(f"Starting {model_name}: {total_oos} OOS steps")

    for test_idx in range(start_idx, T):
        train_end = test_idx - int(cfg["horizon"])
        if train_end < 1:
            continue

        X_train = X[: train_end + 1, :]
        Y_train = Y[: train_end + 1, :]
        X_test = X[test_idx : test_idx + 1, :]

        if not np.all(np.isfinite(X_test)):
            continue

        valid_train = np.all(np.isfinite(X_train), axis=1) & np.all(np.isfinite(Y_train), axis=1)
        X_train = X_train[valid_train]
        Y_train = Y_train[valid_train]

        if X_train.shape[0] < 10:
            continue

        losses = []
        for j, cand_params in enumerate(alpha_candidates):
            loss_j = _validation_mse(
                X_train=X_train,
                Y_train=Y_train,
                params=cand_params,
            )
            losses.append(loss_j)
            alpha_loss_path[test_idx, j] = loss_j

        best_idx = int(np.argmin(losses))
        best_params = alpha_candidates[best_idx]

        val_loss[test_idx] = float(losses[best_idx])
        best_alpha[test_idx] = float(best_params["alpha"])
        Y_forecast[test_idx, :] = _final_forecast(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            params=best_params,
        )

        done += 1
        if (done % 10 == 0) or (done == total_oos):
            pct = 100.0 * done / total_oos
            r2_now = summarize_oos_metrics(Y, Y_forecast, hac_lags=int(cfg["horizon"]))[1]
            print(
                f"[{model_name}] "
                f"{done:4d}/{total_oos} "
                f"({pct:5.1f}%)  "
                f"date={dates[test_idx].strftime('%Y-%m-%d')}  "
                f"alpha={best_alpha[test_idx]:g}  "
                f"val={val_loss[test_idx]:.6f}"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    mse_vec, r2_vec, pval_vec = summarize_oos_metrics(
        Y_true=Y,
        Y_pred=Y_forecast,
        hac_lags=int(cfg["horizon"]),
    )

    return {
        "Y_zero_benchmark": np.zeros_like(Y),
        f"Y_forecast_agg_{model_name}": Y_forecast,
        f"MSE_{model_name}": mse_vec,
        f"R2OOS_{model_name}": r2_vec,
        f"R2OOS_pval_{model_name}": pval_vec,
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
                "Expanding-window OOS penalized linear model with horizon-consistent embargo. "
                "At each OOS date, alpha is selected by purged validation, then the model is "
                "refit on the full training sample to generate the test forecast."
            )
        },
    )
    save_dict.update(run_oos_forecast(X, Y, dates, cfg))

    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    out_mat = os.path.join(results_dir, _glm_result_name(cfg) + ".mat")
    save_results_mat(out_mat, save_dict)

    model_name = str(cfg["params"]["model_name"])
    print("Saved:", out_mat)
    print("R2OOS:", np.round(save_dict[f"R2OOS_{model_name}"], 4))

    return save_dict, out_mat


if __name__ == "__main__":
    run_experiment(CONFIG)