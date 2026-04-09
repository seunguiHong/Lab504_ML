#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import copy
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from utils import (
    get_cpu_count,
    load_dataset,
    enumerate_oos_forecast_indices,
    prepare_validation_matrices,
    prepare_final_training_matrices,
    summarize_oos_metrics,
    build_save_dict,
    result_name,
    save_results_mat,
)

BASE_CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["dy_pc1", "dy_pc2"],
    "target_group": "dy",
    "target_indices": None,
    "horizon": 12,
    "oos_start": "1989-01-31",
    "hyper_freq": 60,
    "nmc": 20,
    "navg": 10,
    "run_tag": "pc12",
    "model_name": "NNModel",
    "results_dir": "results",
    "params": {
        "archi": [3, 3],
        "Dropout": [0.0],
        "l1l2": [1e-4, 5e-5],
        "learning_rate": 0.03,
        "momentum": 0.9,
        "nesterov": True,
        "epochs": 500,
        "patience": 20,
        "batch_size": 32,
        "validation_split": 0.15,
        "purge_size": 12,
        "shuffle": False,
        "standardize_x": True,
        "loss_name": "mse",
        "huber_delta": 1.0,
    },
}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def _normalize_l1l2(x):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return [0.0, 0.0]
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    return [float(arr[0]), float(arr[1])]


def _build_inner_candidates(params: dict):
    dropout_raw = params["Dropout"]
    l1l2_raw = params["l1l2"]

    if np.isscalar(dropout_raw):
        dropout_candidates = [float(dropout_raw)]
    else:
        dropout_candidates = [float(v) for v in np.asarray(dropout_raw, dtype=float).ravel()]

    l1l2_arr = np.asarray(l1l2_raw, dtype=float)
    if l1l2_arr.ndim == 1:
        l1l2_candidates = [_normalize_l1l2(l1l2_arr)]
    else:
        l1l2_candidates = [_normalize_l1l2(row) for row in l1l2_arr]

    candidates = []
    for do in dropout_candidates:
        for reg in l1l2_candidates:
            cand = copy.deepcopy(params)
            cand["Dropout"] = float(do)
            cand["l1l2"] = reg
            candidates.append(cand)

    return candidates


def _make_loss(params: dict):
    loss_name = str(params.get("loss_name", "mse")).lower()
    if loss_name == "mse":
        return "mse"
    if loss_name == "huber":
        return keras.losses.Huber(delta=float(params.get("huber_delta", 1.0)))
    raise ValueError(f"Unknown loss_name: {loss_name}")


def _make_optimizer(params: dict):
    return keras.optimizers.SGD(
        learning_rate=float(params["learning_rate"]),
        momentum=float(params.get("momentum", 0.0)),
        nesterov=bool(params.get("nesterov", False)),
    )


def _build_network(input_dim: int, output_dim: int, params: dict):
    l1_val, l2_val = _normalize_l1l2(params.get("l1l2", [0.0, 0.0]))
    reg = regularizers.L1L2(l1=l1_val, l2=l2_val)

    archi = [int(v) for v in np.asarray(params.get("archi", []), dtype=int).ravel()]
    dropout_rate = float(params.get("Dropout", 0.0))

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in archi:
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=reg, bias_regularizer=reg))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(output_dim, activation="linear"))
    model.compile(optimizer=_make_optimizer(params), loss=_make_loss(params))
    return model


def _fit_validation_model(X_train, Y_train, params: dict, seed: int):
    X_fit, Y_fit, X_val, Y_val, _ = prepare_validation_matrices(
        X_train=X_train,
        Y_train=Y_train,
        validation_fraction=float(params["validation_split"]),
        purge_size=int(params.get("purge_size", 12)),
        standardize_features=bool(params.get("standardize_x", True)),
    )

    _set_seed(seed)
    keras.backend.clear_session()

    model = _build_network(
        input_dim=X_fit.shape[1],
        output_dim=Y_fit.shape[1],
        params=params,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(params["patience"]),
            restore_best_weights=True,
            verbose=0,
        )
    ]

    model.fit(
        X_fit,
        Y_fit,
        validation_data=(X_val, Y_val),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        shuffle=bool(params.get("shuffle", False)),
        verbose=0,
        callbacks=callbacks,
    )

    Y_val_hat = model.predict(X_val, verbose=0)
    val_loss = float(np.mean((Y_val - Y_val_hat) ** 2))
    return val_loss


def _fit_final_model(X_train, Y_train, X_test, params: dict, seed: int):
    X_train_final, Y_train_final, X_test_final, _ = prepare_final_training_matrices(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        standardize_features=bool(params.get("standardize_x", True)),
    )

    _set_seed(seed)
    keras.backend.clear_session()

    model = _build_network(
        input_dim=X_train_final.shape[1],
        output_dim=Y_train_final.shape[1],
        params=params,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=int(params["patience"]),
            restore_best_weights=True,
            verbose=0,
        )
    ]

    model.fit(
        X_train_final,
        Y_train_final,
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        shuffle=bool(params.get("shuffle", False)),
        verbose=0,
        callbacks=callbacks,
    )

    Y_hat = model.predict(X_test_final, verbose=0)
    return Y_hat


def _select_hyperparameters(X_train, Y_train, params: dict, nmc: int):
    candidates = _build_inner_candidates(params)

    best_params = None
    best_score = np.inf

    for cand in candidates:
        losses = []
        for k in range(int(nmc)):
            loss_k = _fit_validation_model(
                X_train=X_train,
                Y_train=Y_train,
                params=cand,
                seed=1234 + k,
            )
            losses.append(loss_k)

        score = float(np.nanmean(losses))
        if score < best_score:
            best_score = score
            best_params = copy.deepcopy(cand)

    return best_params, best_score


def run_oos_forecast(X, Y, dates, cfg: dict):
    model_name = cfg["model_name"]
    horizon = int(cfg["horizon"])
    nmc = int(cfg["nmc"])
    navg = int(cfg["navg"])
    hyper_freq = int(cfg["hyper_freq"])

    oos_indices = enumerate_oos_forecast_indices(dates, cfg["oos_start"])

    T, M = Y.shape
    Y_forecast_all = np.full((T, nmc, M), np.nan)
    Y_forecast_avg = np.full((T, M), np.nan)
    val_loss = np.full((T, nmc), np.nan)

    best_dropout_path = np.full(T, np.nan)
    best_l1_path = np.full(T, np.nan)
    best_l2_path = np.full(T, np.nan)

    current_best_params = None
    current_best_validation_score = np.nan

    total_oos = len(oos_indices)
    oos_counter = 0

    print(model_name)

    for j, forecast_idx in enumerate(oos_indices, start=1):
        train_end = forecast_idx - horizon
        if train_end < 1:
            continue

        X_train = X[: train_end + 1, :]
        Y_train = Y[: train_end + 1, :]
        X_test = X[forecast_idx : forecast_idx + 1, :]

        if not np.all(np.isfinite(X_test)):
            continue

        train_valid = np.all(np.isfinite(X_train), axis=1) & np.all(np.isfinite(Y_train), axis=1)
        X_train = X_train[train_valid]
        Y_train = Y_train[train_valid]

        if X_train.shape[0] < 10:
            continue

        oos_counter += 1
        retune = (oos_counter == 1) or ((oos_counter - 1) % hyper_freq == 0)

        if retune:
            current_best_params, current_best_validation_score = _select_hyperparameters(
                X_train=X_train,
                Y_train=Y_train,
                params=cfg["params"],
                nmc=nmc,
            )

        for k in range(nmc):
            pred_k = _fit_final_model(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                params=current_best_params,
                seed=1234 + k,
            )
            Y_forecast_all[forecast_idx, k, :] = np.asarray(pred_k, dtype=float).reshape(-1)
            val_loss[forecast_idx, k] = float(current_best_validation_score)

        best_seed_order = np.argsort(val_loss[forecast_idx, :])
        Y_forecast_avg[forecast_idx, :] = np.mean(
            Y_forecast_all[forecast_idx, best_seed_order[:navg], :],
            axis=0,
        )

        best_dropout_path[forecast_idx] = float(current_best_params["Dropout"])
        best_l1_path[forecast_idx] = float(current_best_params["l1l2"][0])
        best_l2_path[forecast_idx] = float(current_best_params["l1l2"][1])

        if (j == 1) or (j % 12 == 0) or (j == total_oos):
            r2_now = np.array(
                [summarize_oos_metrics(Y[:, [m]], Y_forecast_avg[:, [m]])[1][0] for m in range(M)]
            )
            print(
                f"[{j:4d}/{total_oos}] "
                f"date={dates[forecast_idx].strftime('%Y-%m-%d')} | "
                f"retune={retune} | "
                f"val={current_best_validation_score:10.6f} | "
                f"arch={current_best_params['archi']} | "
                f"do={current_best_params['Dropout']} | "
                f"l1l2={current_best_params['l1l2']} | "
                f"lr={current_best_params['learning_rate']}"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    mse_vec, r2_vec, pval_vec = summarize_oos_metrics(Y, Y_forecast_avg)

    return {
        f"ValLoss_{model_name}": val_loss,
        f"Y_forecast_{model_name}": Y_forecast_all,
        f"Y_forecast_agg_{model_name}": Y_forecast_avg,
        f"MSE_{model_name}": mse_vec,
        f"R2OOS_{model_name}": r2_vec,
        f"R2OOS_pval_{model_name}": pval_vec,
        f"BestDropout_{model_name}": best_dropout_path,
        f"BestL1_{model_name}": best_l1_path,
        f"BestL2_{model_name}": best_l2_path,
    }


def run_experiment(custom_config=None):
    cfg = copy.deepcopy(custom_config if custom_config is not None else BASE_CONFIG)

    ncpus = get_cpu_count()
    print(f"CPU count: {ncpus}")

    X_df, Y_df = load_dataset(
        mat_path=cfg["mat_path"],
        feature_groups=cfg["feature_groups"],
        target_group=cfg["target_group"],
        target_indices=cfg.get("target_indices", None),
    )

    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(X_df.index)

    save_dict = build_save_dict(cfg, X_df, Y_df, Y)
    save_dict.update(run_oos_forecast(X, Y, dates, cfg))

    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    out_mat = os.path.join(results_dir, result_name(cfg) + ".mat")
    save_results_mat(out_mat, save_dict)

    print("Saved to", out_mat)
    print("R2OOS:", np.round(save_dict[f"R2OOS_{cfg['model_name']}"], 4))

    return save_dict, out_mat, X_df.columns.tolist(), Y_df.columns.tolist()


if __name__ == "__main__":
    run_experiment(BASE_CONFIG)