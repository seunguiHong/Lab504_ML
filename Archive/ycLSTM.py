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
    load_dataset,
    enumerate_oos_forecast_indices,
    compute_training_end_index,
    prepare_validation_matrices,
    prepare_final_training_matrices,
    build_dropout_l1l2_candidates,
    top_validation_seed_mean,
    summarize_oos_metrics,
    build_save_dict,
    save_results_mat,
)


CONFIG = {
    "data_path": "data/target_and_features.mat",
    "feature_groups": ["d12m_fwd"],
    "target_group": "dy",
    "target_indices": None,

    "horizon": 12,
    "oos_start": "1989-01-31",
    "hyper_freq": 60,

    "nmc": 10,
    "navg": 3,
    "run_tag": "lstm_d12m_fwd",
    "out_file": "results/ycLSTM_d12m_fwd.mat",

    "params": {
        "seq_len": 12,
        "lstm_units": [8],
        "dense_archi": [],
        "Dropout": [0.0],
        "l1l2": [0.0],
        "learning_rate": 0.02,
        "momentum": 0.9,
        "nesterov": True,
        "clipnorm": 1.0,
        "recurrent_dropout": 0.0,
        "standardize_x": True,
        "epochs": 500,
        "patience": 20,
        "batch_size": 32,
        "validation_split": 0.15,
        "purge_size": 12,
        "shuffle": False,
        "loss_name": "mse",
        "huber_delta": 1.0,
    },
}


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def make_optimizer(params):
    return keras.optimizers.SGD(
        learning_rate=float(params["learning_rate"]),
        momentum=float(params.get("momentum", 0.0)),
        nesterov=bool(params.get("nesterov", False)),
        clipnorm=float(params.get("clipnorm", 1.0)),
    )


def make_loss(params):
    loss_name = str(params.get("loss_name", "mse")).lower()

    if loss_name == "mse":
        return "mse"

    if loss_name == "huber":
        return keras.losses.Huber(delta=float(params.get("huber_delta", 1.0)))

    raise ValueError(f"Unknown loss_name: {loss_name}")


def make_l1l2(params):
    arr = np.asarray(params.get("l1l2", [0.0, 0.0]), dtype=float).ravel()

    if arr.size == 0:
        return regularizers.L1L2(l1=0.0, l2=0.0)

    if arr.size == 1:
        return regularizers.L1L2(l1=float(arr[0]), l2=float(arr[0]))

    return regularizers.L1L2(l1=float(arr[0]), l2=float(arr[1]))


def build_lstm_model(input_shape, output_dim, params):
    reg = make_l1l2(params)

    dropout = float(np.asarray(params.get("Dropout", 0.0), dtype=float).ravel()[0])
    recurrent_dropout = float(params.get("recurrent_dropout", 0.0))
    lstm_units = [int(x) for x in np.asarray(params.get("lstm_units", [8])).ravel()]
    dense_archi = [int(x) for x in np.asarray(params.get("dense_archi", [])).ravel()]

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i, units in enumerate(lstm_units):
        model.add(
            layers.LSTM(
                units=units,
                return_sequences=(i < len(lstm_units) - 1),
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
                bias_regularizer=reg,
            )
        )

    for units in dense_archi:
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=reg,
                bias_regularizer=reg,
            )
        )

        if dropout > 0.0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(int(output_dim), activation="linear"))
    model.compile(optimizer=make_optimizer(params), loss=make_loss(params))

    return model


def build_lstm_train_data(X, Y, forecast_idx, horizon, seq_len):
    train_end = compute_training_end_index(forecast_idx, horizon)

    if train_end < seq_len - 1:
        return None, None

    X_seq = []
    Y_seq = []

    for t in range(seq_len - 1, train_end + 1):
        x_t = X[t - seq_len + 1 : t + 1]
        y_t = Y[t]

        if np.all(np.isfinite(x_t)) and np.all(np.isfinite(y_t)):
            X_seq.append(x_t)
            Y_seq.append(y_t)

    if len(X_seq) == 0:
        return None, None

    return np.stack(X_seq, axis=0), np.stack(Y_seq, axis=0)


def build_lstm_test_data(X, forecast_idx, seq_len):
    if forecast_idx < seq_len - 1:
        return None

    X_test = X[forecast_idx - seq_len + 1 : forecast_idx + 1]

    if not np.all(np.isfinite(X_test)):
        return None

    return X_test[None, :, :]


def validate_seed(X_train, Y_train, params, seed):
    X_fit, Y_fit, X_val, Y_val, _ = prepare_validation_matrices(
        X_train=X_train,
        Y_train=Y_train,
        validation_fraction=float(params["validation_split"]),
        purge_size=int(params["purge_size"]),
        standardize_features=bool(params.get("standardize_x", True)),
    )

    set_seed(seed)
    keras.backend.clear_session()

    model = build_lstm_model(
        input_shape=(X_fit.shape[1], X_fit.shape[2]),
        output_dim=Y_fit.shape[1],
        params=params,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(params["patience"]),
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_fit,
        Y_fit,
        validation_data=(X_val, Y_val),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        shuffle=bool(params["shuffle"]),
        callbacks=[early_stop],
        verbose=0,
    )

    val_loss = float(np.min(history.history["val_loss"]))
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    return val_loss, best_epoch


def evaluate_params(X_train, Y_train, params, nmc):
    val_loss = np.full(int(nmc), np.nan)
    best_epoch = np.full(int(nmc), np.nan)

    for seed in range(int(nmc)):
        val_loss[seed], best_epoch[seed] = validate_seed(
            X_train=X_train,
            Y_train=Y_train,
            params=params,
            seed=1234 + seed,
        )

    return {
        "val_loss": val_loss,
        "best_epoch": best_epoch.astype(int),
        "score": float(np.nanmean(val_loss)),
    }


def select_params(X_train, Y_train, cfg):
    best_params = None
    best_eval = None
    best_score = np.inf

    for params in build_dropout_l1l2_candidates(cfg["params"]):
        eval_result = evaluate_params(
            X_train=X_train,
            Y_train=Y_train,
            params=params,
            nmc=int(cfg["nmc"]),
        )

        if eval_result["score"] < best_score:
            best_params = params
            best_eval = eval_result
            best_score = eval_result["score"]

    return best_params, best_eval


def fit_seed_forecast(X_train, Y_train, X_test, params, epochs, seed):
    X_train, Y_train, X_test, _ = prepare_final_training_matrices(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        standardize_features=bool(params.get("standardize_x", True)),
    )

    set_seed(seed)
    keras.backend.clear_session()

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_dim=Y_train.shape[1],
        params=params,
    )

    model.fit(
        X_train,
        Y_train,
        epochs=max(1, int(epochs)),
        batch_size=int(params["batch_size"]),
        shuffle=bool(params["shuffle"]),
        verbose=0,
    )

    return model.predict(X_test, verbose=0).reshape(-1)


def forecast_seeds(X_train, Y_train, X_test, params, eval_result, nmc):
    forecasts = []

    for seed in range(int(nmc)):
        forecasts.append(
            fit_seed_forecast(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                params=params,
                epochs=int(eval_result["best_epoch"][seed]),
                seed=1234 + seed,
            )
        )

    return np.vstack(forecasts)


def run_oos_forecast(X, Y, dates, cfg):
    dates = pd.DatetimeIndex(dates)
    oos_indices = enumerate_oos_forecast_indices(dates, cfg["oos_start"])

    T, M = Y.shape
    nmc = int(cfg["nmc"])
    navg = int(cfg["navg"])
    seq_len = int(cfg["params"]["seq_len"])
    horizon = int(cfg["horizon"])
    hyper_freq = int(cfg["hyper_freq"])

    if navg > nmc:
        raise ValueError("navg cannot exceed nmc.")

    Y_forecast = np.full((T, M), np.nan)
    Y_forecast_all = np.full((T, nmc, M), np.nan)
    val_loss = np.full((T, nmc), np.nan)

    current_params = None
    current_eval = None
    oos_count = 0
    total_oos = len(oos_indices)

    print(f"Total OOS steps: {total_oos}")

    for step, forecast_idx in enumerate(oos_indices, start=1):
        X_train, Y_train = build_lstm_train_data(
            X=X,
            Y=Y,
            forecast_idx=forecast_idx,
            horizon=horizon,
            seq_len=seq_len,
        )

        X_test = build_lstm_test_data(
            X=X,
            forecast_idx=forecast_idx,
            seq_len=seq_len,
        )

        if X_train is None or X_test is None:
            continue

        oos_count += 1
        retune = (oos_count == 1) or ((oos_count - 1) % hyper_freq == 0)

        if retune:
            current_params, current_eval = select_params(
                X_train=X_train,
                Y_train=Y_train,
                cfg=cfg,
            )
        else:
            current_eval = evaluate_params(
                X_train=X_train,
                Y_train=Y_train,
                params=current_params,
                nmc=nmc,
            )

        seed_forecasts = forecast_seeds(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            params=current_params,
            eval_result=current_eval,
            nmc=nmc,
        )

        Y_forecast_all[forecast_idx] = seed_forecasts
        Y_forecast[forecast_idx] = top_validation_seed_mean(
            seed_forecasts,
            current_eval["val_loss"],
            navg,
        )
        val_loss[forecast_idx] = current_eval["val_loss"]

        if step == 1 or step % 12 == 0 or step == total_oos:
            r2_now = summarize_oos_metrics(
                Y_true=Y,
                Y_pred=Y_forecast,
                hac_lags=horizon,
            )[1]

            print(
                f"[{step:4d}/{total_oos}] "
                f"date={dates[forecast_idx].strftime('%Y-%m-%d')} | "
                f"retune={retune} | "
                f"val={current_eval['score']:10.6f} | "
                f"params={json.dumps(current_params)}"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    mse, r2, pval = summarize_oos_metrics(
        Y_true=Y,
        Y_pred=Y_forecast,
        hac_lags=horizon,
    )

    return {
        "Y_Forecast": Y_forecast,
        "Y_Forecast_All": Y_forecast_all,
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

    result = run_oos_forecast(X, Y, dates, cfg)

    save_dict = build_save_dict(cfg, X_df, Y_df, Y)
    save_dict.update(result)
    save_results_mat(cfg["out_file"], save_dict)

    print("Saved to", cfg["out_file"])
    print("R2OOS:", np.round(save_dict["R2OOS"], 4))

    return save_dict, cfg["out_file"]


if __name__ == "__main__":
    run_experiment(CONFIG)