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
    prepare_validation_matrices,
    prepare_final_training_matrices,
    summarize_oos_metrics,
    build_save_dict,
    result_name,
    save_results_mat,
)


BASE_CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["d12m_y_pc2"],
    "target_group": "dy",
    "target_indices": None,
    "horizon": 12,
    "oos_start": "1989-01-31",
    "hyper_freq": 60,
    "nmc": 10,
    "navg": 3,
    "run_tag": "LSTM_fwd_ensembled",
    "model_name": "LSTMModel",
    "results_dir": "results",
    "params": {
        "seq_len": 12,
        "lstm_units": [8],
        "dense_archi": [],
        "Dropout": [0.0],
        "l1l2": [0.00],
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


def _set_all_seeds(seed: int) -> None:
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


def _make_optimizer(params: dict):
    return keras.optimizers.SGD(
        learning_rate=float(params["learning_rate"]),
        momentum=float(params.get("momentum", 0.0)),
        nesterov=bool(params.get("nesterov", False)),
        clipnorm=float(params.get("clipnorm", 1.0)),
    )


def _make_loss(params: dict):
    loss_name = str(params.get("loss_name", "mse")).lower()
    if loss_name == "mse":
        return "mse"
    if loss_name == "huber":
        return keras.losses.Huber(delta=float(params.get("huber_delta", 1.0)))
    raise ValueError(f"Unknown loss_name: {loss_name}")


def _build_lstm_network(input_shape, output_dim: int, params: dict):
    l1_val, l2_val = _normalize_l1l2(params.get("l1l2", [0.0, 0.0]))
    reg = regularizers.L1L2(l1=l1_val, l2=l2_val)

    dropout_rate = float(np.asarray(params.get("Dropout", 0.0), dtype=float).ravel()[0])
    recurrent_dropout = float(params.get("recurrent_dropout", 0.0))
    lstm_units = [int(v) for v in np.asarray(params.get("lstm_units", [8]), dtype=int).ravel()]
    dense_archi = [int(v) for v in np.asarray(params.get("dense_archi", []), dtype=int).ravel()]

    model = keras.Sequential(name="LSTMModel")
    model.add(layers.Input(shape=input_shape))

    for idx, units in enumerate(lstm_units):
        model.add(
            layers.LSTM(
                units=int(units),
                return_sequences=(idx < len(lstm_units) - 1),
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
                bias_regularizer=reg,
            )
        )

    for units in dense_archi:
        model.add(
            layers.Dense(
                int(units),
                activation="relu",
                kernel_regularizer=reg,
                bias_regularizer=reg,
            )
        )
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(int(output_dim), activation="linear"))
    model.compile(optimizer=_make_optimizer(params), loss=_make_loss(params))
    return model


def _build_sequence_training_tensors(X, Y, forecast_index: int, horizon: int, seq_len: int):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    train_end = int(forecast_index - horizon)
    if train_end < seq_len - 1:
        return None, None

    X_list = []
    Y_list = []

    for t in range(seq_len - 1, train_end + 1):
        x_seq = X[t - seq_len + 1: t + 1, :]
        y_t = Y[t, :]
        if np.all(np.isfinite(x_seq)) and np.all(np.isfinite(y_t)):
            X_list.append(x_seq)
            Y_list.append(y_t)

    if len(X_list) == 0:
        return None, None

    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)


def _build_sequence_test_tensor(X, forecast_index: int, seq_len: int):
    X = np.asarray(X, dtype=float)

    if forecast_index < seq_len - 1:
        return None

    x_seq = X[forecast_index - seq_len + 1: forecast_index + 1, :]
    if not np.all(np.isfinite(x_seq)):
        return None

    return x_seq[None, :, :]


def _fit_validation_seed(X_train, Y_train, params: dict, seed: int):
    X_fit, Y_fit, X_val, Y_val, _ = prepare_validation_matrices(
        X_train=X_train,
        Y_train=Y_train,
        validation_fraction=float(params["validation_split"]),
        purge_size=int(params.get("purge_size", 12)),
        standardize_features=bool(params.get("standardize_x", True)),
    )

    _set_all_seeds(seed)
    keras.backend.clear_session()

    model = _build_lstm_network(
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
        shuffle=bool(params.get("shuffle", False)),
        verbose=0,
        callbacks=[early_stop],
    )

    val_pred = model.predict(X_val, verbose=0)
    val_loss = float(np.mean((Y_val - val_pred) ** 2))
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    return {
        "val_loss": val_loss,
        "best_epoch": best_epoch,
    }


def _evaluate_candidate(X_train, Y_train, params: dict, nmc: int):
    seed_results = []
    for k in range(int(nmc)):
        res_k = _fit_validation_seed(
            X_train=X_train,
            Y_train=Y_train,
            params=params,
            seed=1234 + k,
        )
        seed_results.append(res_k)

    val_losses = np.array([r["val_loss"] for r in seed_results], dtype=float)
    best_epochs = np.array([r["best_epoch"] for r in seed_results], dtype=int)

    return {
        "mean_val_loss": float(np.nanmean(val_losses)),
        "val_losses": val_losses,
        "best_epochs": best_epochs,
    }


def _select_best_candidate(X_train, Y_train, cfg: dict):
    candidates = _build_inner_candidates(cfg["params"])

    best_params = None
    best_eval = None
    best_score = np.inf

    for cand in candidates:
        cand_eval = _evaluate_candidate(
            X_train=X_train,
            Y_train=Y_train,
            params=cand,
            nmc=int(cfg["nmc"]),
        )

        if cand_eval["mean_val_loss"] < best_score:
            best_score = cand_eval["mean_val_loss"]
            best_params = copy.deepcopy(cand)
            best_eval = cand_eval

    return best_params, best_eval


def _fit_final_seed(X_train, Y_train, X_test, params: dict, epochs_final: int, seed: int):
    X_train_final, Y_train_final, X_test_final, _ = prepare_final_training_matrices(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        standardize_features=bool(params.get("standardize_x", True)),
    )

    _set_all_seeds(seed)
    keras.backend.clear_session()

    model = _build_lstm_network(
        input_shape=(X_train_final.shape[1], X_train_final.shape[2]),
        output_dim=Y_train_final.shape[1],
        params=params,
    )

    model.fit(
        X_train_final,
        Y_train_final,
        epochs=max(1, int(epochs_final)),
        batch_size=int(params["batch_size"]),
        shuffle=bool(params.get("shuffle", False)),
        verbose=0,
    )

    pred = model.predict(X_test_final, verbose=0)
    return np.asarray(pred, dtype=float).reshape(-1)


def _fit_final_multi_seed(X_train, Y_train, X_test, params: dict, best_eval: dict, nmc: int):
    outputs = {}
    for k in range(int(nmc)):
        epochs_k = int(best_eval["best_epochs"][k])
        pred_k = _fit_final_seed(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            params=params,
            epochs_final=epochs_k,
            seed=1234 + k,
        )
        outputs[k] = (pred_k, float(best_eval["val_losses"][k]))
    return outputs


def run_oos_forecast(X, Y, dates, cfg: dict):
    model_name = cfg["model_name"]
    seq_len = int(cfg["params"]["seq_len"])
    horizon = int(cfg["horizon"])
    nmc = int(cfg["nmc"])
    navg = int(cfg["navg"])
    hyper_freq = int(cfg["hyper_freq"])

    dates = pd.DatetimeIndex(dates)
    start_candidates = np.where(dates >= pd.Timestamp(cfg["oos_start"]))[0]
    if start_candidates.size == 0:
        raise ValueError("No available sample date on or after oos_start.")
    oos_indices = list(range(int(start_candidates[0]), len(dates)))

    T, M = Y.shape
    Y_forecast_all = np.full((T, nmc, M), np.nan)
    Y_forecast_avg = np.full((T, M), np.nan)
    val_loss = np.full((T, nmc), np.nan)

    best_dropout_path = np.full(T, np.nan)
    best_l1_path = np.full(T, np.nan)
    best_l2_path = np.full(T, np.nan)
    best_epoch_mean_path = np.full(T, np.nan)

    current_best_params = None
    current_best_eval = None

    print(model_name)

    oos_counter = 0
    total_oos = len(oos_indices)

    for j, forecast_index in enumerate(oos_indices, start=1):
        X_train, Y_train = _build_sequence_training_tensors(
            X=X,
            Y=Y,
            forecast_index=forecast_index,
            horizon=horizon,
            seq_len=seq_len,
        )
        X_test = _build_sequence_test_tensor(
            X=X,
            forecast_index=forecast_index,
            seq_len=seq_len,
        )

        if X_train is None or Y_train is None or X_test is None:
            continue

        oos_counter += 1
        retune = (oos_counter == 1) or ((oos_counter - 1) % hyper_freq == 0)

        if retune:
            current_best_params, current_best_eval = _select_best_candidate(
                X_train=X_train,
                Y_train=Y_train,
                cfg=cfg,
            )

        outputs = _fit_final_multi_seed(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            params=current_best_params,
            best_eval=current_best_eval,
            nmc=nmc,
        )

        val_loss[forecast_index, :] = np.array([outputs[k][1] for k in range(nmc)], dtype=float)
        Y_forecast_all[forecast_index, :, :] = np.vstack([outputs[k][0] for k in range(nmc)])

        best_seed_order = np.argsort(val_loss[forecast_index, :])
        Y_forecast_avg[forecast_index, :] = np.mean(
            Y_forecast_all[forecast_index, best_seed_order[:navg], :],
            axis=0,
        )

        reg_arr = np.asarray(current_best_params["l1l2"], dtype=float).ravel()
        best_dropout_path[forecast_index] = float(current_best_params["Dropout"])
        best_l1_path[forecast_index] = float(reg_arr[0])
        best_l2_path[forecast_index] = float(reg_arr[1] if reg_arr.size >= 2 else reg_arr[0])
        best_epoch_mean_path[forecast_index] = float(np.mean(current_best_eval["best_epochs"]))

        current_best_val = float(np.nanmean(current_best_eval["val_losses"]))

        if (j == 1) or (j % 12 == 0) or (j == total_oos):
            r2_now = np.array(
                [summarize_oos_metrics(Y[:, [m]], Y_forecast_avg[:, [m]])[1][0] for m in range(M)]
            )
            print(
                f"[{j:4d}/{total_oos}] "
                f"date={dates[forecast_index].strftime('%Y-%m-%d')} | "
                f"retune={retune} | "
                f"val={current_best_val:10.6f} | "
                f"seq={current_best_params['seq_len']} | "
                f"lstm={current_best_params['lstm_units']} | "
                f"dense={current_best_params['dense_archi']} | "
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
        f"BestEpochMean_{model_name}": best_epoch_mean_path,
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