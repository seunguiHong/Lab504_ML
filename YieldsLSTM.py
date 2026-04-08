#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import json

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import ParameterGrid

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from Utils import (
    load_dataset,
    summarize_oos_metrics,
    build_save_dict,
    extract_summary_rows,
    save_sweep_summary_to_excel,
)

RUN_MODE = "sweep"   # "single" or "sweep"


BASE_CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["dy_pc","macropc"],
    "target_group": "dy",
    "horizon": 12,
    "oos_start": "1989-01-31",
    "hyper_freq": 60,          # retune inner hyperparameters every 60 OOS steps
    "nmc": 1,
    "navg": 1,
    "run_tag": "pc_lstm",
    "model_name": "LSTMModel",
    "params": {
        "seq_len": 12,
        "lstm_units": [8],
        "dense_archi": [],
        "Dropout": [0.0],              # inner candidates
        "l1l2": [1e-4, 1e-5],          # inner candidates or one pair
        "learning_rate": 0.005,
        "momentum": 0.9,
        "nesterov": True,
        "clipnorm": 1.0,
        "standardize_x": True,
        "recurrent_dropout": 0.0,
        "epochs": 500,
        "patience": 20,
        "batch_size": 32,
        "validation_split": 0.15,
        "shuffle": False,
        "loss_name": "mse",
        "huber_delta": 1.0,
    },
}


SWEEP_GRID = {
    "seq_len": [12],
    "lstm_units": [[8], [16]],
    "dense_archi": [[], [8]],
    "learning_rate": [0.01, 0.003],
}


def _pair_tag(x):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return "0-0"
    if arr.size == 1:
        return f"{arr[0]:g}-{arr[0]:g}"
    return f"{arr[0]:g}-{arr[1]:g}"


def _arch_tag(x):
    arr = np.asarray(x).ravel()
    if arr.size == 0:
        return "none"
    return "x".join(str(int(v)) for v in arr)


def _safe_result_name(cfg, sweep=False):
    target = str(cfg["target_group"])
    feat = str(cfg["run_tag"])
    model = str(cfg["model_name"]).replace("Model", "")
    horizon = f"h{int(cfg['horizon'])}"

    if sweep:
        return "__".join(["sweep", target, feat, model, horizon])

    p = cfg["params"]

    seq_tag = f"seq{int(p['seq_len'])}"
    lstm_tag = f"lstm{_arch_tag(p['lstm_units'])}"
    dense_tag = f"dense{_arch_tag(p['dense_archi'])}"

    do = p.get("Dropout", [0.0])
    do_arr = np.asarray(do, dtype=float).ravel()
    do_tag = "|".join(f"{float(v):g}" for v in do_arr)

    reg = p.get("l1l2", [0.0, 0.0])
    reg_arr = np.asarray(reg, dtype=float)
    if reg_arr.ndim == 1:
        reg_tag = _pair_tag(reg_arr)
    else:
        reg_tag = "|".join(_pair_tag(row) for row in reg_arr)

    lr_tag = f"{float(p['learning_rate']):g}"

    return "__".join(
        [
            target,
            feat,
            model,
            horizon,
            seq_tag,
            lstm_tag,
            dense_tag,
            f"do{do_tag}",
            f"reg{reg_tag}",
            f"lr{lr_tag}",
        ]
    )


def _set_all_seeds(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def _make_optimizer(params):
    return keras.optimizers.SGD(
        learning_rate=float(params["learning_rate"]),
        momentum=float(params.get("momentum", 0.0)),
        nesterov=bool(params.get("nesterov", False)),
        clipnorm=float(params.get("clipnorm", 1.0)),
    )


def _make_loss(params):
    loss_name = str(params.get("loss_name", "mse")).lower()

    if loss_name == "mse":
        return "mse"
    if loss_name == "huber":
        return keras.losses.Huber(delta=float(params.get("huber_delta", 1.0)))

    raise ValueError(f"Unknown loss_name: {loss_name}")


def _build_lstm_network(input_shape, output_dim, params):
    arr = np.asarray(params.get("l1l2", [0.0, 0.0]), dtype=float).ravel()
    if arr.size == 0:
        l1_val, l2_val = 0.0, 0.0
    elif arr.size == 1:
        l1_val, l2_val = float(arr[0]), float(arr[0])
    else:
        l1_val, l2_val = float(arr[0]), float(arr[1])

    reg = regularizers.L1L2(l1=l1_val, l2=l2_val)

    dropout = params.get("Dropout", 0.0)
    if np.isscalar(dropout):
        dropout_rate = float(dropout)
    else:
        dropout_rate = float(np.asarray(dropout, dtype=float).ravel()[0])

    recurrent_dropout = float(params.get("recurrent_dropout", 0.0))
    lstm_units = [int(v) for v in np.asarray(params.get("lstm_units", [8])).ravel()]
    dense_archi = [int(v) for v in np.asarray(params.get("dense_archi", [])).ravel()]

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for idx, units in enumerate(lstm_units):
        return_sequences = idx < len(lstm_units) - 1
        model.add(
            layers.LSTM(
                units,
                return_sequences=return_sequences,
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
                units,
                activation="relu",
                kernel_regularizer=reg,
                bias_regularizer=reg,
            )
        )
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(output_dim, activation="linear"))

    model.compile(
        optimizer=_make_optimizer(params),
        loss=_make_loss(params),
    )
    return model


def _build_inner_candidates(params):
    dropout_vals = params["Dropout"]
    l1l2_vals = params["l1l2"]

    if np.isscalar(dropout_vals):
        dropout_vals = [float(dropout_vals)]
    else:
        dropout_vals = [float(v) for v in np.asarray(dropout_vals, dtype=float).ravel()]

    l1l2_arr = np.asarray(l1l2_vals, dtype=float)
    if l1l2_arr.ndim == 1:
        l1l2_candidates = [l1l2_arr.tolist()]
    else:
        l1l2_candidates = [row.tolist() for row in l1l2_arr]

    inner_grid = {
        "Dropout": dropout_vals,
        "l1l2": l1l2_candidates,
    }

    candidates = []
    for combo in ParameterGrid(inner_grid):
        cand = copy.deepcopy(params)
        cand["Dropout"] = float(combo["Dropout"])
        cand["l1l2"] = list(combo["l1l2"])
        candidates.append(cand)

    return candidates


def _split_train_val(X_model, Y_model, validation_split):
    n = X_model.shape[0]
    if n < 6:
        return None

    val_len = int(np.ceil(n * float(validation_split)))
    val_len = max(val_len, 1)
    fit_end = n - val_len

    if fit_end < 2:
        return None

    X_fit = X_model[:fit_end]
    Y_fit = Y_model[:fit_end]
    X_val = X_model[fit_end:]
    Y_val = Y_model[fit_end:]

    if X_fit.shape[0] < 2 or X_val.shape[0] < 1:
        return None

    return X_fit, Y_fit, X_val, Y_val


def _fit_standardizer(X_fit):
    flat = X_fit.reshape(-1, X_fit.shape[-1])
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0, ddof=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def _apply_standardizer(X, mu, sd):
    return (X - mu[None, None, :]) / sd[None, None, :]


def _fit_one_seed(X_model, Y_model, X_test, params, seed):
    split = _split_train_val(
        X_model=X_model,
        Y_model=Y_model,
        validation_split=params.get("validation_split", 0.15),
    )
    if split is None:
        raise ValueError("Not enough training observations after validation split.")

    X_fit, Y_fit, X_val, Y_val = split

    if bool(params.get("standardize_x", True)):
        mu, sd = _fit_standardizer(X_fit)
        X_fit = _apply_standardizer(X_fit, mu, sd)
        X_val = _apply_standardizer(X_val, mu, sd)
        X_test = _apply_standardizer(X_test, mu, sd)

    _set_all_seeds(seed)
    keras.backend.clear_session()

    model = _build_lstm_network(
        input_shape=(X_model.shape[1], X_model.shape[2]),
        output_dim=Y_model.shape[1],
        params=params,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(params.get("patience", 20)),
            restore_best_weights=True,
            verbose=0,
        )
    ]

    model.fit(
        X_fit,
        Y_fit,
        validation_data=(X_val, Y_val),
        epochs=int(params.get("epochs", 500)),
        batch_size=int(params.get("batch_size", 32)),
        shuffle=bool(params.get("shuffle", False)),
        verbose=0,
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val, verbose=0)
    val_loss = float(np.mean((Y_val - val_pred) ** 2))

    test_pred = model.predict(X_test, verbose=0)
    return test_pred, val_loss


def _run_multi_seed(X_model, Y_model, X_test, params, nmc):
    outputs = {}
    for k in range(int(nmc)):
        pred_k, val_k = _fit_one_seed(
            X_model=X_model,
            Y_model=Y_model,
            X_test=X_test,
            params=params,
            seed=1234 + k,
        )
        outputs[k] = (pred_k, val_k)
    return outputs


def build_lstm_model_input(X, Y, forecast_idx, horizon, seq_len):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    train_end = forecast_idx - horizon
    if train_end < seq_len - 1:
        return None, None

    X_list = []
    Y_list = []

    for t in range(seq_len - 1, train_end + 1):
        x_seq = X[t - seq_len + 1 : t + 1, :]
        y_t = Y[t, :]

        if np.all(np.isfinite(x_seq)) and np.all(np.isfinite(y_t)):
            X_list.append(x_seq)
            Y_list.append(y_t)

    if len(X_list) == 0:
        return None, None

    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)


def build_lstm_test_input(X, forecast_idx, seq_len):
    X = np.asarray(X, dtype=float)

    if forecast_idx < seq_len - 1:
        return None

    x_seq = X[forecast_idx - seq_len + 1 : forecast_idx + 1, :]
    if not np.all(np.isfinite(x_seq)):
        return None

    return x_seq[None, :, :]


def _select_best_candidate(X_model, Y_model, X_test, cfg):
    candidates = _build_inner_candidates(cfg["params"])

    best_outputs = None
    best_params = None
    best_score = np.inf

    for cand_params in candidates:
        outputs = _run_multi_seed(
            X_model=X_model,
            Y_model=Y_model,
            X_test=X_test,
            params=cand_params,
            nmc=int(cfg["nmc"]),
        )

        this_val = np.array([outputs[k][1] for k in range(int(cfg["nmc"]))], dtype=float)
        score = float(np.nanmin(this_val))

        if score < best_score:
            best_score = score
            best_outputs = outputs
            best_params = cand_params

    return best_outputs, best_params


def _fit_fixed_candidate(X_model, Y_model, X_test, params, nmc):
    return _run_multi_seed(
        X_model=X_model,
        Y_model=Y_model,
        X_test=X_test,
        params=params,
        nmc=int(nmc),
    )


def run_oos_forecast(X, Y, dates, cfg):
    model_name = cfg["model_name"]
    seq_len = int(cfg["params"]["seq_len"])

    oos_start_ts = pd.Timestamp(cfg["oos_start"])
    start_candidates = np.where(dates >= oos_start_ts)[0]
    if start_candidates.size == 0:
        raise ValueError("No available sample date on or after oos_start.")

    first_oos_idx = int(start_candidates[0])
    oos_indices = list(range(first_oos_idx, X.shape[0]))

    T, M = Y.shape
    nmc = int(cfg["nmc"])
    navg = int(cfg["navg"])
    hyper_freq = int(cfg["hyper_freq"])

    Y_forecast_all = np.full((T, nmc, M), np.nan)
    Y_forecast_avg = np.full((T, M), np.nan)
    val_loss = np.full((T, nmc), np.nan)

    best_dropout_path = np.full(T, np.nan)
    best_l1_path = np.full(T, np.nan)
    best_l2_path = np.full(T, np.nan)

    print(model_name)

    current_best_params = None
    total_oos = len(oos_indices)
    oos_counter = 0

    for j, i in enumerate(oos_indices, start=1):
        X_model, Y_model = build_lstm_model_input(
            X=X,
            Y=Y,
            forecast_idx=i,
            horizon=cfg["horizon"],
            seq_len=seq_len,
        )
        X_test = build_lstm_test_input(
            X=X,
            forecast_idx=i,
            seq_len=seq_len,
        )

        if X_model is None or Y_model is None or X_test is None:
            continue

        oos_counter += 1
        retune = (oos_counter == 1) or ((oos_counter - 1) % hyper_freq == 0)

        if retune:
            outputs, current_best_params = _select_best_candidate(
                X_model=X_model,
                Y_model=Y_model,
                X_test=X_test,
                cfg=cfg,
            )
        else:
            outputs = _fit_fixed_candidate(
                X_model=X_model,
                Y_model=Y_model,
                X_test=X_test,
                params=current_best_params,
                nmc=nmc,
            )

        val_loss[i, :] = np.array([outputs[k][1] for k in range(nmc)], dtype=float)

        pred_list = []
        for k in range(nmc):
            pred_k = np.asarray(outputs[k][0], dtype=float)
            if pred_k.ndim == 1:
                pred_k = pred_k.reshape(1, -1)
            pred_list.append(pred_k)

        Y_forecast_all[i, :, :] = np.concatenate(pred_list, axis=0)

        best_seed_order = np.argsort(val_loss[i, :])
        Y_forecast_avg[i, :] = np.mean(Y_forecast_all[i, best_seed_order[:navg], :], axis=0)

        best_dropout_path[i] = float(current_best_params["Dropout"])

        reg_arr = np.asarray(current_best_params["l1l2"], dtype=float).ravel()
        if reg_arr.size == 1:
            best_l1_path[i] = float(reg_arr[0])
            best_l2_path[i] = float(reg_arr[0])
        else:
            best_l1_path[i] = float(reg_arr[0])
            best_l2_path[i] = float(reg_arr[1])

        current_best_val = np.nanmin(val_loss[i, :])

        if (j == 1) or (j % 12 == 0) or (j == total_oos):
            r2_now = np.array(
                [summarize_oos_metrics(Y[:, [k]], Y_forecast_avg[:, [k]])[1][0] for k in range(M)]
            )
            print(
                f"[{j:4d}/{total_oos}] "
                f"date={dates[i].strftime('%Y-%m-%d')} | "
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
    }


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

    save_dict = build_save_dict(cfg, X_df, Y_df, Y)
    save_dict.update(run_oos_forecast(X, Y, dates, cfg))

    os.makedirs("results", exist_ok=True)
    out_mat = os.path.join("results", _safe_result_name(cfg) + ".mat")
    sio.savemat(out_mat, save_dict)

    print("Saved to", out_mat)
    print("R2OOS:", save_dict[f"R2OOS_{cfg['model_name']}"])

    return save_dict, out_mat, X_df.columns.tolist(), Y_df.columns.tolist()


def build_sweep_configs(base_config, sweep_grid):
    cfg_list = []

    for combo in ParameterGrid(sweep_grid):
        cfg = copy.deepcopy(base_config)
        cfg["params"]["seq_len"] = int(combo["seq_len"])
        cfg["params"]["lstm_units"] = list(combo["lstm_units"])
        cfg["params"]["dense_archi"] = list(combo["dense_archi"])
        cfg["params"]["learning_rate"] = float(combo["learning_rate"])
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

        summary_rows, r2_rows, pval_rows, mse_rows = extract_summary_rows(
            save_dict=save_dict,
            cfg=cfg,
            mat_file=mat_file,
            y_columns=y_columns,
        )

        all_summary_rows.extend(summary_rows)
        all_r2_rows.extend(r2_rows)
        all_pval_rows.extend(pval_rows)
        all_mse_rows.extend(mse_rows)

    out_xlsx = os.path.join("results", _safe_result_name(base_cfg, sweep=True) + ".xlsx")
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