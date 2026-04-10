#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import copy
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

import NNFuncBib as NFB
from utils import (
    get_cpu_count,
    make_dump_dir,
    safe_rmtree,
    load_dataset,
    summarize_oos_metrics,
    build_save_dict,
    result_name,
    save_results_mat,
)

BASE_CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["dy_pc"],
    "target_group": "dy",
    "target_indices": None,
    "horizon": 12,
    "oos_start": "1989-01-31",
    "hyper_freq": 60,
    "nmc": 20,
    "navg": 5,
    "run_tag": "pc12",
    "model_func": NFB.NNModel,
    "model_name": "NNModel",
    "results_dir": "results",
    "params": {
        "archi": [3.3],
        "Dropout": [0.0],
        "l1l2": [1e-4, 1e-5],
        "learning_rate": 0.03,
        "decay_rate": 0.001,
        "momentum": 0.9,
        "nesterov": True,
        "epochs": 500,
        "patience": 20,
        "batch_size": 32,
        "validation_split": 0.15,
        "shuffle": False,
        "loss_name": "huber",
        "huber_delta": 1.0,
    },
}


def build_model_input(X, Y, forecast_index, horizon):
    """
    Build the input expected by NNFuncBib.NNModel.

    By convention, the last row is the forecast row.
    Training data can only use information up to forecast_index - horizon.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    train_end = int(forecast_index - horizon)
    if train_end < 1:
        return None, None

    X_train = X[: train_end + 1, :]
    Y_train = Y[: train_end + 1, :]
    X_test = X[forecast_index : forecast_index + 1, :]
    Y_test = Y[forecast_index : forecast_index + 1, :]

    if X_test.shape[0] != 1 or Y_test.shape[0] != 1:
        return None, None

    X_model = np.vstack([X_train, X_test])
    Y_model = np.vstack([Y_train, Y_test])

    return X_model, Y_model


def _worker_run_one_seed(args):
    model_func, seed_no, X_model, Y_model, params, refit, dumploc = args
    return model_func(
        X=X_model,
        Y=Y_model,
        no=seed_no,
        params=params,
        refit=refit,
        dumploc=dumploc,
    )


def run_parallel_seeds(model_func, ncpus, nmc, X_model, Y_model, params, refit, dumploc):
    """
    Run multiple seeds in parallel and return a dict:
        outputs[k] = (y_pred, val_loss)
    """
    jobs = [
        (model_func, k, X_model, Y_model, params, refit, dumploc)
        for k in range(int(nmc))
    ]

    use_ncpus = min(int(ncpus), int(nmc))
    if use_ncpus <= 1:
        results = [_worker_run_one_seed(job) for job in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=use_ncpus) as pool:
            results = pool.map(_worker_run_one_seed, jobs)

    return {k: results[k] for k in range(int(nmc))}


def _normalize_l1l2(x):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return [0.0, 0.0]
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    return [float(arr[0]), float(arr[1])]


def _build_inner_candidates(params):
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


def _select_best_candidate(X_model, Y_model, cfg, dumploc, ncpus, refit):
    candidates = _build_inner_candidates(cfg["params"])

    best_outputs = None
    best_params = None
    best_score = np.inf

    for cand_params in candidates:
        outputs = run_parallel_seeds(
            model_func=cfg["model_func"],
            ncpus=ncpus,
            nmc=int(cfg["nmc"]),
            X_model=X_model,
            Y_model=Y_model,
            params=cand_params,
            refit=refit,
            dumploc=dumploc,
        )

        val_vec = np.array([outputs[k][1] for k in range(int(cfg["nmc"]))], dtype=float)
        score = float(np.nanmean(val_vec))

        if score < best_score:
            best_score = score
            best_outputs = outputs
            best_params = cand_params

    return best_outputs, best_params, best_score


def run_oos_forecast(X, Y, dates, cfg, dumploc, ncpus):
    model_name = cfg["model_name"]

    dates = pd.DatetimeIndex(dates)
    start_candidates = np.where(dates >= pd.Timestamp(cfg["oos_start"]))[0]
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

    current_best_params = None
    current_best_score = np.nan

    print(model_name)

    oos_counter = 0
    total_oos = len(oos_indices)

    for j, forecast_index in enumerate(oos_indices, start=1):
        X_model, Y_model = build_model_input(
            X=X,
            Y=Y,
            forecast_index=forecast_index,
            horizon=int(cfg["horizon"]),
        )
        if X_model is None or Y_model is None:
            continue

        oos_counter += 1
        retune = (oos_counter == 1) or ((oos_counter - 1) % hyper_freq == 0)

        if retune:
            outputs, current_best_params, current_best_score = _select_best_candidate(
                X_model=X_model,
                Y_model=Y_model,
                cfg=cfg,
                dumploc=dumploc,
                ncpus=ncpus,
                refit=True,
            )
        else:
            outputs = run_parallel_seeds(
                model_func=cfg["model_func"],
                ncpus=ncpus,
                nmc=nmc,
                X_model=X_model,
                Y_model=Y_model,
                params=current_best_params,
                refit=False,
                dumploc=dumploc,
            )

        val_loss[forecast_index, :] = np.array(
            [outputs[k][1] for k in range(nmc)],
            dtype=float,
        )

        pred_list = []
        for k in range(nmc):
            pred_k = np.asarray(outputs[k][0], dtype=float)
            if pred_k.ndim == 1:
                pred_k = pred_k.reshape(1, -1)
            pred_list.append(pred_k)

        Y_forecast_all[forecast_index, :, :] = np.concatenate(pred_list, axis=0)

        best_seed_order = np.argsort(val_loss[forecast_index, :])
        Y_forecast_avg[forecast_index, :] = np.mean(
            Y_forecast_all[forecast_index, best_seed_order[:navg], :],
            axis=0,
        )

        reg_arr = np.asarray(current_best_params["l1l2"], dtype=float).ravel()
        best_dropout_path[forecast_index] = float(current_best_params["Dropout"])
        best_l1_path[forecast_index] = float(reg_arr[0])
        best_l2_path[forecast_index] = float(reg_arr[1] if reg_arr.size >= 2 else reg_arr[0])

        current_best_val = float(np.nanmean(val_loss[forecast_index, :]))

        if (j == 1) or (j % 12 == 0) or (j == total_oos):
            r2_now = np.array(
                [summarize_oos_metrics(Y[:, [m]], Y_forecast_avg[:, [m]])[1][0] for m in range(M)]
            )
            print(
                f"[{j:4d}/{total_oos}] "
                f"date={dates[forecast_index].strftime('%Y-%m-%d')} | "
                f"retune={retune} | "
                f"val={current_best_val:10.6f} | "
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
    dumploc = make_dump_dir()

    X_df, Y_df = load_dataset(
        mat_path=cfg["mat_path"],
        feature_groups=cfg["feature_groups"],
        target_group=cfg["target_group"],
        target_indices=cfg.get("target_indices", None),
    )

    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(X_df.index)

    print(f"CPU count: {ncpus}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"OOS start: {cfg['oos_start']}")
    print(f"Feature groups: {cfg['feature_groups']}")
    print(f"Params: {json.dumps(cfg['params'])}")

    save_dict = build_save_dict(cfg, X_df, Y_df, Y)
    save_dict.update(run_oos_forecast(X, Y, dates, cfg, dumploc, ncpus))

    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    out_mat = os.path.join(results_dir, result_name(cfg) + ".mat")
    save_results_mat(out_mat, save_dict)

    print("Saved to", out_mat)
    print("R2OOS:", np.round(save_dict[f"R2OOS_{cfg['model_name']}"], 4))

    safe_rmtree(dumploc)

    return save_dict, out_mat, X_df.columns.tolist(), Y_df.columns.tolist()


if __name__ == "__main__":
    run_experiment(BASE_CONFIG)