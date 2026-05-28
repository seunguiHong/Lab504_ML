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

import NNFuncBib as NFB
from utils import (
    get_cpu_count,
    make_dump_dir,
    safe_rmtree,
    load_dataset,
    enumerate_oos_forecast_indices,
    compute_training_end_index,
    run_seed_ensemble,
    build_dropout_l1l2_candidates,
    extract_seed_forecasts,
    extract_seed_validation_losses,
    top_validation_seed_mean,
    summarize_oos_metrics,
    build_save_dict,
    save_results_mat,
)


CONFIG = {
    "data_path": "data/target_and_features_sub.mat",
    "feature_groups": ["d12m_fwd"],
    "target_group": "rx",
    "target_indices": None,

    "horizon": 12,
    "oos_start": "1989-01-31",
    "hyper_freq": 60,

    "nmc": 100,
    "navg": 10,
    "run_tag": "ycNN_fwdchg",
    "out_file": "results_v2.0_sub/panelC_ycNN_fwdchg.mat",

    "model_func": NFB.NNModel,

    "params": {
        "archi": [3],
        "Dropout": [0.0],
        "l1l2": [0.0],
        "learning_rate": 0.02,
        "decay_rate": 0.001,
        "momentum": 0.9,
        "nesterov": True,
        "epochs": 500,
        "patience": 20,
        "batch_size": 32,
        "validation_split": 0.15,
        "shuffle": False,
        "loss_name": "mse",
        "huber_delta": 1.0,
    },
}


def build_nn_input(X, Y, forecast_idx, horizon):
    train_end = compute_training_end_index(forecast_idx, horizon)

    if train_end < 1:
        return None, None

    X_train = X[: train_end + 1]
    Y_train = Y[: train_end + 1]
    X_test = X[forecast_idx : forecast_idx + 1]
    Y_test = Y[forecast_idx : forecast_idx + 1]

    if not np.all(np.isfinite(X_test)):
        return None, None

    ok = np.all(np.isfinite(X_train), axis=1) & np.all(np.isfinite(Y_train), axis=1)
    X_train = X_train[ok]
    Y_train = Y_train[ok]

    if X_train.shape[0] < 30:
        return None, None

    X_model = np.vstack([X_train, X_test])
    Y_model = np.vstack([Y_train, Y_test])

    return X_model, Y_model


def fit_candidate(X_model, Y_model, cfg, params, dumploc, ncpus, refit):
    outputs = run_seed_ensemble(
        model_func=cfg["model_func"],
        ncpus=ncpus,
        nmc=cfg["nmc"],
        X_model=X_model,
        Y_model=Y_model,
        params=params,
        dumploc=dumploc,
        refit=refit,
    )

    val_loss = extract_seed_validation_losses(outputs, cfg["nmc"])
    score = float(np.nanmean(val_loss))

    return outputs, score


def select_candidate(X_model, Y_model, cfg, dumploc, ncpus, refit):
    best_outputs = None
    best_params = None
    best_score = np.inf

    for params in build_dropout_l1l2_candidates(cfg["params"]):
        outputs, score = fit_candidate(
            X_model=X_model,
            Y_model=Y_model,
            cfg=cfg,
            params=params,
            dumploc=dumploc,
            ncpus=ncpus,
            refit=refit,
        )

        if score < best_score:
            best_outputs = outputs
            best_params = copy.deepcopy(params)
            best_score = score

    return best_outputs, best_params, best_score


def compute_refit_flag(oos_count, hyper_freq, prev_best_val, best_val_since_refit):
    if oos_count == 1:
        return True

    if oos_count % hyper_freq == 0:
        return bool(np.isfinite(prev_best_val) and prev_best_val > best_val_since_refit)

    return False


def run_oos_forecast(X, Y, dates, cfg, dumploc, ncpus):
    dates = pd.DatetimeIndex(dates)
    oos_indices = enumerate_oos_forecast_indices(dates, cfg["oos_start"])

    T, M = Y.shape
    nmc = int(cfg["nmc"])
    navg = int(cfg["navg"])
    horizon = int(cfg["horizon"])
    hyper_freq = int(cfg["hyper_freq"])

    if navg > nmc:
        raise ValueError("navg cannot exceed nmc.")

    Y_forecast = np.full((T, M), np.nan)
    Y_forecast_all = np.full((T, nmc, M), np.nan)
    val_loss = np.full((T, nmc), np.nan)

    current_params = None
    current_score = np.nan

    oos_count = 0
    total_oos = len(oos_indices)

    prev_best_val = np.nan
    best_val_since_refit = np.inf

    print(f"Total OOS steps: {total_oos}")

    for step, forecast_idx in enumerate(oos_indices, start=1):
        X_model, Y_model = build_nn_input(
            X=X,
            Y=Y,
            forecast_idx=forecast_idx,
            horizon=horizon,
        )

        if X_model is None:
            continue

        oos_count += 1

        retune = (oos_count == 1) or (oos_count % hyper_freq == 0)
        refit = compute_refit_flag(
            oos_count=oos_count,
            hyper_freq=hyper_freq,
            prev_best_val=prev_best_val,
            best_val_since_refit=best_val_since_refit,
        )

        if retune:
            outputs, current_params, current_score = select_candidate(
                X_model=X_model,
                Y_model=Y_model,
                cfg=cfg,
                dumploc=dumploc,
                ncpus=ncpus,
                refit=refit,
            )
        else:
            outputs, current_score = fit_candidate(
                X_model=X_model,
                Y_model=Y_model,
                cfg=cfg,
                params=current_params,
                dumploc=dumploc,
                ncpus=ncpus,
                refit=refit,
            )

        seed_forecasts = extract_seed_forecasts(outputs, nmc)
        seed_val_loss = extract_seed_validation_losses(outputs, nmc)

        Y_forecast_all[forecast_idx] = seed_forecasts
        Y_forecast[forecast_idx] = top_validation_seed_mean(
            seed_forecasts=seed_forecasts,
            seed_val_loss=seed_val_loss,
            navg=navg,
        )
        val_loss[forecast_idx] = seed_val_loss

        current_best_val = float(np.nanmin(seed_val_loss))

        if refit:
            best_val_since_refit = current_best_val
        else:
            best_val_since_refit = min(best_val_since_refit, current_best_val)

        prev_best_val = current_best_val

        if step == 1 or step % 12 == 0 or step == total_oos:
            r2_now = summarize_oos_metrics(
                Y_true=Y,
                Y_pred=Y_forecast,
                hac_lags=horizon,
            )[1]

            print(
                f"[{step:4d}/{total_oos}] "
                f"date={dates[forecast_idx].strftime('%Y-%m-%d')} | "
                f"oos_count={oos_count} | "
                f"retune={retune} | "
                f"refit={refit} | "
                f"val_mean={float(np.nanmean(seed_val_loss)):10.6f} | "
                f"val_min={current_best_val:10.6f} | "
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

    ncpus = get_cpu_count()
    dumploc = make_dump_dir()

    try:
        X_df, Y_df = load_dataset(
            data_path=cfg["data_path"],
            feature_groups=cfg["feature_groups"],
            target_group=cfg["target_group"],
            target_indices=cfg.get("target_indices"),
        )

        X = X_df.to_numpy(dtype=float)
        Y = Y_df.to_numpy(dtype=float)
        dates = pd.DatetimeIndex(X_df.index)

        print(f"CPU count: {ncpus}")
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
            dumploc=dumploc,
            ncpus=ncpus,
        )

        save_dict = build_save_dict(cfg, X_df, Y_df, Y)
        save_dict.update(result)

        save_results_mat(cfg["out_file"], save_dict)

        print("Saved to", cfg["out_file"])
        print("R2OOS:", np.round(save_dict["R2OOS"], 4))

        return save_dict, cfg["out_file"]

    finally:
        safe_rmtree(dumploc)


if __name__ == "__main__":
    run_experiment(CONFIG)