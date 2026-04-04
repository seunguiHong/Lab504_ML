#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import json

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import ParameterGrid

import NNFuncBib as NFB
from Utils import (
    get_cpu_count,
    make_dump_dir,
    safe_rmtree,
    load_dataset,
    build_model_input,
    run_parallel_seeds,
    summarize_oos_metrics,
    build_save_dict,
    result_name,
    extract_summary_rows,
    save_sweep_summary_to_excel,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

RUN_MODE = "sweep"

BASE_CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "feature_groups": ["dy_pc1", "dy_pc2", "dy_pc3"],
    "target_group": "dy",
    "horizon": 12,
    "oos_start": "1989-01-31",
    "hyper_freq": 60,
    "nmc": 10,
    "navg": 2,
    "run_tag": "pc123",
    "model_func": NFB.NNModel,
    "model_name": "NNModel",
    "params": {
        "archi": [3, 3],
        "Dropout": 0.0,
        "l1l2": [1e-4, 1e-5],
        "learning_rate": 0.03,
        "decay_rate": 0.001,
        "momentum": 0.9,
        "nesterov": True,
        "epochs": 500,
        "patience": 20,
        "batch_size": 32,
        "validation_split": 0.15,
        "shuffle": True,
        "loss_name": "mse",
        "huber_delta": 1.0,
    },
}

SWEEP_GRID = {
    "archi": [[3, 3], [3]],
    "Dropout": [0.0, 0.1],
    "l1l2": [[1e-4, 1e-5], [1e-3, 1e-4]],
    "learning_rate": [0.03, 0.01],
}


def run_oos_forecast(X, Y, dates, cfg, dumploc, ncpus):
    model_func = cfg["model_func"]
    model_name = cfg["model_name"]

    oos_start_ts = pd.Timestamp(cfg["oos_start"])
    start_candidates = np.where(dates >= oos_start_ts)[0]
    if start_candidates.size == 0:
        raise ValueError("No available sample date on or after oos_start.")

    first_oos_idx = int(start_candidates[0])
    oos_indices = list(range(first_oos_idx, X.shape[0]))

    T, M = Y.shape
    nmc = int(cfg["nmc"])
    navg = int(cfg["navg"])

    Y_forecast_all = np.full((T, nmc, M), np.nan)
    Y_forecast_avg = np.full((T, M), np.nan)
    val_loss = np.full((T, nmc), np.nan)

    print(model_name)

    oos_counter = 0
    best_val_since_refit = np.inf
    prev_best_val = np.nan
    total_oos = len(oos_indices)

    for j, i in enumerate(oos_indices, start=1):
        X_model, Y_model = build_model_input(X, Y, i, cfg["horizon"])
        if X_model is None:
            continue

        oos_counter += 1

        if oos_counter == 1:
            refit = True
        elif oos_counter % int(cfg["hyper_freq"]) == 0:
            refit = bool(np.isfinite(prev_best_val) and prev_best_val > best_val_since_refit)
        else:
            refit = False

        outputs = run_parallel_seeds(
            model_func,
            ncpus,
            nmc,
            X_model,
            Y_model,
            params=cfg["params"],
            refit=refit,
            dumploc=dumploc,
        )

        val_loss[i, :] = np.array([outputs[k][1] for k in range(nmc)], dtype=float)
        Y_forecast_all[i, :, :] = np.concatenate([outputs[k][0] for k in range(nmc)], axis=0)

        best_seed_order = np.argsort(val_loss[i, :])
        Y_forecast_avg[i, :] = np.mean(Y_forecast_all[i, best_seed_order[:navg], :], axis=0)

        current_best_val = np.nanmin(val_loss[i, :])
        if refit:
            best_val_since_refit = current_best_val
        else:
            best_val_since_refit = min(best_val_since_refit, current_best_val)
        prev_best_val = current_best_val

        if (j == 1) or (j % 12 == 0) or (j == total_oos):
            r2_now = np.array(
                [summarize_oos_metrics(Y[:, [k]], Y_forecast_avg[:, [k]])[1][0] for k in range(M)]
            )
            print(
                f"[{j:4d}/{total_oos}] "
                f"date={dates[i].strftime('%Y-%m-%d')} | "
                f"refit={refit} | "
                f"val={current_best_val:10.6f}"
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
    }


def run_experiment(custom_config=None):
    cfg = copy.deepcopy(custom_config if custom_config is not None else BASE_CONFIG)

    ncpus = get_cpu_count()
    dumploc = make_dump_dir()

    X_df, Y_df = load_dataset(
        mat_path=cfg["mat_path"],
        feature_groups=cfg["feature_groups"],
        target_group=cfg["target_group"],
    )

    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(X_df.index)

    save_dict = build_save_dict(cfg, X_df, Y_df, Y)
    save_dict.update(run_oos_forecast(X, Y, dates, cfg, dumploc, ncpus))

    os.makedirs("results", exist_ok=True)
    out_mat = os.path.join("results", result_name(cfg) + ".mat")
    sio.savemat(out_mat, save_dict)

    print("Saved to", out_mat)
    safe_rmtree(dumploc)

    return save_dict, out_mat, X_df.columns.tolist(), Y_df.columns.tolist()


def build_sweep_configs(base_config, sweep_grid):
    cfg_list = []

    for combo in ParameterGrid(sweep_grid):
        cfg = copy.deepcopy(base_config)
        cfg["params"].update(combo)
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

    out_xlsx = os.path.join("results", result_name(base_cfg, sweep=True) + ".xlsx")
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