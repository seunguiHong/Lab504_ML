#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import json
import time

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import ParameterGrid

import NNFuncBib as NFB
from Utils import (
    to_1d,
    to_name_list,
    yyyymm_to_month_end,
    r2_oos,
    r2_oos_pvalue,
    get_cpu_count,
    make_dump_dir,
    safe_rmtree,
    build_model_input,
    run_parallel_seeds,
    build_save_dict,
    extract_summary_rows,
    save_sweep_summary_to_excel,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

RUN_MODE = "sweep"

BASE_CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "target_group": "dy",
    "horizon": 12,
    "oos_start": "1989-01-31",
    "hyper_freq": 60,
    "nmc": 1,
    "navg": 1,
    "run_tag": "macro_dual",
    "model_func": NFB.NNDualModel,
    "model_name": "NNDualModel",
    "branch_config": {
        "yield": {
            "feature_groups": ["dy_pc1", "dy_pc2"],
            "archi": [3, 3],
            "dropout": 0.0,
            "l1l2": [1e-4, 5e-5],
        },
        "macro": {
            "feature_groups": ["macro"],
            "archi": [32, 16],
            "dropout": 0.1,
            "l1l2": [1e-4, 5e-5],
        },
    },
    "params": {
        "head_archi": [16],
        "head_dropout": 0.0,
        "head_l1l2": [1e-4, 5e-5],
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
    "yield_archi": [[3, 3], [3]],
    "yield_dropout": [0.0, 0.1],
    "yield_l1l2": [[1e-4, 5e-5], [1e-3, 1e-4]],
    "macro_archi": [[32, 16], [16, 8]],
    "macro_dropout": [0.0, 0.1],
    "macro_l1l2": [[1e-4, 5e-5], [1e-3, 1e-4]],
    "head_archi": [[16], [8]],
    "head_dropout": [0.0, 0.1],
    "head_l1l2": [[1e-4, 5e-5], [1e-3, 1e-4]],
    "learning_rate": [0.03, 0.01],
}


def load_group_df(Xmat, group_name):
    block = getattr(Xmat, group_name)

    time_vec = to_1d(block.Time).astype(int)
    data = np.asarray(block.data, dtype=float)
    names = to_name_list(block.names)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    dates = yyyymm_to_month_end(time_vec)

    if group_name != "yields":
        names = [f"{group_name}__{name}" for name in names]

    df = pd.DataFrame(data, index=dates, columns=names)
    df.index.name = "Date"
    return df


def load_dual_dataset(mat_path, branch_config, target_group):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    Xmat = mat["X"]
    ymat = mat["y"]

    yblock = getattr(ymat, target_group)
    y_time = to_1d(yblock.Time).astype(int)
    y_data = np.asarray(yblock.data, dtype=float)
    y_names = to_name_list(yblock.names)

    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)

    y_dates = yyyymm_to_month_end(y_time)
    Y_df = pd.DataFrame(y_data, index=y_dates, columns=y_names)
    Y_df.index.name = "Date"

    merged = Y_df.copy()
    branch_frames = {}

    for branch_name, spec in branch_config.items():
        parts = [load_group_df(Xmat, grp) for grp in spec["feature_groups"]]
        branch_df = pd.concat(parts, axis=1)
        branch_frames[branch_name] = branch_df
        merged = merged.join(branch_df, how="inner")

    merged = merged.dropna().copy()

    X_parts = []
    branch_specs = []
    col_start = 0

    for branch_name, spec in branch_config.items():
        cols = [c for c in branch_frames[branch_name].columns if c in merged.columns]
        branch_df = merged[cols].copy()

        X_parts.append(branch_df)

        col_end = col_start + branch_df.shape[1]
        branch_specs.append(
            {
                "name": branch_name,
                "feature_groups": list(spec["feature_groups"]),
                "columns": cols,
                "col_start": int(col_start),
                "col_end": int(col_end),
                "input_dim": int(branch_df.shape[1]),
                "archi": list(spec["archi"]),
                "dropout": float(spec["dropout"]),
                "l1l2": list(spec["l1l2"]),
            }
        )
        col_start = col_end

    X_df = pd.concat(X_parts, axis=1)
    Y_df = merged[Y_df.columns].copy()

    return X_df, Y_df, branch_specs


def macro_result_name(cfg, sweep=False):
    if sweep:
        return f"sweep__{cfg['target_group']}__{cfg['run_tag']}__MacroDual__h{cfg['horizon']}"

    yspec = cfg["branch_config"]["yield"]
    mspec = cfg["branch_config"]["macro"]
    hspec = cfg["params"]

    y_arch = "x".join(map(str, yspec["archi"]))
    m_arch = "x".join(map(str, mspec["archi"]))
    h_arch = "x".join(map(str, hspec["head_archi"]))

    return (
        f"{cfg['target_group']}__{cfg['run_tag']}__MacroDual__h{cfg['horizon']}"
        f"__ya{y_arch}__ma{m_arch}__ha{h_arch}"
        f"__ylr{yspec['l1l2'][0]:g}-{yspec['l1l2'][1]:g}"
        f"__mlr{mspec['l1l2'][0]:g}-{mspec['l1l2'][1]:g}"
        f"__hlr{hspec['head_l1l2'][0]:g}-{hspec['head_l1l2'][1]:g}"
        f"__lr{hspec['learning_rate']:g}"
    )


def run_oos_forecast(X, Y, dates, cfg, dumploc, ncpus, model_params):
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
    run_start = time.time()

    for j, i in enumerate(oos_indices, start=1):
        iter_start = time.time()

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
            params=model_params,
            refit=refit,
            dumploc=dumploc,
        )

        val_loss[i, :] = np.array([outputs[k][1] for k in range(nmc)], dtype=float)
        Y_forecast_all[i, :, :] = np.concatenate([outputs[k][0] for k in range(nmc)], axis=0)

        seed_order = np.argsort(val_loss[i, :])
        Y_forecast_avg[i, :] = np.mean(Y_forecast_all[i, seed_order[:navg], :], axis=0)

        current_best_val = np.nanmin(val_loss[i, :])

        if refit:
            best_val_since_refit = current_best_val
        else:
            best_val_since_refit = min(best_val_since_refit, current_best_val)

        prev_best_val = current_best_val

        if (j == 1) or (j % 12 == 0) or (j == total_oos):
            iter_elapsed = time.time() - iter_start
            total_elapsed = time.time() - run_start
            r2_now = np.array([r2_oos(Y[:, k], Y_forecast_avg[:, k]) for k in range(M)])

            print(
                f"[{j:4d}/{total_oos}] "
                f"date={dates[i].strftime('%Y-%m-%d')} | "
                f"refit={refit} | "
                f"val={current_best_val:10.6f} | "
                f"iter={iter_elapsed:7.2f}s | "
                f"elapsed={total_elapsed:8.1f}s"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    return {
        f"ValLoss_{model_name}": val_loss,
        f"Y_forecast_{model_name}": Y_forecast_all,
        f"Y_forecast_agg_{model_name}": Y_forecast_avg,
        f"MSE_{model_name}": np.nanmean(np.square(Y - Y_forecast_avg), axis=0),
        f"R2OOS_{model_name}": np.array([r2_oos(Y[:, k], Y_forecast_avg[:, k]) for k in range(M)]),
        f"R2OOS_pval_{model_name}": np.array([r2_oos_pvalue(Y[:, k], Y_forecast_avg[:, k]) for k in range(M)]),
    }


def run_experiment(custom_config=None):
    cfg = copy.deepcopy(custom_config if custom_config is not None else BASE_CONFIG)

    ncpus = get_cpu_count()
    dumploc = make_dump_dir()

    print("CPU count is:", ncpus)

    X_df, Y_df, branch_specs = load_dual_dataset(
        mat_path=cfg["mat_path"],
        branch_config=cfg["branch_config"],
        target_group=cfg["target_group"],
    )

    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(X_df.index)

    model_params = dict(cfg["params"])
    model_params["branch_specs"] = branch_specs

    save_dict = build_save_dict(cfg, X_df, Y_df, Y)
    save_dict["BranchConfigJSON"] = json.dumps(cfg["branch_config"])
    save_dict["BranchSpecsJSON"] = json.dumps(branch_specs)
    save_dict["ParamsJSON"] = json.dumps(model_params)
    save_dict["RunConfigJSON"] = json.dumps(
        {
            "horizon": cfg["horizon"],
            "oos_start": cfg["oos_start"],
            "hyper_freq": cfg["hyper_freq"],
            "nmc": cfg["nmc"],
            "navg": cfg["navg"],
            "run_tag": cfg["run_tag"],
        }
    )

    save_dict.update(
        run_oos_forecast(
            X=X,
            Y=Y,
            dates=dates,
            cfg=cfg,
            dumploc=dumploc,
            ncpus=ncpus,
            model_params=model_params,
        )
    )

    print("R2OOS:", np.round(save_dict[f"R2OOS_{cfg['model_name']}"], 4))

    os.makedirs("results_macro_dual", exist_ok=True)
    out_file = os.path.join("results_macro_dual", macro_result_name(cfg) + ".mat")
    sio.savemat(out_file, save_dict)

    print("Saved to", out_file)
    safe_rmtree(dumploc)

    return save_dict, out_file, X_df.columns.tolist(), Y_df.columns.tolist()


def build_sweep_configs(base_config, sweep_grid):
    cfg_list = []

    for combo in ParameterGrid(sweep_grid):
        cfg = copy.deepcopy(base_config)

        cfg["branch_config"]["yield"]["archi"] = combo["yield_archi"]
        cfg["branch_config"]["yield"]["dropout"] = combo["yield_dropout"]
        cfg["branch_config"]["yield"]["l1l2"] = combo["yield_l1l2"]

        cfg["branch_config"]["macro"]["archi"] = combo["macro_archi"]
        cfg["branch_config"]["macro"]["dropout"] = combo["macro_dropout"]
        cfg["branch_config"]["macro"]["l1l2"] = combo["macro_l1l2"]

        cfg["params"]["head_archi"] = combo["head_archi"]
        cfg["params"]["head_dropout"] = combo["head_dropout"]
        cfg["params"]["head_l1l2"] = combo["head_l1l2"]
        cfg["params"]["learning_rate"] = combo["learning_rate"]

        cfg_list.append(cfg)

    return cfg_list


def run_hyperparameter_sweep(base_config=None, sweep_grid=None):
    base_cfg = copy.deepcopy(base_config if base_config is not None else BASE_CONFIG)
    grid = copy.deepcopy(sweep_grid if sweep_grid is not None else SWEEP_GRID)

    os.makedirs("results_macro_dual", exist_ok=True)

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
        print("yield branch:", json.dumps(cfg["branch_config"]["yield"], ensure_ascii=False))
        print("macro branch:", json.dumps(cfg["branch_config"]["macro"], ensure_ascii=False))
        print("head params:", json.dumps(cfg["params"], ensure_ascii=False))

        save_dict, mat_file, _, y_columns = run_experiment(cfg)

        summary_cfg = copy.deepcopy(cfg)
        summary_cfg["model_name"] = cfg["model_name"]

        summary_rows, r2_rows, pval_rows, mse_rows = extract_summary_rows(
            save_dict=save_dict,
            cfg=summary_cfg,
            mat_file=mat_file,
            y_columns=y_columns,
        )

        all_summary_rows.extend(summary_rows)
        all_r2_rows.extend(r2_rows)
        all_pval_rows.extend(pval_rows)
        all_mse_rows.extend(mse_rows)

    out_xlsx = os.path.join("results_macro_dual", macro_result_name(base_cfg, sweep=True) + ".xlsx")

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