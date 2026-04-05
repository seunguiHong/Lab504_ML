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
            "dropout": [0.0],
            "l1l2": [[1e-4, 5e-5]],
        },
        "macro": {
            "feature_groups": ["macro"],
            "archi": [32, 16],
            "dropout": [0.1],
            "l1l2": [[1e-4, 5e-5]],
        },
    },
    "params": {
        "head_archi": [16],
        "head_dropout": [0.0],
        "head_l1l2": [[1e-4, 5e-5]],
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
    "macro_archi": [[32, 16], [16, 8]],
    "head_archi": [[16], [8]],
    "learning_rate": [0.03, 0.01],

    "yield_dropout": [[0.0], [0.1, 0.3]],
    "yield_l1l2": [[[1e-4, 5e-5]], [[1e-4, 5e-5], [1e-3, 1e-4]]],

    "macro_dropout": [[0.0], [0.1, 0.3]],
    "macro_l1l2": [[[1e-4, 5e-5]], [[1e-4, 5e-5], [1e-3, 1e-4]]],

    "head_dropout": [[0.0], [0.1, 0.3]],
    "head_l1l2": [[[1e-4, 5e-5]], [[1e-4, 5e-5], [1e-3, 1e-4]]],
}


def _arch_tag(x):
    return "x".join(str(int(v)) for v in np.asarray(x).ravel())


def _pair_tag(x):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return "0-0"
    if arr.size == 1:
        return f"{arr[0]:g}-{arr[0]:g}"
    return f"{arr[0]:g}-{arr[1]:g}"


def _list_tag(x):
    return "|".join(f"{float(v):g}" for v in np.asarray(x, dtype=float).ravel())


def _reg_tag(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return _pair_tag(arr)
    return "|".join(_pair_tag(row) for row in arr)


def macro_result_name(cfg, sweep=False):
    target = str(cfg["target_group"])
    run_tag = str(cfg["run_tag"])
    horizon = f"h{int(cfg['horizon'])}"

    if sweep:
        return "__".join(["sweep", target, run_tag, "MacroDual", horizon])

    yspec = cfg["branch_config"]["yield"]
    mspec = cfg["branch_config"]["macro"]
    hspec = cfg["params"]

    parts = [
        target,
        run_tag,
        "MacroDual",
        horizon,
        f"ya{_arch_tag(yspec['archi'])}",
        f"ma{_arch_tag(mspec['archi'])}",
        f"ha{_arch_tag(hspec['head_archi'])}",
        f"yd{_list_tag(yspec['dropout'])}",
        f"md{_list_tag(mspec['dropout'])}",
        f"hd{_list_tag(hspec['head_dropout'])}",
        f"ylr{_reg_tag(yspec['l1l2'])}",
        f"mlr{_reg_tag(mspec['l1l2'])}",
        f"hlr{_reg_tag(hspec['head_l1l2'])}",
        f"lr{float(hspec['learning_rate']):g}",
    ]
    return "__".join(parts)


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
                "dropout": float(np.asarray(spec["dropout"], dtype=float).ravel()[0]),
                "l1l2": list(np.asarray(spec["l1l2"], dtype=float).reshape(-1)[:2]),
            }
        )
        col_start = col_end

    X_df = pd.concat(X_parts, axis=1)
    Y_df = merged[Y_df.columns].copy()

    return X_df, Y_df, branch_specs


def _build_inner_candidates(cfg):
    y_do = np.asarray(cfg["branch_config"]["yield"]["dropout"], dtype=float).ravel().tolist()
    m_do = np.asarray(cfg["branch_config"]["macro"]["dropout"], dtype=float).ravel().tolist()
    h_do = np.asarray(cfg["params"]["head_dropout"], dtype=float).ravel().tolist()

    y_reg = np.asarray(cfg["branch_config"]["yield"]["l1l2"], dtype=float)
    m_reg = np.asarray(cfg["branch_config"]["macro"]["l1l2"], dtype=float)
    h_reg = np.asarray(cfg["params"]["head_l1l2"], dtype=float)

    if y_reg.ndim == 1:
        y_reg = [y_reg.tolist()]
    else:
        y_reg = [row.tolist() for row in y_reg]

    if m_reg.ndim == 1:
        m_reg = [m_reg.tolist()]
    else:
        m_reg = [row.tolist() for row in m_reg]

    if h_reg.ndim == 1:
        h_reg = [h_reg.tolist()]
    else:
        h_reg = [row.tolist() for row in h_reg]

    inner_grid = {
        "yield_dropout": y_do,
        "yield_l1l2": y_reg,
        "macro_dropout": m_do,
        "macro_l1l2": m_reg,
        "head_dropout": h_do,
        "head_l1l2": h_reg,
    }

    candidates = []
    for combo in ParameterGrid(inner_grid):
        cand = copy.deepcopy(cfg)

        cand["branch_config"]["yield"]["dropout"] = float(combo["yield_dropout"])
        cand["branch_config"]["yield"]["l1l2"] = list(combo["yield_l1l2"])

        cand["branch_config"]["macro"]["dropout"] = float(combo["macro_dropout"])
        cand["branch_config"]["macro"]["l1l2"] = list(combo["macro_l1l2"])

        cand["params"]["head_dropout"] = float(combo["head_dropout"])
        cand["params"]["head_l1l2"] = list(combo["head_l1l2"])

        candidates.append(cand)

    return candidates


def _apply_candidate_to_branch_specs(base_branch_specs, cand_cfg):
    out = copy.deepcopy(base_branch_specs)
    spec_map = {spec["name"]: spec for spec in out}

    spec_map["yield"]["archi"] = list(cand_cfg["branch_config"]["yield"]["archi"])
    spec_map["yield"]["dropout"] = float(cand_cfg["branch_config"]["yield"]["dropout"])
    spec_map["yield"]["l1l2"] = list(cand_cfg["branch_config"]["yield"]["l1l2"])

    spec_map["macro"]["archi"] = list(cand_cfg["branch_config"]["macro"]["archi"])
    spec_map["macro"]["dropout"] = float(cand_cfg["branch_config"]["macro"]["dropout"])
    spec_map["macro"]["l1l2"] = list(cand_cfg["branch_config"]["macro"]["l1l2"])

    return out


def _select_best_candidate(X_model, Y_model, cfg, base_branch_specs, dumploc, ncpus, refit):
    candidates = _build_inner_candidates(cfg)

    best_outputs = None
    best_cfg = None
    best_score = np.inf

    for cand_cfg in candidates:
        model_params = dict(cand_cfg["params"])
        model_params["head_archi"] = list(cand_cfg["params"]["head_archi"])
        model_params["head_dropout"] = float(cand_cfg["params"]["head_dropout"])
        model_params["head_l1l2"] = list(cand_cfg["params"]["head_l1l2"])
        model_params["branch_specs"] = _apply_candidate_to_branch_specs(base_branch_specs, cand_cfg)

        outputs = run_parallel_seeds(
            cand_cfg["model_func"],
            ncpus,
            int(cand_cfg["nmc"]),
            X_model,
            Y_model,
            params=model_params,
            refit=refit,
            dumploc=dumploc,
        )

        this_val = np.array([outputs[k][1] for k in range(int(cand_cfg["nmc"]))], dtype=float)
        score = float(np.nanmin(this_val))

        if score < best_score:
            best_score = score
            best_outputs = outputs
            best_cfg = cand_cfg

    return best_outputs, best_cfg


def run_oos_forecast(X, Y, dates, cfg, base_branch_specs, dumploc, ncpus):
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

        outputs, best_cfg = _select_best_candidate(
            X_model=X_model,
            Y_model=Y_model,
            cfg=cfg,
            base_branch_specs=base_branch_specs,
            dumploc=dumploc,
            ncpus=ncpus,
            refit=refit,
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
                f"elapsed={total_elapsed:8.1f}s | "
                f"ya={best_cfg['branch_config']['yield']['archi']} | "
                f"ma={best_cfg['branch_config']['macro']['archi']} | "
                f"ha={best_cfg['params']['head_archi']} | "
                f"yd={best_cfg['branch_config']['yield']['dropout']} | "
                f"md={best_cfg['branch_config']['macro']['dropout']} | "
                f"hd={best_cfg['params']['head_dropout']} | "
                f"ylr={best_cfg['branch_config']['yield']['l1l2']} | "
                f"mlr={best_cfg['branch_config']['macro']['l1l2']} | "
                f"hlr={best_cfg['params']['head_l1l2']}"
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
    model_params["head_archi"] = list(cfg["params"]["head_archi"])
    model_params["head_dropout"] = cfg["params"]["head_dropout"]
    model_params["head_l1l2"] = cfg["params"]["head_l1l2"]

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
            base_branch_specs=branch_specs,
            dumploc=dumploc,
            ncpus=ncpus,
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

        cfg["branch_config"]["yield"]["archi"] = list(combo["yield_archi"])
        cfg["branch_config"]["macro"]["archi"] = list(combo["macro_archi"])
        cfg["params"]["head_archi"] = list(combo["head_archi"])
        cfg["params"]["learning_rate"] = float(combo["learning_rate"])

        cfg["branch_config"]["yield"]["dropout"] = list(np.asarray(combo["yield_dropout"], dtype=float).ravel())
        cfg["branch_config"]["yield"]["l1l2"] = [list(v) for v in np.asarray(combo["yield_l1l2"], dtype=float).reshape(-1, 2)]

        cfg["branch_config"]["macro"]["dropout"] = list(np.asarray(combo["macro_dropout"], dtype=float).ravel())
        cfg["branch_config"]["macro"]["l1l2"] = [list(v) for v in np.asarray(combo["macro_l1l2"], dtype=float).reshape(-1, 2)]

        cfg["params"]["head_dropout"] = list(np.asarray(combo["head_dropout"], dtype=float).ravel())
        cfg["params"]["head_l1l2"] = [list(v) for v in np.asarray(combo["head_l1l2"], dtype=float).reshape(-1, 2)]

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