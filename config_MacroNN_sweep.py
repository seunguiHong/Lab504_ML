#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import itertools
import json
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# Data
# ============================================================

data_path = "data/target_and_features.mat"

# BBT (2021) Group Ensemble: 8 macro categories.
feature_groups = [
    "macro_output",
    "macro_labor",
    "macro_housing",
    "macro_orders",
    "macro_money",
    "macro_ratesfx",
    "macro_prices",
    "macro_stock",
]

target_group = "dy"
target_indices = None

# ============================================================
# Forecasting design
# ============================================================

horizon = 12
oos_start = "1989-01-31"
hyper_freq = 60

# ============================================================
# Monte Carlo / Ensemble
# ============================================================

nmc = 20
navg = 5

# Always use all available CPU workers.
ncpus = multiprocessing.cpu_count()

# Print progress every N OOS forecast origins.
log_freq = 12

# ============================================================
# Model
# ============================================================

model = "MacroNN"

# ============================================================
# Base parameters
# ============================================================

BASE_PARAMS = {
    "archi": [3],

    # Keras NN training parameters (applied to each group NN).
    "Dropout": [0.0],
    "l1l2": [[0.0, 0.0]],
    "learning_rate": 0.02,
    "decay_rate": 0.001,
    "momentum": 0.9,
    "nesterov": True,
    "epochs": 500,
    "patience": 20,
    "batch_size": 32,
    "validation_split": 0.15,
    "purge_size": 12,
    "shuffle": False,
    "loss_name": "mse",
    "huber_delta": 1.0,

    # Group Ensemble aggregation.
    "aggregation": "mean",
}

# ============================================================
# Sweep grid
# ============================================================
# Engine.py already treats Dropout and l1l2 as candidate grids
# inside one run. This outer sweep evaluates separate experiments
# for training parameters such as architecture, learning rate,
# patience, and the regularization candidate set itself.

SWEEP_GRID = {
    "archi": [
        [3],
        [8],
        [16],
        [8, 4],
    ],

    "learning_rate": [
        0.003,
        0.005,
        0.01,
        0.02,
    ],

    "patience": [
        20,
        40,
    ],

    "batch_size": [
        32,
    ],

    # Passed to Engine as candidate grids inside each run.
    # Engine selects among these by validation loss whenever refit=True.
    "Dropout": [
        [0.0],
        [0.0, 0.1],
        [0.0, 0.2],
    ],

    "l1l2": [
        [[0.0, 0.0]],
        [[1e-5, 1e-5]],
        [[1e-4, 5e-5]],
        [[1e-4, 1e-4]],
    ],
}

# ============================================================
# Output
# ============================================================

output_dir = "results_v2.0_sweep"

base_run_tag = "sweep_MacroNN_macro8_dy_oos1980"

summary_file = os.path.join(output_dir, base_run_tag + "__summary.xlsx")
summary_csv_file = os.path.join(output_dir, base_run_tag + "__summary.csv")
failed_file = os.path.join(output_dir, base_run_tag + "__failed_runs.txt")

# These are overwritten inside apply_case_to_config().
params = copy.deepcopy(BASE_PARAMS)
run_tag = base_run_tag
out_file = os.path.join(output_dir, run_tag + ".mat")


# ============================================================
# Tag helpers
# ============================================================

def compact_float(x):
    x = float(x)

    if x == 0.0:
        return "0"

    if abs(x) < 0.001 or abs(x) >= 1000:
        s = f"{x:.0e}"
    else:
        s = f"{x:g}"

    s = s.replace("+", "")
    s = s.replace("-", "m")
    s = s.replace(".", "p")

    return s


def archi_tag(archi):
    if len(archi) == 0:
        return "OLS"

    return "x".join(str(int(v)) for v in archi)


def dropout_tag(dropout_list):
    vals = [compact_float(v) for v in dropout_list]
    return "do" + "_".join(vals)


def normalize_l1l2_row(row):
    if isinstance(row, (int, float)):
        v = float(row)
        return [v, v]

    arr = list(row)

    if len(arr) == 0:
        return [0.0, 0.0]

    if len(arr) == 1:
        v = float(arr[0])
        return [v, v]

    return [float(arr[0]), float(arr[1])]


def l1l2_tag(l1l2_grid):
    rows = []

    for row in l1l2_grid:
        l1, l2 = normalize_l1l2_row(row)
        rows.append("l1" + compact_float(l1) + "_l2" + compact_float(l2))

    return "__".join(rows)


def case_tag(case):
    parts = [
        f"a{archi_tag(case['archi'])}",
        f"lr{compact_float(case['learning_rate'])}",
        f"pat{int(case['patience'])}",
        f"bs{int(case['batch_size'])}",
        dropout_tag(case["Dropout"]),
        l1l2_tag(case["l1l2"]),
    ]

    return "__".join(parts)


# ============================================================
# Sweep construction
# ============================================================

def build_sweep_cases(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    cases = []

    for combo in itertools.product(*values):
        case = dict(zip(keys, combo))
        cases.append(case)

    return cases


def apply_case_to_config(case):
    global params, run_tag, out_file

    params = copy.deepcopy(BASE_PARAMS)

    for key, value in case.items():
        params[key] = copy.deepcopy(value)

    tag = case_tag(case)

    run_tag = base_run_tag + "__" + tag
    out_file = os.path.join(output_dir, run_tag + ".mat")


# ============================================================
# Summary helpers
# ============================================================

def result_summary_row(case, save_dict, saved_file, run_index, total_runs, status="success", error_message=""):
    import numpy as np

    r2 = np.asarray(save_dict.get("R2OOS", []), dtype=float).reshape(-1)
    pval = np.asarray(save_dict.get("R2OOS_pval", []), dtype=float).reshape(-1)
    mse = np.asarray(save_dict.get("MSE", []), dtype=float).reshape(-1)

    row = {
        "run_index": run_index,
        "total_runs": total_runs,
        "status": status,
        "error_message": error_message,
        "run_tag": run_tag,
        "out_file": saved_file,
        "model": model,
        "feature_groups": json.dumps(feature_groups),
        "target_group": target_group,
        "target_indices": json.dumps(target_indices),
        "oos_start": oos_start,
        "horizon": horizon,
        "hyper_freq": hyper_freq,
        "nmc": nmc,
        "navg": navg,
        "ncpus": ncpus,
        "archi": json.dumps(case["archi"]),
        "learning_rate": case["learning_rate"],
        "patience": case["patience"],
        "batch_size": case["batch_size"],
        "Dropout": json.dumps(case["Dropout"]),
        "l1l2": json.dumps(case["l1l2"]),
        "R2OOS_mean": float(np.nanmean(r2)) if r2.size > 0 else np.nan,
        "R2OOS_median": float(np.nanmedian(r2)) if r2.size > 0 else np.nan,
        "MSE_mean": float(np.nanmean(mse)) if mse.size > 0 else np.nan,
    }

    for j, value in enumerate(r2, start=1):
        row[f"R2OOS_{j}"] = value

    for j, value in enumerate(pval, start=1):
        row[f"pval_{j}"] = value

    for j, value in enumerate(mse, start=1):
        row[f"MSE_{j}"] = value

    return row


def failed_summary_row(case, run_index, total_runs, error_message):
    return {
        "run_index": run_index,
        "total_runs": total_runs,
        "status": "failed",
        "error_message": error_message,
        "run_tag": run_tag,
        "out_file": out_file,
        "model": model,
        "feature_groups": json.dumps(feature_groups),
        "target_group": target_group,
        "target_indices": json.dumps(target_indices),
        "oos_start": oos_start,
        "horizon": horizon,
        "hyper_freq": hyper_freq,
        "nmc": nmc,
        "navg": navg,
        "ncpus": ncpus,
        "archi": json.dumps(case["archi"]),
        "learning_rate": case["learning_rate"],
        "patience": case["patience"],
        "batch_size": case["batch_size"],
        "Dropout": json.dumps(case["Dropout"]),
        "l1l2": json.dumps(case["l1l2"]),
        "R2OOS_mean": None,
        "R2OOS_median": None,
        "MSE_mean": None,
    }


def order_summary_columns(df):
    preferred_cols = [
        "run_index",
        "total_runs",
        "status",
        "error_message",
        "run_tag",
        "out_file",
        "model",
        "feature_groups",
        "target_group",
        "target_indices",
        "oos_start",
        "horizon",
        "hyper_freq",
        "nmc",
        "navg",
        "ncpus",
        "archi",
        "learning_rate",
        "patience",
        "batch_size",
        "Dropout",
        "l1l2",
        "R2OOS_mean",
        "R2OOS_median",
        "MSE_mean",
    ]

    existing_preferred = [c for c in preferred_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_preferred]

    return df[existing_preferred + remaining_cols]


def update_summary_files(summary_rows):
    import pandas as pd

    if not summary_rows:
        return None

    df = pd.DataFrame(summary_rows)
    df = order_summary_columns(df)

    os.makedirs(output_dir, exist_ok=True)

    # CSV is written first because it is robust and can be inspected
    # even if Excel writing fails or if the xlsx file is open elsewhere.
    df.to_csv(summary_csv_file, index=False)

    try:
        df.to_excel(summary_file, index=False)
        return summary_file

    except Exception as e:
        print("\n" + "!" * 72)
        print("Excel summary update failed.")
        print(f"Excel file : {summary_file}")
        print(f"CSV backup : {summary_csv_file}")
        print(f"Error      : {e}")
        print("!" * 72)

        return summary_csv_file


def update_failed_log(failed):
    if not failed:
        if os.path.exists(failed_file):
            os.remove(failed_file)
        return None

    with open(failed_file, "w", encoding="utf-8") as f:
        for tag, err in failed:
            f.write(f"{tag}\n")
            f.write(f"{err}\n\n")

    return failed_file


# ============================================================
# Console output
# ============================================================

def print_sweep_header(cases):
    print("\n" + "=" * 72)
    print("Hyperparameter sweep – MacroNN (BBT Group Ensemble)")
    print("=" * 72)
    print(f"Number of runs : {len(cases)}")
    print(f"Output dir     : {output_dir}")
    print(f"Summary file   : {summary_file}")
    print(f"CSV backup     : {summary_csv_file}")
    print(f"Model          : {model}")
    print(f"Feature groups : {feature_groups}")
    print(f"Target group   : {target_group}")
    print(f"OOS start      : {oos_start}")
    print(f"Horizon        : {horizon}")
    print(f"Hyper frequency: {hyper_freq}")
    print(f"NMC / NAVG     : {nmc} / {navg}")
    print(f"CPU workers    : {ncpus}")
    print("=" * 72 + "\n")


def print_sweep_footer(cases, failed, summary_path, failed_path):
    print("\n" + "=" * 72)
    print("Sweep finished")
    print("=" * 72)
    print(f"Total runs : {len(cases)}")
    print(f"Failed     : {len(failed)}")

    if summary_path is not None:
        print(f"Summary    : {summary_path}")

    if os.path.exists(summary_csv_file):
        print(f"CSV backup : {summary_csv_file}")

    if failed_path is not None:
        print(f"Failed log : {failed_path}")

    if failed:
        print("\nFailed runs:")
        for tag, err in failed:
            print(f"- {tag}: {err}")

    print("=" * 72)


# ============================================================
# Main sweep runner
# ============================================================

def run_sweep():
    import sys
    import Engine

    os.makedirs(output_dir, exist_ok=True)

    cases = build_sweep_cases(SWEEP_GRID)
    total_runs = len(cases)

    print_sweep_header(cases)

    this_module = sys.modules[__name__]

    failed = []
    summary_rows = []
    summary_path = None
    failed_path = None

    for i, case in enumerate(cases, start=1):
        apply_case_to_config(case)

        print("\n" + "#" * 72)
        print(f"Sweep run {i} / {total_runs}")
        print(f"Run tag : {run_tag}")
        print(f"Out file: {out_file}")
        print("#" * 72)

        try:
            save_dict, saved_file = Engine.run(this_module)

            row = result_summary_row(
                case=case,
                save_dict=save_dict,
                saved_file=saved_file,
                run_index=i,
                total_runs=total_runs,
                status="success",
                error_message="",
            )

            summary_rows.append(row)

            # Update summary immediately after each successful MAT save.
            summary_path = update_summary_files(summary_rows)

            print("\n" + "-" * 72)
            print("Summary updated")
            print(f"Rows        : {len(summary_rows)}")
            print(f"Excel/CSV   : {summary_path}")
            print("-" * 72)

        except Exception as e:
            err = str(e)
            failed.append((run_tag, err))

            row = failed_summary_row(
                case=case,
                run_index=i,
                total_runs=total_runs,
                error_message=err,
            )

            summary_rows.append(row)

            # Update summary and failed log immediately after failure.
            summary_path = update_summary_files(summary_rows)
            failed_path = update_failed_log(failed)

            print("\n" + "!" * 72)
            print("Run failed")
            print(f"Run tag : {run_tag}")
            print(f"Error   : {err}")
            print(f"Summary : {summary_path}")
            if failed_path is not None:
                print(f"Failed log: {failed_path}")
            print("!" * 72)

    # Final write, mainly to ensure the last state is flushed.
    summary_path = update_summary_files(summary_rows)
    failed_path = update_failed_log(failed)

    print_sweep_footer(
        cases=cases,
        failed=failed,
        summary_path=summary_path,
        failed_path=failed_path,
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    run_sweep()
