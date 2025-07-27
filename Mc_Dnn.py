#!/usr/bin/env python3
"""
run_models_dnn.py — period-agnostic runner (DNN)
================================================
* Deep-learning forecasts for US Treasury excess-returns
* NO period argument: CSVs are read directly from DATA_ROOT.
"""

import os, sys, argparse, warnings, joblib, datetime as dt
import pandas as pd
import torch
from rolling_framework import Machine          # in-house wrapper
warnings.filterwarnings("ignore")

# ───────────────────────────── user config ──
DATA_ROOT      = "/Users/ethan_hong/Dropbox/0_Lab_504/Codes/504_ML/LABORATORY/data"   # CSV directory
OUTPUT_ROOT    = "/Users/ethan_hong/Dropbox/0_Lab_504/Codes/504_ML/LABORATORY/code_output" # results + pkl sub-dir
RESULTS_TAG = "DNN"                      # default tag in filenames

BURN_IN_START = "197108"
BURN_IN_END   = "198009"
PERIOD_START  = "198101"
PERIOD_END    = "202312"
OFFSET        = 12                       # months ahead

param_grid_single = {
    "dnn__module__hidden":          [(3,)],
    "dnn__module__dropout":         [0.2],
    "dnn__optimizer__lr":           [0.01],
    "dnn__optimizer__weight_decay": [0.0005],
}
param_grid_dual = {
    "dnn__module__hidden1":         [(32,)],
    "dnn__module__drop1":           [0.2],
    "dnn__module__hidden2":         [(3, 3)],
    "dnn__module__drop2":           [0.2],
    "dnn__optimizer__lr":           [0.001],
    "dnn__optimizer__weight_decay": [0.0005],
}

PORTFOLIO_WEIGHTS = (
    pd.Series({"xr_2": 2, "xr_5": 5, "xr_7": 7, "xr_10": 10}, name="w")
      .pipe(lambda s: s / s.sum())
)
COLS_DW = PORTFOLIO_WEIGHTS.index.tolist()

# ───────────────────────────── data I/O ──
def build_file_paths() -> dict:
    return {
        "exrets": f"{DATA_ROOT}/exrets.csv",
        "fwds":   f"{DATA_ROOT}/fwds.csv",
        "macro":  f"{DATA_ROOT}/MacroFactors.csv",
        "lsc":    f"{DATA_ROOT}/lsc.csv",
        "cp":     f"{DATA_ROOT}/cp.csv",
    }

def load_raw():
    f = build_file_paths()
    try:
        y  = pd.read_csv(f["exrets"], index_col="Time")[["xr_2","xr_5","xr_7","xr_10"]]
        fw = pd.read_csv(f["fwds"],   index_col="Time")
        ma = pd.read_csv(f["macro"],  index_col="Time")
        cp = pd.read_csv(f["cp"],     index_col="Time")
    except FileNotFoundError as e:
        sys.exit(f"[ERROR] CSV not found → {e.filename}\n"
                 "        check DATA_ROOT path.")
    return y, {
        "DNN_single_CP+FWDS":  pd.concat([cp, fw], axis=1),
        "DNN_dual_MACRO+FWDS": pd.concat([ma, fw], axis=1),
    }

# ───────────────────────────── model list ──
DNN_SPECS = [
    ("DNN_single_CP+FWDS", dict(model_type="DNN",
                                option=None,
                                params=param_grid_single)),
    ("DNN_dual_MACRO+FWDS", dict(model_type="DNN_DUAL",
                                 option=None,
                                 params=param_grid_dual)),
]

# ───────────────────────────── core run ──
def main() -> None:
    ap = argparse.ArgumentParser("Run DNN models & export CSV/PKL")
    ap.add_argument("--gpu", default="", help="CUDA id (blank=CPU)")
    ap.add_argument("--tag", default=RESULTS_TAG,
                    help="filename tag prefix")
    ap.add_argument("--out", default=None,
                    help="explicit CSV path (overrides auto-naming)")
    args, _ = ap.parse_known_args()          # ignore Jupyter flags

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    _ = torch.cuda.is_available()

    y_raw, X_dict = load_raw()

    os.makedirs(OUTPUT_ROOT,               exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "pkl"), exist_ok=True)

    rows = []
    for tag, spec in DNN_SPECS:
        print(f"\n▶ Training {tag}")
        m = Machine(
            X_dict[tag], y_raw,
            spec["model_type"],
            option           = spec["option"],
            params_grid      = spec["params"],
            burn_in_start    = BURN_IN_START,
            burn_in_end      = BURN_IN_END,
            period           = [PERIOD_START, PERIOD_END],
            forecast_horizon = OFFSET,
        )
        m.training()

        pkl_path = os.path.join(OUTPUT_ROOT, "pkl",
                                f"{tag}_{args.tag}.pkl")
        joblib.dump(m.strategy.last_model_, pkl_path)

        r2 = m.R2OOS()
        rows.append({
            "model":  tag,
            "r2_xr_2": r2["xr_2"], "r2_xr_5": r2["xr_5"],
            "r2_xr_7": r2["xr_7"], "r2_xr_10": r2["xr_10"],
            "ew_port": m.r2_oos_portfolio(),
            "dw_port": m.r2_oos_portfolio(cols=COLS_DW,
                                          weights=PORTFOLIO_WEIGHTS),
        })

    csv_path = args.out or os.path.join(
        OUTPUT_ROOT, f"results_{args.tag}.csv"
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n★ Summary saved → {csv_path}")

# ─────────────────────────────
if __name__ == "__main__":
    main()