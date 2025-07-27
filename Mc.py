#!/usr/bin/env python3
"""
run_models_no_dnn.py  ─ period-agnostic runner (NON-DNN)
=======================================================

· Linear / penalised / dimension-reduced / tree-based models  
· **No `--period` argument** – `DATA_ROOT` 안의 CSV 파일만 있으면 바로 실행

Typical usage
-------------
$ python run_models_no_dnn.py                      # CPU, default tag
$ python run_models_no_dnn.py --tag MYRUN          # custom tag in filenames
$ python run_models_no_dnn.py --out ./foo.csv      # explicit CSV path

Author : Seungui Hong · 2025-07-08
"""
# ─────────────────────────────────────────────────────────────── imports ──
import argparse, os, sys, warnings, joblib, datetime as dt
import pandas as pd
warnings.filterwarnings("ignore")

from rolling_framework import Machine                 # in-house wrapper

# ───────────────────────────────────────────────────── user configuration ──
DATA_ROOT      = "/Users/ethan_hong/Dropbox/0_Lab_504/Codes/504_ML/LABORATORY/data"   # CSV directory
OUTPUT_ROOT    = "/Users/ethan_hong/Dropbox/0_Lab_504/Codes/504_ML/LABORATORY/code_output" # results + pkl sub-dir
RESULTS_TAG = "nonDNN"                # default filename tag

BURN_IN_START = "197108"
BURN_IN_END   = "198009"
PERIOD_START  = "199009"
PERIOD_END    = "202312"
OFFSET        = 12                    # months ahead

PORTFOLIO_WEIGHTS = (
    pd.Series({"xr_2": 2, "xr_5": 5, "xr_7": 7, "xr_10": 10}, name="w")
    .pipe(lambda s: s / s.sum())
)
COLS_DW = PORTFOLIO_WEIGHTS.index.tolist()

# ───────────────────────────────────────────── predictor blocks & grids ──
PREDICTOR_SETS = [
    ("SL", lambda d: d["SL"][["slope"]]),
    ("CP", lambda d: d["CP"]),
    ("F6", lambda d: d["F6"]),
]

param_grid_rf = {
    "model__estimator__n_estimators":      [300],
    "model__estimator__max_depth":         [2, 8],
    "model__estimator__min_samples_split": [2, 4],
    "model__estimator__min_samples_leaf":  [1, 2, 4],
    "model__estimator__max_features":      [0.25, 0.5, 1.0],
}
param_grid_et = {
    "model__estimator__n_estimators":      [300],
    "model__estimator__max_depth":         [None, 8],
    "model__estimator__min_samples_split": [2, 4],
    "model__estimator__min_samples_leaf":  [1, 2, 4],
    "model__estimator__max_features":      [0.25, 0.5, 1.0],
    "model__estimator__bootstrap":         [False],
}
param_grid_xgb = {
    # "model__estimator__tree_method": ["gpu_hist"],
    "model__estimator__n_estimators":  [300],
    "model__estimator__max_depth":     [2, 4],
    "model__estimator__learning_rate": [0.01],
    "model__estimator__subsample":     [0.7, 0.5],
    "model__estimator__reg_lambda":    [0.1, 1.0],
}
param_grid_lasso      = {"reg__alpha": [0.001, 1.0]}
param_grid_ridge      = {"reg__alpha": [0.001, 1.0]}
param_grid_elasticnet = {"reg__alpha": [0.01, 0.1, 1, 10],
                         "reg__l1_ratio": [0.1, 0.3, 0.5]}

MODEL_SPECS = [
    ("OLS",        dict(model_type="OLS",       option=None,          params=None)),
    ("Ridge",      dict(model_type="Penalized", option="ridge",       params=param_grid_ridge)),
    ("Lasso",      dict(model_type="Penalized", option="lasso",       params=param_grid_lasso)),
    ("ElasticNet", dict(model_type="Penalized", option="elasticnet",  params=param_grid_elasticnet)),
    ("RandomForest",dict(model_type="Tree",     option="RandomForest",params=param_grid_rf)),
    ("ExtraTrees", dict(model_type="Tree",     option="ExtraTrees",  params=param_grid_et)),
    ("XGBoost",    dict(model_type="Tree",     option="XGB",         params=param_grid_xgb)),
]

# ───────────────────────────────────────────────────────── data loading ──
def load_data() -> tuple[pd.DataFrame, dict]:
    """Read CSVs directly from DATA_ROOT (no per-period sub-folder)."""
    try:
        y = pd.read_csv(f"{DATA_ROOT}/exrets.csv", index_col="Time")[
            ["xr_2", "xr_5", "xr_7", "xr_10"]
        ]
        data = {
            "FWDS":  pd.read_csv(f"{DATA_ROOT}/fwds.csv",         index_col="Time"),
            "MACV":  pd.read_csv(f"{DATA_ROOT}/MacroFactors.csv", index_col="Time"),
            "LSC":   pd.read_csv(f"{DATA_ROOT}/lsc.csv",          index_col="Time"),
            # "RVOL":  pd.read_csv(f"{DATA_ROOT}/real_vol.csv",     index_col="Time"),
            "CP":    pd.read_csv(f"{DATA_ROOT}/cp.csv",           index_col="Time"),
        }
    except FileNotFoundError as e:
        sys.exit(f"[ERROR] CSV 파일을 찾을 수 없습니다 → {e.filename}\n"
                 f"DATA_ROOT 경로를 확인하세요.")

    # derived groups
    data["F6"] = data["MACV"][["F1", "F2", "F3", "F4", "F8", "F1^3"]]
    data["SL"] = data["LSC"][["slope"]]
    return y, data

# ─────────────────────────────────────────────── training / metric row ──
def run_single_model(X, y, tag, spec):
    m = Machine(
        X, y,
        spec["model_type"],
        option           = spec["option"],
        params_grid      = spec["params"],
        burn_in_start    = BURN_IN_START,
        burn_in_end      = BURN_IN_END,
        period           = [PERIOD_START, PERIOD_END],
        forecast_horizon = OFFSET,
    )
    m.training()

    # ── (A) 모델 저장 : last_model_이 있을 때만 --------------------------
    model_obj = getattr(m.strategy, "last_model_", None)
    if model_obj is not None:
        pkl_dir = os.path.join(OUTPUT_ROOT, "pkl")
        os.makedirs(pkl_dir, exist_ok=True)
        joblib.dump(model_obj, os.path.join(pkl_dir, f"{tag}.pkl"))
    else:
        print(f"[WARN] {tag} : 'last_model_' 속성이 없어 .pkl 저장을 건너뜀")

    # ── (B) 메트릭 수집 ----------------------------------------------------
    r2 = m.R2OOS()
    return {
        "model"     : tag,
        "r2_xr_2"   : r2["xr_2"],
        "r2_xr_5"   : r2["xr_5"],
        "r2_xr_7"   : r2["xr_7"],
        "r2_xr_10"  : r2["xr_10"],
        "port_R2_EW": m.r2_oos_portfolio(),
        "port_R2_DW": m.r2_oos_portfolio(cols=COLS_DW,
                                         weights=PORTFOLIO_WEIGHTS),
    }
# ──────────────────────────────────────────────────────────────── main ──
def main() -> None:
    ap = argparse.ArgumentParser("Run NON-DNN models & export CSV/PKL")
    ap.add_argument("--tag", default=RESULTS_TAG,
                    help="tag appended to filenames (default=nonDNN)")
    ap.add_argument("--out", default=None,
                    help="explicit CSV path (overrides default naming)")
    args, _ = ap.parse_known_args()          # ignore Jupyter-injected flags

    # ensure dirs
    os.makedirs(OUTPUT_ROOT,            exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "pkl"), exist_ok=True)

    # load once
    y, data_blocks = load_data()

    rows = []
    for set_name, extractor in PREDICTOR_SETS:
        X = extractor(data_blocks)
        for model_code, spec in MODEL_SPECS:
            tag = f"{model_code}-{set_name}_{args.tag}"
            print(f"▶ {tag}")
            rows.append(run_single_model(X, y, tag, spec))

    res = pd.DataFrame(rows)

    # final CSV name
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = args.out or os.path.join(
        OUTPUT_ROOT, f"results_{args.tag}_{timestamp}.csv"
    )
    res.to_csv(csv_path, index=False)
    print(f"\n★ Saved {len(res)} rows → {csv_path}")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()