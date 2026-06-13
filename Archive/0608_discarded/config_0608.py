#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ROOT_DIR = Path(__file__).resolve().parent

# ============================================================
# Data
# ============================================================

data_path = str(ROOT_DIR / "data" / "target_and_features.mat")

feature_groups = ["yoy_fwd"]

target_group = "dy"
target_indices = None

# ============================================================
# Forecasting design
# ============================================================

horizon = 12
oos_start = "1979-01-31"
hyper_freq = 60

# ============================================================
# Monte Carlo / Ensemble
# ============================================================

nmc = 1
navg = 1

ncpus = None
log_freq = 120

# ============================================================
# Model
# ============================================================

model = "NN"

params = {
    "archi": [],
    "Dropout": 0.0,
    "l1l2": [0.0, 0],
    "learning_rate": 0.003,
    "decay_rate": 0.001,
    "momentum": 0.9,
    "nesterov": True,
    "epochs": 100,
    "patience": 6,
    "batch_size": 32,
    "validation_split": 0.15,
    "purge_size": 12,
    "shuffle": False,
    "loss_name": "mse",
    "huber_delta": 1.0,
    "use_bias": True,
    "regularize_output": True,
}

# ============================================================
# Output
# ============================================================

run_tag = "yoy_fwd_NN_a3_lr0p003_l21_oos198001_exact_0608"
out_file = str(ROOT_DIR / "results_v2.0_robustness" / "OLS_0608_yoy_fwd_oos198001.mat")

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import sys
    import Engine_0608 as Engine

    Engine.run(sys.modules[__name__])
