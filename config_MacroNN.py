#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# Data
# ============================================================

data_path = "data/target_and_features.mat"

# BBT (2021) Group Ensemble: each macro category trains its own NN.
feature_groups = [
    "macro_output",
    "macro_labor",
    "macro_housing",
    "macro_orders",
    "macro_money",
    "macro_ratesfx",
    "macro_prices",
    "macro_stock",
    "fwd"
]

target_group = "rx"
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

# Optional. If None, Engine uses available CPU count.
ncpus = None

# ============================================================
# Model
# ============================================================

# BBT (2021) Group Ensemble Neural Network.
# Trains one shallow NN per feature group, then averages predictions.
model = "MacroNN"

params = {
    "archi": [3],

    # Keras NN training parameters (applied to each group NN).
    "Dropout": [0.0, 0.2],
    "l1l2": [1, 0.5],
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

    # Group Ensemble aggregation:
    # Deprecated – the unified model's output layer now learns optimal weights automatically.
    # "aggregation": "mean",
}

# ============================================================
# Output
# ============================================================

run_tag = "MacroNN_macro8_dy"
out_file = "results_v2.0/MacroNN_macro8_dy-v2.mat"

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import sys
    import Engine

    Engine.run(sys.modules[__name__])
