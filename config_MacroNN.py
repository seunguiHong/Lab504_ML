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
# Unified multi-branch network: one input branch per feature group, concatenated
# and trained end-to-end (the output layer learns the group weights).
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
}

# ============================================================
# Output
# ============================================================
#
# Filename is assembled by Engine.build_out_file from the fields below:
#   [Panel<X>_]<target>_<model>_<predictor>_<ensembling>_<regularization>[_<suffix>].mat
# target/model/ensembling/regularization are read from the settings above.
# Leave out_file = None to use the convention. Set it only to override.

results_root = "results"      # output directory
panel = "A"                   # Panel A evaluation (oos_start 1989)
predictor_label = "macro8"    # 8 macro groups + fwd, labelled compactly
name_suffix = None            # optional tag; None to omit

run_tag = "MacroNN_macro8_dy"
out_file = None               # -> results/PanelA_rx_MacroNN_macro8_ens20x5_l11l20p5do0p2.mat

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import sys
    import Engine

    Engine.run(sys.modules[__name__])
