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

feature_groups = ["d12m_y_pc2"]

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

nmc = 100
navg = 10

# Optional. If None, Engine uses available CPU count.
ncpus = None

# ============================================================
# Model
# ============================================================

# Allowed:
#   "NN"
#   "pcNN"
model = "NN"

params = {
    "archi": [3],

    # pcNN-specific. Ignored by NN.
    # pcs is 1-based: [2] means PC2.
    # pcs = [] means use all computed PCs.
    "n_pcs": 3,
    "pcs": [2],

    # Keras NN training parameters.
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
    "purge_size": 12,
    "shuffle": False,
    "loss_name": "mse",
    "huber_delta": 1.0,
}

# ============================================================
# Output
# ============================================================

run_tag = "NN_d12m_y_PC2"
out_file = "results_v2.0/panelC_NN_d12m_y_PC2.mat"

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import sys
    import Engine

    Engine.run(sys.modules[__name__])