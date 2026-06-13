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

feature_groups = ["yoy_fwd"]

target_group = "dy"
target_indices = None

# ============================================================
# Forecasting design
# ============================================================

horizon = 12
oos_start = "1980-01-31"
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
    "pcs": [],

    # Keras NN training parameters.
    "Dropout": [0.0, 0.2],
    "l1l2": [1,0.5],
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
panel = None                  # "A" / "B" for paper panels; None to omit
predictor_label = None        # None -> derived from feature_groups
name_suffix = "oos1980"       # optional tag; None to omit

run_tag = "Rbst_NN_yoy_fwd"
out_file = None               # -> results/yc_NN_yoy_fwd_ens20x5_l11l20p5do0p2_oos1980.mat

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import sys
    import Engine

    Engine.run(sys.modules[__name__])