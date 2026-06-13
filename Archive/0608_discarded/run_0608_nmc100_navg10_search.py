#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import csv
import multiprocessing
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

import config_0608 as C
import Engine_0608 as Engine


ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "results_v2.0_robustness" / "nmc100_navg10_search_0608"
SUMMARY_CSV = OUT_DIR / "summary.csv"

NMC = int(os.environ.get("NN0608_NMC", "100"))
NAVG = int(os.environ.get("NN0608_NAVG", "10"))
NCPUS = int(os.environ.get("NN0608_NCPUS", "20"))
LOG_FREQ = int(os.environ.get("NN0608_LOG_FREQ", "60"))
USE_POOLED = os.environ.get("NN0608_USE_POOLED", "0") == "1"
TARGET_COLUMN = -1
TARGET_R2 = float(os.environ.get("NN0608_TARGET_R2", "0.0"))
EARLY_ABORT_MIN_STEP = int(os.environ.get("NN0608_EARLY_ABORT_MIN_STEP", "30"))
EARLY_ABORT_R2_THRESHOLD = float(os.environ.get("NN0608_EARLY_ABORT_R2_THRESHOLD", "0.0"))

BASE_PARAMS = {
    "archi": [3],
    "Dropout": 0.0,
    "l1l2": [0.0, 1.0],
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
    "use_bias": False,
    "regularize_output": True,
}


PRIORITY_CANDIDATES = [
    "dy9target_a1_lr0p003_l220_ep1_p0_b32",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32",
    "dy9target_a1_lr0p001_l250_ep2_p0_b32",
    "dy9target_a1_lr0p003_l250_ep2_p0_b32",
    "dy9target_a1_lr0p003_l220_ep2_p0_b32",
    "dy9target_a1_lr0p003_l210_ep3_p0_b32",
    "dy9target_a1_lr0p003_l250_ep2_p0_b32_bias1_outreg1",
    "dy9target_a1_lr0p003_l250_ep2_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l250_ep2_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg1",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg1_hf12",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf12",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg1_hf12",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf12",
    "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg1_hf12",
    "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf12",
    "dy9target_a1_lr0p0003_l2100_ep1_p0_b32_bias1_outreg0_hf12",
    "dy9target_a1_lr0p001_l2200_ep1_p0_b32_bias1_outreg0_hf12",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf6_val30",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf6_val30",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf6_huber0p5",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf6_huber0p5",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5_loose",
    "dy9target_a1_lr0p0005_l250_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5_loose",
    "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5_loose",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p25_loose",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p25_loose",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf5_val30_huber0p25_loose",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf5_val30_huber0p25_loose",
    "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf5_val30_huber0p5_loose",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf10_val30_huber0p25_loose",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf10_val30_huber0p25_loose",
    "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf10_val30_huber0p5_loose",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf6",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf6",
    "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf6",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf3",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf3",
    "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf3",
    "dy9target_a1_lr0p003_l250_ep1_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p003_l25_ep3_p0_b32",
    "dy9target_a1_lr0p003_l21_ep5_p0_b32",
    "dy9target_a1_lr0p001_l210_ep5_p0_b32",
    "dy9target_a1_lr0p001_l25_ep5_p0_b32",
    "dy9target_a1_lr0p003_l250_ep2_p0_b32_bias1_outreg1",
    "dy9target_a1_lr0p003_l250_ep2_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l250_ep2_p0_b32_bias1_outreg0",
    "dy9target_a3_lr0p001_l210_ep5_p0_b32",
    "dy9target_a3_lr0p003_l210_ep5_p0_b32",
    "dy9target_a1_lr0p0005_l250_ep1_p0_b32",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg1",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p003_l220_ep1_p0_b32_bias1_outreg1",
    "dy9target_a1_lr0p003_l220_ep1_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l250_huber0p5_ep1_p0_b32",
    "dy9target_a1_lr0p001_l250_ep2_p0_b32_bias1_outreg1",
    "dy9target_a1_lr0p0003_l250_ep1_p0_b32",
    "dy9target_a1_lr0p001_l220_ep1_p0_b32",
    "dy9target_a3_lr0p003_l220_ep1_p0_b32",
    "dy9target_a3_lr0p001_l250_ep1_p0_b32",
    "dy9loss_a1_lr0p003_l220_ep1_p0_b32",
    "dy9loss_a3_lr0p003_l220_ep1_p0_b32",
    "dy9target_a3_lr0p003_l210_ep5_p1_b256",
    "dy9target_a1_lr0p003_l210_ep5_p1_b256",
    "dy9target_a3_lr0p001_l210_ep5_p1_b256",
    "dy9target_a1_lr0p001_l210_ep5_p1_b256",
    "dy9target_a3_lr0p0003_l210_ep3_p0_b256",
    "dy9target_a1_lr0p0003_l210_ep3_p0_b256",
    "dy9loss_a3_lr0p003_l210_ep5_p1_b256",
    "dy9loss_a1_lr0p003_l210_ep5_p1_b256",
    "dy9target_a3_lr0p001_l21_ep20_p3",
    "dy9target_a3_lr0p0005_l22_ep10_p2",
    "dy9target_a1_lr0p0005_l22_ep10_p2",
    "dy9loss_a3_lr0p001_l21_ep20_p3",
    "dy9loss_a3_lr0p0005_l22_ep10_p2",
    "dy9loss_a1_lr0p0005_l22_ep10_p2",
]


SKIP_CANDIDATES = {
    "dy9target_a3_lr0p003_l220_ep1_p0_b32",
    "dy9target_a1_lr0p003_l250_ep2_p0_b32",
    "dy9target_a1_lr0p003_l220_ep2_p0_b32",
    "dy9target_a1_lr0p003_l210_ep3_p0_b32",
    "dy9target_a1_lr0p003_l250_ep2_p0_b32_bias1_outreg1",
    "dy9target_a1_lr0p003_l250_ep2_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l250_ep2_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg1",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg1_hf12",
    "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf12",
    "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg1_hf12",
    "dy9target_a1_lr0p003_l250_ep1_p0_b32_bias1_outreg0",
    "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf6",
    "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf6",
}


CANDIDATES = [
    (
        "dy9target_a1_lr0p003_l220_ep1_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 20.0],
            "learning_rate": 0.003,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a3_lr0p003_l220_ep1_p0_b32",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 20.0],
            "learning_rate": 0.003,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l250_ep1_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l250_ep2_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 2,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l250_ep2_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.003,
            "epochs": 2,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l220_ep2_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 20.0],
            "learning_rate": 0.003,
            "epochs": 2,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l210_ep3_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.003,
            "epochs": 3,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l25_ep3_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 5.0],
            "learning_rate": 0.003,
            "epochs": 3,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l21_ep5_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.003,
            "epochs": 5,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l210_ep5_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.001,
            "epochs": 5,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l25_ep5_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 5.0],
            "learning_rate": 0.001,
            "epochs": 5,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l250_ep2_p0_b32_bias1_outreg1",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.003,
            "epochs": 2,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": True,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l250_ep2_p0_b32_bias1_outreg0",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.003,
            "epochs": 2,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l250_ep2_p0_b32_bias1_outreg0",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 2,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a3_lr0p001_l210_ep5_p0_b32",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.001,
            "epochs": 5,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a3_lr0p003_l210_ep5_p0_b32",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.003,
            "epochs": 5,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p0005_l250_ep1_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.0005,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l2100_ep1_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 100.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg1",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": True,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg1_hf12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": True,
        },
        {"hyper_freq": 12, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 12, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg1_hf12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 100.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": True,
        },
        {"hyper_freq": 12, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 100.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 12, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg1_hf12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 100.0],
            "learning_rate": 0.0005,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": True,
        },
        {"hyper_freq": 12, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 100.0],
            "learning_rate": 0.0005,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 12, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p0003_l2100_ep1_p0_b32_bias1_outreg0_hf12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 100.0],
            "learning_rate": 0.0003,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 12, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l2200_ep1_p0_b32_bias1_outreg0_hf12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 200.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 12, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l250_ep1_p0_b32_bias1_outreg0",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.003,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 100.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l220_ep1_p0_b32_bias1_outreg1",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 20.0],
            "learning_rate": 0.003,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": True,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l220_ep1_p0_b32_bias1_outreg0",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 20.0],
            "learning_rate": 0.003,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l250_huber0p5_ep1_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "loss_name": "huber",
            "huber_delta": 0.5,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l250_ep2_p0_b32_bias1_outreg1",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 2,
            "patience": 0,
            "batch_size": 32,
            "use_bias": True,
            "regularize_output": True,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p0003_l250_ep1_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.0003,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l220_ep1_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 20.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a3_lr0p001_l250_ep1_p0_b32",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 50.0],
            "learning_rate": 0.001,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9loss_a1_lr0p003_l220_ep1_p0_b32",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 20.0],
            "learning_rate": 0.003,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "target_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "dy9loss_a3_lr0p003_l220_ep1_p0_b32",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 20.0],
            "learning_rate": 0.003,
            "epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "target_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "dy9target_a3_lr0p003_l210_ep5_p1_b256",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.003,
            "epochs": 5,
            "patience": 1,
            "batch_size": 256,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p003_l210_ep5_p1_b256",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.003,
            "epochs": 5,
            "patience": 1,
            "batch_size": 256,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a3_lr0p001_l210_ep5_p1_b256",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.001,
            "epochs": 5,
            "patience": 1,
            "batch_size": 256,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p001_l210_ep5_p1_b256",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.001,
            "epochs": 5,
            "patience": 1,
            "batch_size": 256,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a3_lr0p0003_l210_ep3_p0_b256",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.0003,
            "epochs": 3,
            "patience": 0,
            "batch_size": 256,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p0003_l210_ep3_p0_b256",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.0003,
            "epochs": 3,
            "patience": 0,
            "batch_size": 256,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9loss_a3_lr0p003_l210_ep5_p1_b256",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.003,
            "epochs": 5,
            "patience": 1,
            "batch_size": 256,
            "target_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "dy9loss_a1_lr0p003_l210_ep5_p1_b256",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 10.0],
            "learning_rate": 0.003,
            "epochs": 5,
            "patience": 1,
            "batch_size": 256,
            "target_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "baseline_a3_lr0p003_l21",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.003,
        },
        {"hyper_freq": 60},
    ),
    (
        "a3_lr0p001_l21_ep20_p3_twlong8",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.001,
            "epochs": 20,
            "patience": 3,
            "target_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 8.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "a1_lr0p001_l21_ep20_p3_twlong8",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.001,
            "epochs": 20,
            "patience": 3,
            "target_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 8.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "dy9target_a3_lr0p001_l21_ep20_p3",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.001,
            "epochs": 20,
            "patience": 3,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a3_lr0p0005_l22_ep10_p2",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 2.0],
            "learning_rate": 0.0005,
            "epochs": 10,
            "patience": 2,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9target_a1_lr0p0005_l22_ep10_p2",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 2.0],
            "learning_rate": 0.0005,
            "epochs": 10,
            "patience": 2,
        },
        {"hyper_freq": 60, "target_indices": [8]},
    ),
    (
        "dy9loss_a3_lr0p001_l21_ep20_p3",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.001,
            "epochs": 20,
            "patience": 3,
            "target_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "dy9loss_a3_lr0p0005_l22_ep10_p2",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 2.0],
            "learning_rate": 0.0005,
            "epochs": 10,
            "patience": 2,
            "target_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "dy9loss_a1_lr0p0005_l22_ep10_p2",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 2.0],
            "learning_rate": 0.0005,
            "epochs": 10,
            "patience": 2,
            "target_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "a3_lr0p0005_l21_ep20_p3_twlong12",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.0005,
            "epochs": 20,
            "patience": 3,
            "target_weights": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 12.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "a1_lr0p0005_l21_ep20_p3_twlong12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.0005,
            "epochs": 20,
            "patience": 3,
            "target_weights": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 12.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "a1_lr0p001_l20p1_ep20_p3_twlong12",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 0.1],
            "learning_rate": 0.001,
            "epochs": 20,
            "patience": 3,
            "target_weights": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 12.0],
        },
        {"hyper_freq": 60},
    ),
    (
        "a1_lr0p001_l21_ep20_p3_twlong8_bias1_outreg1",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.001,
            "epochs": 20,
            "patience": 3,
            "target_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 8.0],
            "use_bias": True,
            "regularize_output": True,
        },
        {"hyper_freq": 60},
    ),
    (
        "a1_lr0p001_l21_ep20_p3_twlong8_bias1_outreg0",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.001,
            "epochs": 20,
            "patience": 3,
            "target_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 8.0],
            "use_bias": True,
            "regularize_output": False,
        },
        {"hyper_freq": 60},
    ),
    (
        "a3_lr0p002_l21",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.002,
        },
        {"hyper_freq": 60},
    ),
    (
        "a3_lr0p001_l21",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.001,
        },
        {"hyper_freq": 60},
    ),
    (
        "a1_lr0p001_l21",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.001,
        },
        {"hyper_freq": 60},
    ),
    (
        "a1_lr0p003_l21",
        {
            "archi": [1],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.003,
        },
        {"hyper_freq": 60},
    ),
    (
        "a3_lr0p003_l20p5",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 0.5],
            "learning_rate": 0.003,
        },
        {"hyper_freq": 60},
    ),
    (
        "a3_lr0p003_l22",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 2.0],
            "learning_rate": 0.003,
        },
        {"hyper_freq": 60},
    ),
    (
        "a3_lr0p003_l21_hf120",
        {
            "archi": [3],
            "Dropout": 0.0,
            "l1l2": [0.0, 1.0],
            "learning_rate": 0.003,
        },
        {"hyper_freq": 120},
    ),
]


FAST_REFIT_CANDIDATES = [
    ("dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf6", 6, 50.0, 0.001),
    ("dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf6", 6, 100.0, 0.001),
    ("dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf6", 6, 100.0, 0.0005),
    ("dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf3", 3, 50.0, 0.001),
    ("dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf3", 3, 100.0, 0.001),
    ("dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf3", 3, 100.0, 0.0005),
]

for _name, _hyper_freq, _l2, _learning_rate in FAST_REFIT_CANDIDATES:
    CANDIDATES.append(
        (
            _name,
            {
                "archi": [1],
                "Dropout": 0.0,
                "l1l2": [0.0, _l2],
                "learning_rate": _learning_rate,
                "epochs": 1,
                "patience": 0,
                "batch_size": 32,
                "use_bias": True,
                "regularize_output": False,
            },
            {"hyper_freq": _hyper_freq, "target_indices": [8]},
        )
    )


STABILIZED_CANDIDATES = [
    (
        "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf6_val30",
        6,
        50.0,
        0.001,
        0.30,
        12,
        "mse",
        1.0,
    ),
    (
        "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf6_val30",
        6,
        100.0,
        0.001,
        0.30,
        12,
        "mse",
        1.0,
    ),
    (
        "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf6_huber0p5",
        6,
        50.0,
        0.001,
        0.15,
        12,
        "huber",
        0.5,
    ),
    (
        "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf6_huber0p5",
        6,
        100.0,
        0.001,
        0.15,
        12,
        "huber",
        0.5,
    ),
    (
        "dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5",
        3,
        50.0,
        0.001,
        0.30,
        12,
        "huber",
        0.5,
    ),
    (
        "dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5",
        3,
        100.0,
        0.001,
        0.30,
        12,
        "huber",
        0.5,
    ),
]

for (
    _name,
    _hyper_freq,
    _l2,
    _learning_rate,
    _validation_split,
    _purge_size,
    _loss_name,
    _huber_delta,
) in STABILIZED_CANDIDATES:
    CANDIDATES.append(
        (
            _name,
            {
                "archi": [1],
                "Dropout": 0.0,
                "l1l2": [0.0, _l2],
                "learning_rate": _learning_rate,
                "epochs": 1,
                "patience": 0,
                "batch_size": 32,
                "validation_split": _validation_split,
                "purge_size": _purge_size,
                "loss_name": _loss_name,
                "huber_delta": _huber_delta,
                "use_bias": True,
                "regularize_output": False,
            },
            {"hyper_freq": _hyper_freq, "target_indices": [8]},
        )
    )


REFINED_HUBER_CANDIDATES = [
    ("dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5_loose", 3, 100.0, 0.001, 0.5),
    ("dy9target_a1_lr0p0005_l250_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5_loose", 3, 50.0, 0.0005, 0.5),
    ("dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p5_loose", 3, 100.0, 0.0005, 0.5),
    ("dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p25_loose", 3, 50.0, 0.001, 0.25),
    ("dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf3_val30_huber0p25_loose", 3, 100.0, 0.001, 0.25),
    ("dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf5_val30_huber0p25_loose", 5, 50.0, 0.001, 0.25),
    ("dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf5_val30_huber0p25_loose", 5, 100.0, 0.001, 0.25),
    ("dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf5_val30_huber0p5_loose", 5, 100.0, 0.0005, 0.5),
    ("dy9target_a1_lr0p001_l250_ep1_p0_b32_bias1_outreg0_hf10_val30_huber0p25_loose", 10, 50.0, 0.001, 0.25),
    ("dy9target_a1_lr0p001_l2100_ep1_p0_b32_bias1_outreg0_hf10_val30_huber0p25_loose", 10, 100.0, 0.001, 0.25),
    ("dy9target_a1_lr0p0005_l2100_ep1_p0_b32_bias1_outreg0_hf10_val30_huber0p5_loose", 10, 100.0, 0.0005, 0.5),
]

for _name, _hyper_freq, _l2, _learning_rate, _huber_delta in REFINED_HUBER_CANDIDATES:
    CANDIDATES.append(
        (
            _name,
            {
                "archi": [1],
                "Dropout": 0.0,
                "l1l2": [0.0, _l2],
                "learning_rate": _learning_rate,
                "epochs": 1,
                "patience": 0,
                "batch_size": 32,
                "validation_split": 0.30,
                "purge_size": 12,
                "loss_name": "huber",
                "huber_delta": _huber_delta,
                "use_bias": True,
                "regularize_output": False,
            },
            {
                "hyper_freq": _hyper_freq,
                "target_indices": [8],
                "early_abort_r2_threshold": -0.02,
            },
        )
    )


def iter_ordered_candidates():
    by_name = {name: (name, param_overrides, config_overrides)
               for name, param_overrides, config_overrides in CANDIDATES}
    yielded = set()

    for name in PRIORITY_CANDIDATES:
        if name in yielded:
            continue

        candidate = by_name.get(name)
        if candidate is not None:
            yielded.add(name)
            if name not in SKIP_CANDIDATES:
                yield candidate

    for name, param_overrides, config_overrides in CANDIDATES:
        if name not in yielded and name not in SKIP_CANDIDATES:
            yield name, param_overrides, config_overrides


_POOL = None
_POOL_WORKERS = None


def pooled_run_seed_ensemble(model_name, ncpus, nmc, X_model, Y_model, params, dumploc, refit):
    global _POOL, _POOL_WORKERS

    nmc = int(nmc)
    n_workers = min(int(ncpus), nmc)

    jobs = [
        (model_name, seed, X_model, Y_model, params, bool(refit), dumploc)
        for seed in range(nmc)
    ]

    if n_workers <= 1:
        results = [Engine.run_one_seed_job(job) for job in jobs]
    else:
        if _POOL is None or _POOL_WORKERS != n_workers:
            close_pool()
            ctx = multiprocessing.get_context("spawn")
            _POOL = ctx.Pool(processes=n_workers)
            _POOL_WORKERS = n_workers

        results = _POOL.map(Engine.run_one_seed_job, jobs)

    return {seed: results[seed] for seed in range(nmc)}


def close_pool():
    global _POOL, _POOL_WORKERS

    if _POOL is not None:
        _POOL.close()
        _POOL.join()

    _POOL = None
    _POOL_WORKERS = None


def configure_candidate(name, param_overrides, config_overrides):
    params = copy.deepcopy(BASE_PARAMS)
    params.update(param_overrides)
    params.setdefault("use_bias", False)
    params.setdefault("regularize_output", True)

    C.oos_start = "1980-01-31"
    C.target_indices = config_overrides.get("target_indices", None)
    C.nmc = NMC
    C.navg = NAVG
    C.ncpus = NCPUS
    C.log_freq = LOG_FREQ
    C.model = "NN"
    C.params = params
    C.hyper_freq = int(config_overrides.get("hyper_freq", 60))
    C.early_abort_min_step = int(config_overrides.get("early_abort_min_step", EARLY_ABORT_MIN_STEP))
    C.early_abort_r2_threshold = config_overrides.get(
        "early_abort_r2_threshold",
        EARLY_ABORT_R2_THRESHOLD,
    )
    C.early_abort_column = TARGET_COLUMN
    C.run_tag = f"nmc100_navg10_0608_{name}"
    C.out_file = str(OUT_DIR / f"{name}.mat")


def read_r2(file_path):
    data = sio.loadmat(file_path)
    return np.ravel(data["R2OOS"]).astype(float)


def append_summary(name, r2, params, hyper_freq, out_file):
    row = {
        "name": name,
        "dy9_r2": float(r2[TARGET_COLUMN]),
        "mean_r2": float(np.nanmean(r2)),
        "min_r2": float(np.nanmin(r2)),
        "all_positive": bool(np.all(r2 > 0.0)),
        "r2_vector": " ".join(f"{x:.12f}" for x in r2),
        "hyper_freq": int(hyper_freq),
        "params": repr(params),
        "out_file": str(out_file),
    }

    write_header = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summary_has_candidate(name):
    if not SUMMARY_CSV.exists():
        return False

    with SUMMARY_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return any(row.get("name") == name for row in reader)


def run_search():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if USE_POOLED:
        Engine.run_seed_ensemble = pooled_run_seed_ensemble

    try:
        for name, param_overrides, config_overrides in iter_ordered_candidates():
            configure_candidate(name, param_overrides, config_overrides)
            out_file = Path(C.out_file)

            print("\n" + "#" * 72, flush=True)
            print(f"Candidate: {name}", flush=True)
            print(f"Output   : {out_file}", flush=True)
            print("#" * 72, flush=True)

            if out_file.exists():
                r2 = read_r2(out_file)
                print("Existing result found; using saved MAT.", flush=True)
            else:
                Engine.run(C)
                r2 = read_r2(out_file)

            if not summary_has_candidate(name):
                params_record = copy.deepcopy(C.params)
                params_record["target_indices"] = C.target_indices
                append_summary(name, r2, params_record, C.hyper_freq, out_file)

            dy9_r2 = float(r2[TARGET_COLUMN])
            print(f"Candidate R2OOS: {np.array2string(r2, precision=12)}", flush=True)
            print(f"dy_9 R2OOS     : {dy9_r2:.12f}", flush=True)

            if dy9_r2 >= TARGET_R2:
                print(
                    f"STOP: dy_9 reached target {TARGET_R2:.12f} for {name}.",
                    flush=True,
                )
                return 0

            print(
                f"dy_9 is below target {TARGET_R2:.12f}; "
                "continuing to next hyperparameter set.",
                flush=True,
            )

        print("No listed candidate produced positive dy_9. Add more candidates and rerun.", flush=True)
        return 1
    finally:
        if USE_POOLED:
            close_pool()


if __name__ == "__main__":
    raise SystemExit(run_search())
