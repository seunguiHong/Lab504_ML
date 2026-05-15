#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import random

import numpy as np
import torch
import torch.nn as nn


def prepare_validation(model, X_fit_raw, X_val_raw, X_test_raw, C):
    if model == "rawNN":
        return _prepare_raw_validation(X_fit_raw, X_val_raw, X_test_raw, C)

    if model == "pcNN":
        return _prepare_pc_validation(X_fit_raw, X_val_raw, X_test_raw, C)

    raise ValueError(f"Unknown model: {model}")


def prepare_final(model, X_hist_raw, X_test_raw, C):
    if model == "rawNN":
        return _prepare_raw_final(X_hist_raw, X_test_raw, C)

    if model == "pcNN":
        return _prepare_pc_final(X_hist_raw, X_test_raw, C)

    raise ValueError(f"Unknown model: {model}")


def build_candidates(C):
    if len(C.archi) == 0:
        return [{"archi": []}]

    candidates = []

    for dropout in C.dropout_grid:
        for l1l2 in C.l1l2_grid:
            l1, l2 = _parse_l1l2(l1l2)

            candidates.append(
                {
                    "archi": list(C.archi),
                    "dropout": float(dropout),
                    "l1": float(l1),
                    "l2": float(l2),
                    "learning_rate": float(C.learning_rate),
                    "decay_rate": float(C.decay_rate),
                    "momentum": float(C.momentum),
                    "nesterov": bool(C.nesterov),
                    "epochs": int(C.epochs),
                    "patience": int(C.patience),
                    "batch_size": int(C.batch_size),
                    "shuffle": bool(C.shuffle),
                    "loss": str(C.loss).lower(),
                    "huber_delta": float(C.huber_delta),
                }
            )

    return candidates


def train_validation(X_fit, Y_fit, X_val, Y_val, candidate, seed):
    if len(candidate["archi"]) == 0:
        beta = _fit_ols(X_fit, Y_fit)
        pred = _predict_ols(X_val, beta)
        return float(np.mean((Y_val - pred) ** 2)), 1

    return _train_mlp_validation(X_fit, Y_fit, X_val, Y_val, candidate, seed)


def train_final(X_train, Y_train, X_test, candidate, seed, epochs):
    if len(candidate["archi"]) == 0:
        beta = _fit_ols(X_train, Y_train)
        return _predict_ols(X_test, beta).reshape(-1)

    return _train_mlp_final(X_train, Y_train, X_test, candidate, seed, epochs)


def _prepare_raw_validation(X_fit_raw, X_val_raw, X_test_raw, C):
    X_fit = np.asarray(X_fit_raw, dtype=float)
    X_val = np.asarray(X_val_raw, dtype=float)
    X_test = np.asarray(X_test_raw, dtype=float)

    if C.standardize:
        scaler = _fit_standardizer(X_fit)
        X_fit = _apply_standardizer(X_fit, scaler)
        X_val = _apply_standardizer(X_val, scaler)
        X_test = _apply_standardizer(X_test, scaler)

    return X_fit, X_val, X_test


def _prepare_raw_final(X_hist_raw, X_test_raw, C):
    X_hist = np.asarray(X_hist_raw, dtype=float)
    X_test = np.asarray(X_test_raw, dtype=float)

    if C.standardize:
        scaler = _fit_standardizer(X_hist)
        X_hist = _apply_standardizer(X_hist, scaler)
        X_test = _apply_standardizer(X_test, scaler)

    return X_hist, X_test


def _prepare_pc_validation(X_fit_raw, X_val_raw, X_test_raw, C):
    pca = _fit_pca(X_fit_raw, C.pc_ncomp)

    X_fit = _apply_pca(X_fit_raw, pca)[:, C.pc_keep]
    X_val = _apply_pca(X_val_raw, pca)[:, C.pc_keep]
    X_test = _apply_pca(X_test_raw, pca)[:, C.pc_keep]

    if C.standardize:
        scaler = _fit_standardizer(X_fit)
        X_fit = _apply_standardizer(X_fit, scaler)
        X_val = _apply_standardizer(X_val, scaler)
        X_test = _apply_standardizer(X_test, scaler)

    return X_fit, X_val, X_test


def _prepare_pc_final(X_hist_raw, X_test_raw, C):
    pca = _fit_pca(X_hist_raw, C.pc_ncomp)

    X_hist = _apply_pca(X_hist_raw, pca)[:, C.pc_keep]
    X_test = _apply_pca(X_test_raw, pca)[:, C.pc_keep]

    if C.standardize:
        scaler = _fit_standardizer(X_hist)
        X_hist = _apply_standardizer(X_hist, scaler)
        X_test = _apply_standardizer(X_test, scaler)

    return X_hist, X_test


def _fit_pca(X, n_components):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("PCA input must be a 2D array.")

    mu = np.mean(X, axis=0)
    Xc = X - mu

    _, _, vt = np.linalg.svd(Xc, full_matrices=False)

    n_keep = min(int(n_components), vt.shape[0])
    loadings = vt[:n_keep, :].T

    for j in range(loadings.shape[1]):
        k = int(np.argmax(np.abs(loadings[:, j])))
        if loadings[k, j] < 0:
            loadings[:, j] *= -1.0

    return {"mean": mu, "loadings": loadings}


def _apply_pca(X, pca):
    X = np.asarray(X, dtype=float)
    return (X - pca["mean"]) @ pca["loadings"]


def _fit_standardizer(X):
    X = np.asarray(X, dtype=float)

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd[sd < 1e-12] = 1.0

    return {"mean": mu, "std": sd}


def _apply_standardizer(X, scaler):
    X = np.asarray(X, dtype=float)
    return (X - scaler["mean"]) / scaler["std"]


def _fit_ols(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    X_reg = _add_intercept(X)
    beta, _, _, _ = np.linalg.lstsq(X_reg, Y, rcond=None)

    return beta


def _predict_ols(X, beta):
    X = np.asarray(X, dtype=float)
    return _add_intercept(X) @ beta


def _add_intercept(X):
    return np.column_stack([np.ones(X.shape[0], dtype=float), X])


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, archi, dropout):
        super().__init__()

        layers = []
        prev_dim = int(input_dim)

        for width in archi:
            layers.append(nn.Linear(prev_dim, int(width)))
            layers.append(nn.ReLU())

            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))

            prev_dim = int(width)

        layers.append(nn.Linear(prev_dim, int(output_dim)))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_mlp_validation(X_fit, Y_fit, X_val, Y_val, candidate, seed):
    _set_seed(seed)

    X_fit_t = _tensor(X_fit)
    Y_fit_t = _tensor(Y_fit)
    X_val_t = _tensor(X_val)
    Y_val_t = _tensor(Y_val)

    model = MLP(
        input_dim=X_fit_t.shape[1],
        output_dim=Y_fit_t.shape[1],
        archi=candidate["archi"],
        dropout=candidate["dropout"],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=candidate["learning_rate"],
        momentum=candidate["momentum"],
        nesterov=candidate["nesterov"],
    )

    best_loss = np.inf
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    for epoch in range(1, candidate["epochs"] + 1):
        _set_learning_rate(
            optimizer,
            candidate["learning_rate"],
            candidate["decay_rate"],
            epoch,
        )

        _train_epoch(
            model=model,
            optimizer=optimizer,
            X=X_fit_t,
            Y=Y_fit_t,
            candidate=candidate,
            seed=seed + epoch,
        )

        val_loss = _eval_loss(model, X_val_t, Y_val_t, candidate)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= candidate["patience"]:
            break

    model.load_state_dict(best_state)

    return float(best_loss), int(best_epoch)


def _train_mlp_final(X_train, Y_train, X_test, candidate, seed, epochs):
    _set_seed(seed)

    X_train_t = _tensor(X_train)
    Y_train_t = _tensor(Y_train)
    X_test_t = _tensor(X_test)

    model = MLP(
        input_dim=X_train_t.shape[1],
        output_dim=Y_train_t.shape[1],
        archi=candidate["archi"],
        dropout=candidate["dropout"],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=candidate["learning_rate"],
        momentum=candidate["momentum"],
        nesterov=candidate["nesterov"],
    )

    epochs = max(1, int(epochs))

    for epoch in range(1, epochs + 1):
        _set_learning_rate(
            optimizer,
            candidate["learning_rate"],
            candidate["decay_rate"],
            epoch,
        )

        _train_epoch(
            model=model,
            optimizer=optimizer,
            X=X_train_t,
            Y=Y_train_t,
            candidate=candidate,
            seed=seed + epoch,
        )

    model.eval()

    with torch.no_grad():
        pred = model(X_test_t).cpu().numpy()

    return pred.reshape(-1)


def _train_epoch(model, optimizer, X, Y, candidate, seed):
    model.train()

    n = X.shape[0]
    batch_size = max(1, int(candidate["batch_size"]))

    if candidate["shuffle"]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
    else:
        order = np.arange(n)

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]

        xb = X[idx]
        yb = Y[idx]

        optimizer.zero_grad()

        pred = model(xb)
        loss = _criterion(pred, yb, candidate)
        loss = loss + _regularization(model, candidate["l1"], candidate["l2"])

        loss.backward()
        optimizer.step()


def _eval_loss(model, X, Y, candidate):
    model.eval()

    with torch.no_grad():
        pred = model(X)
        loss = _criterion(pred, Y, candidate)

    return float(loss.cpu().item())


def _criterion(pred, target, candidate):
    if candidate["loss"] == "huber":
        return nn.functional.huber_loss(
            pred,
            target,
            delta=float(candidate["huber_delta"]),
            reduction="mean",
        )

    if candidate["loss"] == "mse":
        return nn.functional.mse_loss(pred, target, reduction="mean")

    raise ValueError(f"Unknown loss: {candidate['loss']}")


def _regularization(model, l1, l2):
    penalty = None

    for name, param in model.named_parameters():
        if "weight" not in name:
            continue

        if penalty is None:
            penalty = torch.zeros((), dtype=param.dtype, device=param.device)

        if l1 > 0:
            penalty = penalty + float(l1) * torch.sum(torch.abs(param))

        if l2 > 0:
            penalty = penalty + float(l2) * torch.sum(param ** 2)

    if penalty is None:
        return torch.tensor(0.0)

    return penalty


def _set_learning_rate(optimizer, initial_lr, decay_rate, epoch):
    lr = float(initial_lr) / (1.0 + float(decay_rate) * max(0, epoch - 1))

    for group in optimizer.param_groups:
        group["lr"] = lr


def _tensor(x):
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    return torch.tensor(x, dtype=torch.float32)


def _set_seed(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)


def _parse_l1l2(x):
    arr = np.asarray(x, dtype=float).ravel()

    if arr.size == 0:
        return 0.0, 0.0

    if arr.size == 1:
        return float(arr[0]), float(arr[0])

    return float(arr[0]), float(arr[1])