#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

PURGE_SIZE = 12


def _to_2d(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _split_train_test(X, Y):
    X = _to_2d(X)
    Y = _to_2d(Y)

    if X.shape[0] < 2 or Y.shape[0] < 2:
        raise ValueError("Need at least two rows in X and Y.")

    X_train = X[:-1, :]
    Y_train = Y[:-1, :]
    X_test = X[-1:, :]
    return X_train, Y_train, X_test


def _get_model_path(dumploc, seed):
    return os.path.join(dumploc, f"BestModel_{seed}.keras")


def _parse_l1l2(value):
    if value is None:
        return 0.0, 0.0

    if np.isscalar(value):
        v = float(value)
        return v, v

    arr = np.asarray(value, dtype=float).ravel()

    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        v = float(arr[0])
        return v, v

    return float(arr[0]), float(arr[1])


def _parse_dropout(value):
    if value is None:
        return 0.0

    if np.isscalar(value):
        return float(value)

    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        return 0.0

    return float(arr[0])


def _get_loss(loss_name, huber_delta):
    import keras

    if str(loss_name).lower() == "huber":
        return keras.losses.Huber(delta=float(huber_delta))

    return "mean_squared_error"


def _build_optimizer(fit_cfg):
    import keras

    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=float(fit_cfg["learning_rate"]),
        decay_steps=1,
        decay_rate=float(fit_cfg["decay_rate"]),
    )

    return keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=float(fit_cfg["momentum"]),
        nesterov=bool(fit_cfg["nesterov"]),
    )


def _scale_fit_val_test(X_fit, X_val, X_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_fit_scaled.astype(np.float32),
        X_val_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
    )


def _purged_split(X_train, Y_train, validation_split):
    n_train = X_train.shape[0]
    val_len = int(np.ceil(n_train * float(validation_split)))
    fit_end = n_train - PURGE_SIZE - val_len
    val_start = fit_end + PURGE_SIZE

    X_fit = X_train[:fit_end, :]
    Y_fit = Y_train[:fit_end, :]
    X_val = X_train[val_start:, :]
    Y_val = Y_train[val_start:, :]

    return X_fit, Y_fit, X_val, Y_val


def _normalize_params(params):
    params = {} if params is None else dict(params)

    if "archi" not in params:
        raise ValueError("params must contain 'archi'.")

    defaults = {
        "Dropout": 0.0,
        "l1l2": [0.0, 0.0],
        "learning_rate": 0.03,
        "decay_rate": 0.001,
        "momentum": 0.9,
        "nesterov": True,
        "epochs": 500,
        "patience": 20,
        "batch_size": 32,
        "validation_split": 0.15,
        "shuffle": True,
        "loss_name": "mse",
        "huber_delta": 1.0,
    }

    for key, value in defaults.items():
        params.setdefault(key, value)

    return params


def _build_single_model(input_dim, output_dim, archi, dropout_u, l1l2penal, fit_cfg):
    import keras
    from keras import layers, regularizers

    dropout_u = _parse_dropout(dropout_u)
    l1_pen, l2_pen = _parse_l1l2(l1l2penal)
    reg = regularizers.l1_l2(l1=l1_pen, l2=l2_pen)

    model = keras.Sequential(name="NNModel")
    model.add(layers.Input(shape=(int(input_dim),)))

    for layer_idx, width in enumerate(archi):
        if layer_idx == 0 and dropout_u > 0.0:
            model.add(layers.Dropout(dropout_u))

        model.add(
            layers.Dense(
                int(width),
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                kernel_regularizer=reg,
            )
        )

        if dropout_u > 0.0:
            model.add(layers.Dropout(dropout_u))

    model.add(layers.BatchNormalization())
    model.add(
        layers.Dense(
            int(output_dim),
            activation="linear",
            kernel_initializer="he_normal",
            bias_initializer="zeros",
        )
    )

    model.compile(
        loss=_get_loss(fit_cfg["loss_name"], fit_cfg["huber_delta"]),
        optimizer=_build_optimizer(fit_cfg),
    )

    return model


def _fit_single_model(X, Y, seed, params, dumploc, refit):
    import keras
    from keras.callbacks import EarlyStopping
    from keras.models import load_model

    X_train, Y_train, X_test = _split_train_test(X, Y)

    X_fit, Y_fit, X_val, Y_val = _purged_split(
        X_train,
        Y_train,
        validation_split=params["validation_split"],
    )

    X_fit, X_val, X_test = _scale_fit_val_test(X_fit, X_val, X_test)

    keras.utils.set_random_seed(int(seed))
    model_path = _get_model_path(dumploc, seed)

    if bool(refit) or not os.path.exists(model_path):
        keras.backend.clear_session()
        model = _build_single_model(
            input_dim=X_fit.shape[1],
            output_dim=Y_fit.shape[1],
            archi=params["archi"],
            dropout_u=params["Dropout"],
            l1l2penal=params["l1l2"],
            fit_cfg=params,
        )
    else:
        model = load_model(model_path)

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-6,
        patience=int(params["patience"]),
        mode="min",
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_fit,
        Y_fit,
        epochs=int(params["epochs"]),
        validation_data=(X_val, Y_val),
        batch_size=int(params["batch_size"]),
        shuffle=bool(params["shuffle"]),
        callbacks=[early_stop],
        verbose=0,
    )

    model.save(model_path)

    y_pred = model.predict(X_test, verbose=0).astype(np.float32)
    val_hist = history.history.get("val_loss", [])
    val_min = float(np.min(val_hist)) if len(val_hist) > 0 else np.nan

    return y_pred, val_min


def NNModel(X, Y, no, params=None, refit=None, dumploc=None):
    if dumploc is None:
        raise ValueError("Missing dumploc argument")

    params = _normalize_params(params)

    return _fit_single_model(
        X=X,
        Y=Y,
        seed=no,
        params=params,
        dumploc=dumploc,
        refit=refit,
    )


def _split_branches(X_block, branch_specs):
    X_block = _to_2d(X_block)
    blocks = []

    for spec in branch_specs:
        col_start = int(spec["col_start"])
        col_end = int(spec["col_end"])
        blocks.append(X_block[:, col_start:col_end])

    return blocks


def _normalize_dual_params(params):
    params = {} if params is None else dict(params)

    if "branch_specs" not in params:
        raise ValueError("params must contain 'branch_specs'.")

    defaults = {
        "head_archi": [],
        "head_dropout": 0.0,
        "head_l1l2": [0.0, 0.0],
        "learning_rate": 0.03,
        "decay_rate": 0.001,
        "momentum": 0.9,
        "nesterov": True,
        "epochs": 500,
        "patience": 20,
        "batch_size": 32,
        "validation_split": 0.15,
        "shuffle": True,
        "loss_name": "mse",
        "huber_delta": 1.0,
    }

    for key, value in defaults.items():
        params.setdefault(key, value)

    return params


def _build_dual_model(branch_specs, output_dim, params):
    import keras
    from keras import layers, regularizers

    def _make_reg(value):
        l1_pen, l2_pen = _parse_l1l2(value)
        return regularizers.l1_l2(l1=l1_pen, l2=l2_pen)

    branch_inputs = []
    branch_outputs = []

    for spec in branch_specs:
        branch_input = layers.Input(
            shape=(int(spec["input_dim"]),),
            name=f"{spec['name']}_input",
        )
        h = branch_input

        branch_archi = list(spec.get("archi", []))
        branch_dropout = _parse_dropout(spec.get("dropout", 0.0))
        branch_reg = _make_reg(spec.get("l1l2", [0.0, 0.0]))

        for layer_idx, width in enumerate(branch_archi):
            if layer_idx == 0 and branch_dropout > 0.0:
                h = layers.Dropout(branch_dropout, name=f"{spec['name']}_dropin")(h)

            h = layers.Dense(
                int(width),
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                kernel_regularizer=branch_reg,
                name=f"{spec['name']}_dense_{layer_idx + 1}",
            )(h)

            if branch_dropout > 0.0:
                h = layers.Dropout(branch_dropout, name=f"{spec['name']}_drop_{layer_idx + 1}")(h)

        branch_inputs.append(branch_input)
        branch_outputs.append(h)

    if len(branch_outputs) == 1:
        h = branch_outputs[0]
    else:
        h = layers.Concatenate(name="concat_branches")(branch_outputs)

    head_archi = list(params.get("head_archi", []))
    head_dropout = _parse_dropout(params.get("head_dropout", 0.0))
    head_reg = _make_reg(params.get("head_l1l2", [0.0, 0.0]))

    for layer_idx, width in enumerate(head_archi):
        h = layers.Dense(
            int(width),
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            kernel_regularizer=head_reg,
            name=f"head_dense_{layer_idx + 1}",
        )(h)

        if head_dropout > 0.0:
            h = layers.Dropout(head_dropout, name=f"head_drop_{layer_idx + 1}")(h)

    h = layers.BatchNormalization(name="head_bn")(h)

    y_out = layers.Dense(
        int(output_dim),
        activation="linear",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name="y_out",
    )(h)

    model = keras.Model(inputs=branch_inputs, outputs=y_out, name="NNDualModel")
    model.compile(
        loss=_get_loss(params["loss_name"], params["huber_delta"]),
        optimizer=_build_optimizer(params),
    )

    return model


def _fit_dual_model(X, Y, seed, params, dumploc, refit):
    import keras
    from keras.callbacks import EarlyStopping
    from keras.models import load_model

    if dumploc is None:
        raise ValueError("Missing dumploc argument")

    branch_specs = params["branch_specs"]

    X_train, Y_train, X_test = _split_train_test(X, Y)

    X_fit_raw, Y_fit, X_val_raw, Y_val = _purged_split(
        X_train,
        Y_train,
        validation_split=params["validation_split"],
    )

    X_test_blocks = _split_branches(X_test, branch_specs)
    X_fit_blocks = _split_branches(X_fit_raw, branch_specs)
    X_val_blocks = _split_branches(X_val_raw, branch_specs)

    X_fit_list = []
    X_val_list = []
    X_test_list = []

    for X_fit_block, X_val_block, X_test_block in zip(X_fit_blocks, X_val_blocks, X_test_blocks):
        X_fit_scaled, X_val_scaled, X_test_scaled = _scale_fit_val_test(
            X_fit_block,
            X_val_block,
            X_test_block,
        )
        X_fit_list.append(X_fit_scaled)
        X_val_list.append(X_val_scaled)
        X_test_list.append(X_test_scaled)

    keras.utils.set_random_seed(int(seed))
    model_path = _get_model_path(dumploc, seed)

    if bool(refit) or not os.path.exists(model_path):
        keras.backend.clear_session()
        model = _build_dual_model(
            branch_specs=branch_specs,
            output_dim=Y_fit.shape[1],
            params=params,
        )
    else:
        model = load_model(model_path)

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-6,
        patience=int(params["patience"]),
        mode="min",
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_fit_list,
        Y_fit,
        epochs=int(params["epochs"]),
        validation_data=(X_val_list, Y_val),
        batch_size=int(params["batch_size"]),
        shuffle=bool(params["shuffle"]),
        callbacks=[early_stop],
        verbose=0,
    )

    model.save(model_path)

    y_pred = model.predict(X_test_list, verbose=0).astype(np.float32)
    val_hist = history.history.get("val_loss", [])
    val_min = float(np.min(val_hist)) if len(val_hist) > 0 else np.nan

    return y_pred, val_min


def NNDualModel(X, Y, no, params=None, refit=None, dumploc=None):
    if dumploc is None:
        raise ValueError("Missing dumploc argument")

    params = _normalize_dual_params(params)

    return _fit_dual_model(
        X=X,
        Y=Y,
        seed=no,
        params=params,
        dumploc=dumploc,
        refit=refit,
    )