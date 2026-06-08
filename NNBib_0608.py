#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np


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
    return os.path.join(dumploc, f"BestModel_{int(seed)}.keras")


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


def _target_weights(params, output_dim):
    weights = params.get("target_weights", None)

    if weights is None:
        return None

    arr = np.asarray(weights, dtype=np.float32).ravel()

    if arr.size == 0:
        return None

    if arr.size != int(output_dim):
        raise ValueError(
            f"target_weights has {arr.size} entries, expected {int(output_dim)}."
        )

    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("target_weights must be finite and non-negative.")

    mean_weight = float(np.mean(arr))
    if mean_weight <= 0.0:
        raise ValueError("target_weights must have positive average weight.")

    return (arr / mean_weight).astype(np.float32)


def _validation_loss(Y_true, Y_pred, params):
    err2 = (np.asarray(Y_true, dtype=np.float32) - np.asarray(Y_pred, dtype=np.float32)) ** 2
    weights = _target_weights(params, err2.shape[1])

    if weights is not None:
        err2 = err2 * weights.reshape(1, -1)

    return float(np.mean(err2))


def _get_loss(loss_name, huber_delta, target_weights=None):
    import keras

    loss_name = str(loss_name).lower()

    if target_weights is not None:
        import tensorflow as tf

        weights = tf.constant(np.asarray(target_weights, dtype=np.float32).reshape(1, -1))

        if loss_name in {"mse", "mean_squared_error"}:
            def weighted_mse(y_true, y_pred):
                return tf.reduce_mean(tf.square(y_pred - y_true) * weights)

            return weighted_mse

        if loss_name == "huber":
            delta = tf.constant(float(huber_delta), dtype=tf.float32)

            def weighted_huber(y_true, y_pred):
                err = tf.abs(y_pred - y_true)
                quadratic = tf.minimum(err, delta)
                linear = err - quadratic
                loss = 0.5 * tf.square(quadratic) + delta * linear
                return tf.reduce_mean(loss * weights)

            return weighted_huber

    if loss_name == "huber":
        return keras.losses.Huber(delta=float(huber_delta))

    if loss_name in {"mse", "mean_squared_error"}:
        return "mean_squared_error"

    raise ValueError(f"Unknown loss_name: {loss_name}")


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


def _purged_split(X_train, Y_train, validation_split, purge_size):
    n_train = X_train.shape[0]

    val_len = int(np.ceil(n_train * float(validation_split)))
    fit_end = n_train - int(purge_size) - val_len
    val_start = fit_end + int(purge_size)

    if fit_end <= 0:
        raise ValueError(
            f"Invalid purged split: n_train={n_train}, "
            f"val_len={val_len}, purge_size={purge_size}."
        )

    X_fit = X_train[:fit_end, :]
    Y_fit = Y_train[:fit_end, :]
    X_val = X_train[val_start:, :]
    Y_val = Y_train[val_start:, :]

    if X_val.shape[0] == 0:
        raise ValueError(
            f"Validation block is empty: n_train={n_train}, "
            f"val_start={val_start}."
        )

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
        "purge_size": 12,
        "shuffle": True,
        "loss_name": "mse",
        "huber_delta": 1.0,
        "n_pcs": 3,
        "pcs": [],
        "aggregation": "mean",
        "use_bias": True,
        "regularize_output": False,
        "target_weights": None,
    }

    for key, value in defaults.items():
        params.setdefault(key, value)

    return params


def _build_single_model(input_dim, output_dim, archi, dropout_u, l1l2penal, fit_cfg):
    import keras
    from keras import layers, regularizers

    dropout_u = _parse_dropout(dropout_u)
    l1_pen, l2_pen = _parse_l1l2(l1l2penal)
    use_bias = bool(fit_cfg.get("use_bias", True))

    reg = regularizers.l1_l2(l1=l1_pen, l2=l2_pen)
    output_regularizer = reg if bool(fit_cfg.get("regularize_output", False)) else None

    model = keras.Sequential(name="NN")
    model.add(layers.Input(shape=(int(input_dim),)))

    for layer_idx, width in enumerate(archi):
        if layer_idx == 0 and dropout_u > 0.0:
            model.add(layers.Dropout(dropout_u))

        model.add(
            layers.Dense(
                int(width),
                activation="relu",
                use_bias=use_bias,
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
            use_bias=use_bias,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            kernel_regularizer=output_regularizer,
        )
    )

    target_weights = _target_weights(fit_cfg, output_dim)

    model.compile(
        loss=_get_loss(fit_cfg["loss_name"], fit_cfg["huber_delta"], target_weights),
        optimizer=_build_optimizer(fit_cfg),
    )

    return model


def _fit_ols(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    X_reg = np.column_stack([np.ones(X.shape[0]), X])
    beta, _, _, _ = np.linalg.lstsq(X_reg, Y, rcond=None)

    return beta


def _predict_ols(X, beta):
    X = np.asarray(X, dtype=np.float64)
    X_reg = np.column_stack([np.ones(X.shape[0]), X])
    return (X_reg @ beta).astype(np.float32)


def _fit_pca(X, n_components):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError("PCA input must be 2D.")

    mu = np.mean(X, axis=0)
    Xc = X - mu

    _, _, vt = np.linalg.svd(Xc, full_matrices=False)

    n_keep = min(int(n_components), vt.shape[0])
    loadings = vt[:n_keep, :].T

    for j in range(loadings.shape[1]):
        k = int(np.argmax(np.abs(loadings[:, j])))
        if loadings[k, j] < 0:
            loadings[:, j] *= -1.0

    return {
        "mean": mu.astype(np.float32),
        "loadings": loadings.astype(np.float32),
    }


def _apply_pca(X, pca):
    X = np.asarray(X, dtype=np.float32)
    return (X - pca["mean"]) @ pca["loadings"]


def _pc_indices_from_params(params, n_available):
    pcs = params.get("pcs", [])

    if pcs is None:
        pcs = []

    if np.isscalar(pcs):
        pcs = [int(pcs)]
    else:
        pcs = [int(v) for v in np.asarray(pcs).ravel()]

    if len(pcs) == 0:
        return list(range(n_available))

    if min(pcs) < 1 or max(pcs) > n_available:
        raise ValueError(
            f"pcs={pcs} is incompatible with available PCs 1,...,{n_available}."
        )

    return [j - 1 for j in pcs]


def _prepare_pcnn_fit_val_test(X_fit_raw, X_val_raw, X_test_raw, params):
    n_pcs = int(params.get("n_pcs", 3))

    pca = _fit_pca(X_fit_raw, n_components=n_pcs)

    X_fit_pc = _apply_pca(X_fit_raw, pca)
    X_val_pc = _apply_pca(X_val_raw, pca)
    X_test_pc = _apply_pca(X_test_raw, pca)

    pc_idx = _pc_indices_from_params(params, X_fit_pc.shape[1])

    X_fit_pc = X_fit_pc[:, pc_idx]
    X_val_pc = X_val_pc[:, pc_idx]
    X_test_pc = X_test_pc[:, pc_idx]

    return (
        X_fit_pc.astype(np.float32),
        X_val_pc.astype(np.float32),
        X_test_pc.astype(np.float32),
    )


def _train_or_load_keras_model(X_fit, Y_fit, X_val, Y_val, X_test, seed, params, dumploc, refit):
    import keras
    from keras.callbacks import EarlyStopping
    from keras.models import load_model

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
        if _target_weights(params, Y_fit.shape[1]) is None:
            model = load_model(model_path)
        else:
            model = load_model(model_path, compile=False)
            model.compile(
                loss=_get_loss(
                    params["loss_name"],
                    params["huber_delta"],
                    _target_weights(params, Y_fit.shape[1]),
                ),
                optimizer=_build_optimizer(params),
            )

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

    # Use direct call to avoid tf.function retracing warnings for single/small test inputs
    y_pred = model(X_test, training=False).numpy().astype(np.float32)

    val_hist = history.history.get("val_loss", [])
    val_min = float(np.min(val_hist)) if len(val_hist) > 0 else np.nan

    return y_pred, val_min


def _fit_single_nn_model(X, Y, seed, params, dumploc, refit):
    X_train, Y_train, X_test = _split_train_test(X, Y)

    X_fit, Y_fit, X_val, Y_val = _purged_split(
        X_train,
        Y_train,
        validation_split=params["validation_split"],
        purge_size=params["purge_size"],
    )

    X_fit, X_val, X_test = _scale_fit_val_test(X_fit, X_val, X_test)

    if len(params["archi"]) == 0:
        beta = _fit_ols(X_fit, Y_fit)
        y_val_pred = _predict_ols(X_val, beta)
        y_pred = _predict_ols(X_test, beta)
        val_loss = _validation_loss(Y_val, y_val_pred, params)
        return y_pred, val_loss

    return _train_or_load_keras_model(
        X_fit=X_fit,
        Y_fit=Y_fit,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        seed=seed,
        params=params,
        dumploc=dumploc,
        refit=refit,
    )


def _fit_single_pcnn_model(X, Y, seed, params, dumploc, refit):
    X_train, Y_train, X_test_raw = _split_train_test(X, Y)

    X_fit_raw, Y_fit, X_val_raw, Y_val = _purged_split(
        X_train,
        Y_train,
        validation_split=params["validation_split"],
        purge_size=params["purge_size"],
    )

    X_fit, X_val, X_test = _prepare_pcnn_fit_val_test(
        X_fit_raw=X_fit_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        params=params,
    )

    X_fit, X_val, X_test = _scale_fit_val_test(X_fit, X_val, X_test)

    if len(params["archi"]) == 0:
        beta = _fit_ols(X_fit, Y_fit)
        y_val_pred = _predict_ols(X_val, beta)
        y_pred = _predict_ols(X_test, beta)
        val_loss = _validation_loss(Y_val, y_val_pred, params)
        return y_pred, val_loss

    return _train_or_load_keras_model(
        X_fit=X_fit,
        Y_fit=Y_fit,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        seed=seed,
        params=params,
        dumploc=dumploc,
        refit=refit,
    )


def NN(X, Y, no, params=None, refit=None, dumploc=None):
    if dumploc is None:
        raise ValueError("Missing dumploc argument.")

    params = _normalize_params(params)

    return _fit_single_nn_model(
        X=X,
        Y=Y,
        seed=no,
        params=params,
        dumploc=dumploc,
        refit=refit,
    )


def pcNN(X, Y, no, params=None, refit=None, dumploc=None):
    if dumploc is None:
        raise ValueError("Missing dumploc argument.")

    params = _normalize_params(params)

    return _fit_single_pcnn_model(
        X=X,
        Y=Y,
        seed=no,
        params=params,
        dumploc=dumploc,
        refit=refit,
    )


def _build_multibranch_model(group_sizes, output_dim, archi, dropout_u, l1l2penal, fit_cfg):
    import keras
    from keras import layers, regularizers
    
    dropout_u = _parse_dropout(dropout_u)
    l1_pen, l2_pen = _parse_l1l2(l1l2penal)
    reg = regularizers.l1_l2(l1=l1_pen, l2=l2_pen)
    
    inputs = []
    branches = []
    
    for g_idx, g_size in enumerate(group_sizes):
        if g_size == 0:
            continue
        
        group_input = layers.Input(shape=(int(g_size),), name=f"group_input_{g_idx}")
        inputs.append(group_input)
        
        x = group_input
        for layer_idx, width in enumerate(archi):
            if layer_idx == 0 and dropout_u > 0.0:
                x = layers.Dropout(dropout_u)(x)
            
            x = layers.Dense(
                int(width),
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                kernel_regularizer=reg,
                name=f"branch_{g_idx}_dense_{layer_idx}"
            )(x)
            
            if dropout_u > 0.0:
                x = layers.Dropout(dropout_u)(x)
        
        branches.append(x)
    
    if len(branches) > 1:
        merged = layers.Concatenate(name="concat_branches")(branches)
    else:
        merged = branches[0]
        
    merged = layers.BatchNormalization(name="batch_norm")(merged)
    
    outputs = layers.Dense(
        int(output_dim),
        activation="linear",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name="output_layer"
    )(merged)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="MacroNN")
    
    target_weights = _target_weights(fit_cfg, output_dim)

    model.compile(
        loss=_get_loss(fit_cfg["loss_name"], fit_cfg["huber_delta"], target_weights),
        optimizer=_build_optimizer(fit_cfg),
    )
    
    return model


def _train_or_load_multibranch_model(X_fit_list, Y_fit, X_val_list, Y_val, X_test_list, seed, params, dumploc, refit):
    import keras
    from keras.callbacks import EarlyStopping
    from keras.models import load_model

    keras.utils.set_random_seed(int(seed))

    model_path = _get_model_path(dumploc, seed)
    group_sizes = [int(s) for s in params["group_sizes"] if int(s) > 0]
    output_dim = Y_fit.shape[1]

    if bool(refit) or not os.path.exists(model_path):
        keras.backend.clear_session()

        model = _build_multibranch_model(
            group_sizes=group_sizes,
            output_dim=output_dim,
            archi=params["archi"],
            dropout_u=params["Dropout"],
            l1l2penal=params["l1l2"],
            fit_cfg=params,
        )

    else:
        if _target_weights(params, Y_fit.shape[1]) is None:
            model = load_model(model_path)
        else:
            model = load_model(model_path, compile=False)
            model.compile(
                loss=_get_loss(
                    params["loss_name"],
                    params["huber_delta"],
                    _target_weights(params, Y_fit.shape[1]),
                ),
                optimizer=_build_optimizer(params),
            )

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

    # Use direct model call to avoid tf.function retracing warnings
    y_pred = model(X_test_list, training=False).numpy().astype(np.float32)

    val_hist = history.history.get("val_loss", [])
    val_min = float(np.min(val_hist)) if len(val_hist) > 0 else np.nan

    return y_pred, val_min


def MacroNN(X, Y, no, params=None, refit=None, dumploc=None):
    """BBT (2021) Group-ensemble Neural Network.

    Trains a single unified network with multi-branch inputs (one branch per feature group),
    concatenates their representation, and trains them end-to-end.

    Parameters
    ----------
    X : array-like, shape (T, sum(group_sizes))
        Concatenated feature matrix. Last row is the test observation.
    Y : array-like, shape (T, M)
        Target matrix. Last row is the test observation.
    no : int
        Random seed index.
    params : dict
        Must contain ``group_sizes``: list[int] with the column count
        for each group. ``sum(group_sizes) == X.shape[1]``.
    refit : bool
        Whether to retrain from scratch.
    dumploc : str
        Directory for saving / loading model checkpoints.

    Returns
    -------
    y_pred : ndarray, shape (1, M)
        Unified multi-branch out-of-sample prediction.
    val_loss : float
        Validation loss of the unified model.
    """
    if dumploc is None:
        raise ValueError("Missing dumploc argument.")

    params = _normalize_params(params)
    group_sizes = params.get("group_sizes", None)

    if group_sizes is None:
        raise ValueError("MacroNN requires 'group_sizes' in params.")

    group_sizes = [int(s) for s in group_sizes]

    X = _to_2d(np.asarray(X, dtype=np.float32))
    Y = _to_2d(np.asarray(Y, dtype=np.float32))

    total_cols = sum(group_sizes)
    if total_cols != X.shape[1]:
        raise ValueError(
            f"sum(group_sizes)={total_cols} != X.shape[1]={X.shape[1]}"
        )

    # 1. Split train, test, fit, val
    X_train, Y_train, X_test = _split_train_test(X, Y)
    X_fit_raw, Y_fit, X_val_raw, Y_val = _purged_split(
        X_train,
        Y_train,
        validation_split=params["validation_split"],
        purge_size=params["purge_size"],
    )

    # 2. Slice and scale each group
    boundaries = np.cumsum([0] + group_sizes)
    X_fit_list = []
    X_val_list = []
    X_test_list = []

    for g in range(len(group_sizes)):
        g_size = group_sizes[g]
        if g_size == 0:
            continue

        g_fit = X_fit_raw[:, boundaries[g]:boundaries[g + 1]]
        g_val = X_val_raw[:, boundaries[g]:boundaries[g + 1]]
        g_test = X_test[:, boundaries[g]:boundaries[g + 1]]

        # Scale each group individually
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        g_fit_scaled = scaler.fit_transform(g_fit).astype(np.float32)
        g_val_scaled = scaler.transform(g_val).astype(np.float32)
        g_test_scaled = scaler.transform(g_test).astype(np.float32)

        X_fit_list.append(g_fit_scaled)
        X_val_list.append(g_val_scaled)
        X_test_list.append(g_test_scaled)

    # Handle OLS edge-case if archi is empty
    if len(params["archi"]) == 0:
        # Perform OLS on concatenated scaled inputs
        X_fit_cat = np.hstack(X_fit_list)
        X_val_cat = np.hstack(X_val_list)
        X_test_cat = np.hstack(X_test_list)

        beta = _fit_ols(X_fit_cat, Y_fit)
        y_val_pred = _predict_ols(X_val_cat, beta)
        y_pred = _predict_ols(X_test_cat, beta)
        val_loss = _validation_loss(Y_val, y_val_pred, params)
        return y_pred, val_loss

    # 3. Train unified multibranch model
    return _train_or_load_multibranch_model(
        X_fit_list=X_fit_list,
        Y_fit=Y_fit,
        X_val_list=X_val_list,
        Y_val=Y_val,
        X_test_list=X_test_list,
        seed=no,
        params=params,
        dumploc=dumploc,
        refit=refit,
    )
