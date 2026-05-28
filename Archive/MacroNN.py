#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import json
import time
import pickle

import numpy as np
import pandas as pd
import scipy.io as sio

import Utils as U

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


BASE_CONFIG = {
    "mat_path": "data/target_and_features.mat",
    "target_group": "dy",
    "horizon": 12,
    "oos_start": "1989-01-31",
    "hyper_freq": 60,
    "nmc": 1,
    "navg": 1,
    "run_tag": "macro_group",
    "model_name": "NNMacroGroup",
    "feature_groups": [],
    "branch_config": {
        "yield": {
            "feature_groups": ["dy_pc1", "dy_pc2"],
            "archi": [3, 3],
        },
        "macro": {
            "feature_groups": [
                "macro_output",
                "macro_labor",
            ],
            "archi": [3],
        },
    },
    "params": {
        "dropout": 0.00,
        "l1l2": [1e-4, 5e-5],
        "learning_rate": 0.01,
        "decay_rate": 0.01,
        "momentum": 0.9,
        "nesterov": True,
        "epochs": 500,
        "patience": 20,
        "batch_size": 32,
        "validation_split": 0.15,
        "shuffle": False,
        "loss_name": "mse",
        "huber_delta": 1.0,
    },
}


def result_name(cfg):
    target = str(cfg["target_group"])
    run_tag = str(cfg["run_tag"])
    horizon = f"h{int(cfg['horizon'])}"
    y_arch = "x".join(str(v) for v in cfg["branch_config"]["yield"]["archi"])
    m_arch = "x".join(str(v) for v in cfg["branch_config"]["macro"]["archi"])
    dropout = float(cfg["params"]["dropout"])
    lr = float(cfg["params"]["learning_rate"])

    return "__".join([
        target,
        run_tag,
        cfg["model_name"],
        horizon,
        f"ya{y_arch}",
        f"ma{m_arch}",
        f"do{dropout:g}",
        f"lr{lr:g}",
    ])


def load_block(Xmat, block_name):
    block = getattr(Xmat, block_name)

    time_vec = U.to_1d(block.Time).astype(int)
    data = np.asarray(block.data, dtype=float)
    names = U.to_name_list(block.names)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    dates = U.yyyymm_to_month_end(time_vec)
    names = [f"{block_name}__{c}" for c in names]

    df = pd.DataFrame(data, index=dates, columns=names)
    df.index.name = "Date"
    return df


def load_dataset(mat_path, branch_config, target_group):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    Xmat = mat["X"]
    ymat = mat["y"]

    yblock = getattr(ymat, target_group)
    y_time = U.to_1d(yblock.Time).astype(int)
    y_data = np.asarray(yblock.data, dtype=float)
    y_names = U.to_name_list(yblock.names)

    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)

    y_dates = U.yyyymm_to_month_end(y_time)
    Y_df = pd.DataFrame(y_data, index=y_dates, columns=y_names)
    Y_df.index.name = "Date"

    yield_parts = [load_block(Xmat, g) for g in branch_config["yield"]["feature_groups"]]
    X_yield_df = pd.concat(yield_parts, axis=1)

    macro_group_names = list(branch_config["macro"]["feature_groups"])
    X_macro_group_dfs = [load_block(Xmat, g) for g in macro_group_names]

    merged = Y_df.copy()
    merged = merged.join(X_yield_df, how="inner")
    for gdf in X_macro_group_dfs:
        merged = merged.join(gdf, how="inner")

    merged = merged.dropna().copy()

    Y_df = merged[Y_df.columns].copy()
    X_yield_df = merged[X_yield_df.columns].copy()
    X_macro_group_dfs = [merged[gdf.columns].copy() for gdf in X_macro_group_dfs]

    return Y_df, X_yield_df, X_macro_group_dfs, macro_group_names


def split_train_test_last(X, Y, horizon):
    X_train = X[:-horizon, :]
    Y_train = Y[:-horizon, :]
    X_test = X[-1, :].reshape(1, -1)
    return X_train, Y_train, X_test


def split_group_arrays(X, group_sizes):
    out = []
    start = 0
    for size in group_sizes:
        end = start + size
        out.append(X[:, start:end])
        start = end
    return out


def build_optimizer(params):
    import tensorflow as tf

    return tf.keras.optimizers.SGD(
        learning_rate=float(params["learning_rate"]),
        momentum=float(params["momentum"]),
        nesterov=bool(params["nesterov"]),
    )


def build_loss(params):
    import tensorflow as tf

    loss_name = str(params["loss_name"]).lower()
    if loss_name == "mse":
        return "mean_squared_error"
    if loss_name == "huber":
        return tf.keras.losses.Huber(delta=float(params["huber_delta"]))
    raise ValueError(f"Unsupported loss_name: {params['loss_name']}")


def build_group_model(group_dims, yield_dim, output_dim, params):
    import tensorflow as tf

    macro_archi = list(params["archi_macro"])
    yield_archi = list(params["archi_yield"])
    dropout = float(params["dropout"])
    l1_val, l2_val = list(np.asarray(params["l1l2"], dtype=float).reshape(-1)[:2])
    reg = tf.keras.regularizers.l1_l2(l1=l1_val, l2=l2_val)

    macro_inputs = []
    macro_hidden = []

    for i, dim in enumerate(group_dims):
        inp = tf.keras.Input(shape=(int(dim),), name=f"macro_group_{i+1}")
        x = inp
        for units in macro_archi:
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(
                int(units),
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer="he_normal",
                kernel_regularizer=reg,
            )(x)
        macro_inputs.append(inp)
        macro_hidden.append(x)

    if len(macro_hidden) == 1:
        macro_merge = macro_hidden[0]
    else:
        macro_merge = tf.keras.layers.Concatenate()(macro_hidden)

    yield_input = tf.keras.Input(shape=(int(yield_dim),), name="yield_branch")
    y = yield_input
    for units in yield_archi:
        y = tf.keras.layers.Dropout(dropout)(y)
        y = tf.keras.layers.Dense(
            int(units),
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer="he_normal",
            kernel_regularizer=reg,
        )(y)

    merged = tf.keras.layers.Concatenate()([macro_merge, y])
    merged = tf.keras.layers.Dropout(dropout)(merged)
    merged = tf.keras.layers.BatchNormalization()(merged)

    output = tf.keras.layers.Dense(
        int(output_dim),
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
    )(merged)

    model = tf.keras.Model(inputs=macro_inputs + [yield_input], outputs=output)
    model.compile(loss=build_loss(params), optimizer=build_optimizer(params))
    return model


def NNMacroGroupModel(X, Y, no, params=None, refit=None, dumploc=None):
    if params is None:
        raise ValueError("params must be provided.")
    if dumploc is None:
        raise ValueError("dumploc must be provided.")

    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    tf.random.set_seed(int(no))
    np.random.seed(int(no))

    horizon = int(params["horizon"])
    group_sizes = list(params["group_sizes"])
    yield_dim = int(params["yield_dim"])

    macro_dim = int(np.sum(group_sizes))
    X_macro = X[:, :macro_dim]
    X_yield = X[:, macro_dim:macro_dim + yield_dim]

    X_macro_train, Y_train, X_macro_test = split_train_test_last(X_macro, Y, horizon)
    X_yield_train, _, X_yield_test = split_train_test_last(X_yield, Y, horizon)

    macro_train_groups = split_group_arrays(X_macro_train, group_sizes)
    macro_test_groups = split_group_arrays(X_macro_test, group_sizes)

    macro_scalers = []
    macro_train_scaled = []
    macro_test_scaled = []

    for xtr, xte in zip(macro_train_groups, macro_test_groups):
        scaler = StandardScaler()
        xtr_s = scaler.fit_transform(xtr)
        xte_s = scaler.transform(xte)
        macro_scalers.append(scaler)
        macro_train_scaled.append(xtr_s)
        macro_test_scaled.append(xte_s)

    yield_scaler = StandardScaler()
    X_yield_train_scaled = yield_scaler.fit_transform(X_yield_train)
    X_yield_test_scaled = yield_scaler.transform(X_yield_test)

    model_path = os.path.join(dumploc, f"BestModel_{no}.keras")
    hist_path = os.path.join(dumploc, f"BestModelHist_{no}.pkl")

    if refit:
        model = build_group_model(
            group_dims=group_sizes,
            yield_dim=yield_dim,
            output_dim=Y_train.shape[1],
            params=params,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-6,
                patience=int(params["patience"]),
                verbose=0,
                restore_best_weights=True,
            )
        ]

        fit_inputs = macro_train_scaled + [X_yield_train_scaled]

        history = model.fit(
            fit_inputs,
            Y_train,
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            validation_split=float(params["validation_split"]),
            shuffle=bool(params["shuffle"]),
            verbose=0,
            callbacks=callbacks,
        )

        val_loss = float(np.min(history.history["val_loss"]))
        model.save(model_path)

        with open(hist_path, "wb") as f:
            pickle.dump(val_loss, f)

    else:
        model = tf.keras.models.load_model(model_path)

        with open(hist_path, "rb") as f:
            val_loss = pickle.load(f)

    pred_inputs = macro_test_scaled + [X_yield_test_scaled]
    Ypred = model.predict(pred_inputs, verbose=0)

    return Ypred, val_loss


def run_oos_forecast(X, Y, dates, cfg, model_params, dumploc, ncpus):
    model_name = cfg["model_name"]

    oos_start_ts = pd.Timestamp(cfg["oos_start"])
    start_candidates = np.where(dates >= oos_start_ts)[0]
    if start_candidates.size == 0:
        raise ValueError("No available sample date on or after oos_start.")

    first_oos_idx = int(start_candidates[0])
    oos_indices = list(range(first_oos_idx, X.shape[0]))

    T, M = Y.shape
    nmc = int(cfg["nmc"])
    navg = int(cfg["navg"])

    Y_forecast_all = np.full((T, nmc, M), np.nan)
    Y_forecast_avg = np.full((T, M), np.nan)
    val_loss = np.full((T, nmc), np.nan)

    print(model_name)

    oos_counter = 0
    total_oos = len(oos_indices)
    run_start = time.time()

    for j, i in enumerate(oos_indices, start=1):
        iter_start = time.time()

        X_model, Y_model = U.build_model_input(X, Y, i, cfg["horizon"])
        if X_model is None:
            continue

        oos_counter += 1
        refit = (oos_counter == 1) or (oos_counter % int(cfg["hyper_freq"]) == 0)

        outputs = U.run_parallel_seeds(
            NNMacroGroupModel,
            ncpus,
            nmc,
            X_model,
            Y_model,
            params=model_params,
            refit=refit,
            dumploc=dumploc,
        )

        val_loss[i, :] = np.array([outputs[k][1] for k in range(nmc)], dtype=float)
        Y_forecast_all[i, :, :] = np.concatenate([outputs[k][0] for k in range(nmc)], axis=0)

        seed_order = np.argsort(val_loss[i, :])
        Y_forecast_avg[i, :] = np.mean(Y_forecast_all[i, seed_order[:navg], :], axis=0)

        current_best_val = np.nanmin(val_loss[i, :])

        if (j == 1) or (j % 12 == 0) or (j == total_oos):
            iter_elapsed = time.time() - iter_start
            total_elapsed = time.time() - run_start
            r2_now = np.array([U.r2_oos(Y[:, k], Y_forecast_avg[:, k]) for k in range(M)])

            print(
                f"[{j:4d}/{total_oos}] "
                f"date={dates[i].strftime('%Y-%m-%d')} | "
                f"refit={refit} | "
                f"val={current_best_val:10.6f} | "
                f"iter={iter_elapsed:7.2f}s | "
                f"elapsed={total_elapsed:8.1f}s"
            )
            print("  R2OOS:", np.round(r2_now, 4))

    return {
        f"ValLoss_{model_name}": val_loss,
        f"Y_forecast_{model_name}": Y_forecast_all,
        f"Y_forecast_agg_{model_name}": Y_forecast_avg,
        f"MSE_{model_name}": np.nanmean(np.square(Y - Y_forecast_avg), axis=0),
        f"R2OOS_{model_name}": np.array([U.r2_oos(Y[:, k], Y_forecast_avg[:, k]) for k in range(M)]),
        f"R2OOS_pval_{model_name}": np.array([U.r2_oos_pvalue(Y[:, k], Y_forecast_avg[:, k]) for k in range(M)]),
    }


def run_experiment(custom_config=None):
    cfg = copy.deepcopy(custom_config if custom_config is not None else BASE_CONFIG)

    cfg["feature_groups"] = (
        list(cfg["branch_config"]["yield"]["feature_groups"]) +
        list(cfg["branch_config"]["macro"]["feature_groups"])
    )
    cfg["model_name"] = cfg.get("model_name", "NNMacroGroup")

    ncpus = U.get_cpu_count()
    dumploc = U.make_dump_dir()

    print("CPU count is:", ncpus)

    Y_df, X_yield_df, X_macro_group_dfs, macro_group_names = load_dataset(
        mat_path=cfg["mat_path"],
        branch_config=cfg["branch_config"],
        target_group=cfg["target_group"],
    )

    X_macro_parts = [gdf.to_numpy(dtype=float) for gdf in X_macro_group_dfs]
    group_sizes = [x.shape[1] for x in X_macro_parts]
    X_macro = np.concatenate(X_macro_parts, axis=1)

    X_yield = X_yield_df.to_numpy(dtype=float)
    if X_yield.ndim == 1:
        X_yield = X_yield.reshape(-1, 1)

    X = np.concatenate([X_macro, X_yield], axis=1)
    Y = Y_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(Y_df.index)

    model_params = {
        "horizon": int(cfg["horizon"]),
        "group_sizes": group_sizes,
        "yield_dim": int(X_yield.shape[1]),
        "archi_macro": list(cfg["branch_config"]["macro"]["archi"]),
        "archi_yield": list(cfg["branch_config"]["yield"]["archi"]),
        "dropout": float(cfg["params"]["dropout"]),
        "l1l2": list(np.asarray(cfg["params"]["l1l2"], dtype=float).reshape(-1)[:2]),
        "learning_rate": float(cfg["params"]["learning_rate"]),
        "decay_rate": float(cfg["params"]["decay_rate"]),
        "momentum": float(cfg["params"]["momentum"]),
        "nesterov": bool(cfg["params"]["nesterov"]),
        "epochs": int(cfg["params"]["epochs"]),
        "patience": int(cfg["params"]["patience"]),
        "batch_size": int(cfg["params"]["batch_size"]),
        "validation_split": float(cfg["params"]["validation_split"]),
        "shuffle": bool(cfg["params"]["shuffle"]),
        "loss_name": str(cfg["params"]["loss_name"]),
        "huber_delta": float(cfg["params"]["huber_delta"]),
    }

    X_df = pd.concat([pd.concat(X_macro_group_dfs, axis=1), X_yield_df], axis=1)

    save_dict = U.build_save_dict(cfg, X_df, Y_df, Y)
    save_dict["BranchConfigJSON"] = json.dumps(cfg["branch_config"])
    save_dict["MacroGroupNamesJSON"] = json.dumps(macro_group_names)
    save_dict["GroupSizes"] = np.asarray(group_sizes, dtype=int)
    save_dict["ModelParamsJSON"] = json.dumps(model_params)

    save_dict.update(
        run_oos_forecast(
            X=X,
            Y=Y,
            dates=dates,
            cfg=cfg,
            model_params=model_params,
            dumploc=dumploc,
            ncpus=ncpus,
        )
    )

    model_name = cfg["model_name"]
    print("R2OOS:", np.round(save_dict[f"R2OOS_{model_name}"], 4))

    os.makedirs("results_macro_group", exist_ok=True)
    out_file = os.path.join("results_macro_group", result_name(cfg) + ".mat")
    sio.savemat(out_file, save_dict)

    print("Saved to", out_file)
    U.safe_rmtree(dumploc)

    return save_dict, out_file


if __name__ == "__main__":
    run_experiment(BASE_CONFIG)