from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge


class Strategy(ABC):
    """
    Base class for all forecasting strategies.

    Each strategy implements fit_predict on a rolling window:
        - fit on (X_tr, y_tr)
        - return prediction for a single test point x_te
    """

    def __init__(self, target_cols: List[str]):
        self.target_cols = list(target_cols)

    @abstractmethod
    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        """Fit on training window and predict for x_te."""
        raise NotImplementedError


class SklearnRegStrategy(Strategy):
    """
    Wrapper around a sklearn regressor that supports multi-output regression.

    Feature block is optionally restricted via feature_cols.
    """

    def __init__(
        self,
        target_cols: List[str],
        model: Any,
        feature_cols: Optional[List[str]] = None,
    ):
        super().__init__(target_cols)
        self.model = model
        self.feature_cols = list(feature_cols) if feature_cols is not None else None

    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        if X_tr.shape[0] == 0:
            raise ValueError("training window is empty in SklearnRegStrategy")

        if self.feature_cols is not None and len(self.feature_cols) > 0:
            X_block = X_tr[self.feature_cols]
            x_block = x_te[self.feature_cols]
        else:
            X_block = X_tr
            x_block = x_te

        X_np = X_block.to_numpy(float)
        Y_np = y_tr[self.target_cols].to_numpy(float)
        x_np = np.asarray(x_block, dtype=float).reshape(1, -1)

        self.model.fit(X_np, Y_np)
        y_hat = self.model.predict(x_np).reshape(-1)

        return pd.Series(y_hat, index=self.target_cols)


class ARMStrategy(Strategy):
    """
    Augmented regression model with base linear block + residual block.

    For each target column:
        y = base(X_base) + residual(X_resid)

    - base block: sklearn linear / ridge / lasso / tree
    - residual block: sklearn or TorchPlainMLPStrategy (residual_kind="mlp")
    """

    def __init__(
        self,
        target_cols: List[str],
        base_cols: List[str],
        feature_cols: List[str],
        base_type: str = "ridge",
        base_params: Optional[Dict[str, Any]] = None,
        residual_kind: Optional[str] = None,
        residual_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(target_cols)
        self.base_cols = list(base_cols)
        self.feature_cols = list(feature_cols)
        self.base_type = base_type.lower()
        self.base_params = base_params or {}
        self.residual_kind = residual_kind.lower() if residual_kind is not None else None
        self.residual_params = residual_params or {}

        # base model instance
        if self.base_type == "ols":
            self.base_model = LinearRegression(**self.base_params)
        elif self.base_type == "ridge":
            self.base_model = Ridge(**self.base_params)
        elif self.base_type == "lasso":
            self.base_model = Lasso(**self.base_params)
        elif self.base_type == "rf":
            self.base_model = RandomForestRegressor(**self.base_params)
        elif self.base_type == "etr":
            self.base_model = ExtraTreesRegressor(**self.base_params)
        else:
            raise ValueError(f"Unknown base_type for ARMStrategy: {self.base_type}")

    def _make_residual_strategy(self) -> Optional[Strategy]:
        """Instantiate residual strategy based on residual_kind."""
        if self.residual_kind is None or self.residual_kind == "none":
            return None

        kind = self.residual_kind
        if kind == "ridge":
            model = Ridge(**self.residual_params)
            return SklearnRegStrategy(self.target_cols, model, feature_cols=self.feature_cols)
        if kind == "lasso":
            model = Lasso(**self.residual_params)
            return SklearnRegStrategy(self.target_cols, model, feature_cols=self.feature_cols)
        if kind == "rf":
            model = RandomForestRegressor(**self.residual_params)
            return SklearnRegStrategy(self.target_cols, model, feature_cols=self.feature_cols)
        if kind == "etr":
            model = ExtraTreesRegressor(**self.residual_params)
            return SklearnRegStrategy(self.target_cols, model, feature_cols=self.feature_cols)
        if kind == "mlp":
            # Torch MLP residual
            from .sdmlp import TorchPlainMLPStrategy

            return TorchPlainMLPStrategy(
                target_cols=self.target_cols,
                feature_cols=self.feature_cols,
                hidden_sizes=self.residual_params.get("hidden_sizes", [64, 64]),
                lr=self.residual_params.get("lr", 1e-3),
                weight_decay=self.residual_params.get("weight_decay", 0.0),
                max_epochs=self.residual_params.get("max_epochs", 200),
                batch_size=self.residual_params.get("batch_size", 64),
                dropout=self.residual_params.get("dropout", 0.0),
                device=self.residual_params.get("device", "auto"),
                target_weights=self.residual_params.get("target_weights", None),
            )

        raise ValueError(f"Unknown residual_kind for ARMStrategy: {kind}")

    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        if X_tr.shape[0] == 0:
            raise ValueError("training window is empty in ARMStrategy")

        # base block fit
        Xb_tr = X_tr[self.base_cols]
        Xb_np = Xb_tr.to_numpy(float)
        Y_np = y_tr[self.target_cols].to_numpy(float)
        xb_np = x_te[self.base_cols].to_numpy(float).reshape(1, -1)

        self.base_model.fit(Xb_np, Y_np)
        base_tr = self.base_model.predict(Xb_np)  # (n, n_targets)
        base_te = self.base_model.predict(xb_np).reshape(-1)  # (n_targets,)

        # residual block (optional)
        residual_strategy = self._make_residual_strategy()
        if residual_strategy is None:
            return pd.Series(base_te, index=self.target_cols)

        R_tr = Y_np - base_tr  # residual matrix
        R_df = pd.DataFrame(R_tr, columns=self.target_cols)

        y_hat_resid = residual_strategy.fit_predict(
            X_tr[X_tr.columns],  # residual strategy will select its own feature_cols
            R_df,
            x_te,
        )

        y_hat = base_te + y_hat_resid.to_numpy()
        return pd.Series(y_hat, index=self.target_cols)


class CSARMStrategy(Strategy):
    """
    Campbell–Shiller augmented regression:

        y_{t,j} = a_j + b_j s_{t,j} + residual_model(z_t),

    where:
        - s_{t,j} is the slope regressor given by slope_map[y_j]
        - residual_model is optional and may be sklearn or TorchPlainMLPStrategy.

    If residual_kind is None or "none", this reduces to CS-only regression.
    """

    def __init__(
        self,
        target_cols: List[str],
        slope_map: Dict[str, str],
        feature_cols: List[str],
        residual_kind: Optional[str] = None,
        residual_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(target_cols)
        self.slope_map = dict(slope_map)
        self.feature_cols = list(feature_cols)
        self.residual_kind = residual_kind.lower() if residual_kind is not None else None
        self.residual_params = residual_params or {}

        # per-target (a_j, b_j)
        self.coefs_: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _ols_ab(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0])
        b = float(beta[1])
        return a, b

    def _make_residual_strategy(self) -> Optional[Strategy]:
        if self.residual_kind is None or self.residual_kind == "none":
            return None

        kind = self.residual_kind
        if kind == "ridge":
            model = Ridge(**self.residual_params)
            return SklearnRegStrategy(self.target_cols, model, feature_cols=self.feature_cols)
        if kind == "lasso":
            model = Lasso(**self.residual_params)
            return SklearnRegStrategy(self.target_cols, model, feature_cols=self.feature_cols)
        if kind == "rf":
            model = RandomForestRegressor(**self.residual_params)
            return SklearnRegStrategy(self.target_cols, model, feature_cols=self.feature_cols)
        if kind == "etr":
            model = ExtraTreesRegressor(**self.residual_params)
            return SklearnRegStrategy(self.target_cols, model, feature_cols=self.feature_cols)
        if kind == "mlp":
            from .sdmlp import TorchPlainMLPStrategy

            return TorchPlainMLPStrategy(
                target_cols=self.target_cols,
                feature_cols=self.feature_cols,
                hidden_sizes=self.residual_params.get("hidden_sizes", [64, 64]),
                lr=self.residual_params.get("lr", 1e-3),
                weight_decay=self.residual_params.get("weight_decay", 0.0),
                max_epochs=self.residual_params.get("max_epochs", 200),
                batch_size=self.residual_params.get("batch_size", 64),
                dropout=self.residual_params.get("dropout", 0.0),
                device=self.residual_params.get("device", "auto"),
                target_weights=self.residual_params.get("target_weights", None),
            )

        raise ValueError(f"Unknown residual_kind for CSARMStrategy: {kind}")

    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        if X_tr.shape[0] == 0:
            raise ValueError("training window is empty in CSARMStrategy")

        n = X_tr.shape[0]
        T = len(self.target_cols)

        # estimate CS coefs per target
        S = np.zeros((n, T), dtype=float)
        Y = y_tr[self.target_cols].to_numpy(float)
        for j, y_col in enumerate(self.target_cols):
            s_col = self.slope_map[y_col]
            x = X_tr[s_col].to_numpy(float)
            S[:, j] = x
            a_j, b_j = self._ols_ab(Y[:, j], x)
            self.coefs_[y_col] = {"a": a_j, "b": b_j}

        # base CS fit on training
        base_tr = np.zeros_like(Y)
        for j, y_col in enumerate(self.target_cols):
            a_j = self.coefs_[y_col]["a"]
            b_j = self.coefs_[y_col]["b"]
            base_tr[:, j] = a_j + b_j * S[:, j]

        base_resid = Y - base_tr  # residual matrix
        resid_df = pd.DataFrame(base_resid, columns=self.target_cols)

        # residual block
        residual_strategy = self._make_residual_strategy()
        if residual_strategy is None:
            # CS-only prediction
            y_hat = []
            for y_col in self.target_cols:
                s_col = self.slope_map[y_col]
                s_val = float(x_te[s_col])
                a_j = self.coefs_[y_col]["a"]
                b_j = self.coefs_[y_col]["b"]
                y_hat.append(a_j + b_j * s_val)
            return pd.Series(y_hat, index=self.target_cols)

        # fit residual model on feature_cols -> residuals
        y_hat_resid = residual_strategy.fit_predict(
            X_tr[X_tr.columns],  # residual strategy will select its own feature_cols
            resid_df,
            x_te,
        )

        # final prediction: CS base + residual
        y_hat = []
        for idx, y_col in enumerate(self.target_cols):
            s_col = self.slope_map[y_col]
            s_val = float(x_te[s_col])
            a_j = self.coefs_[y_col]["a"]
            b_j = self.coefs_[y_col]["b"]
            base_val = a_j + b_j * s_val
            y_hat.append(base_val + float(y_hat_resid.iloc[idx]))

        return pd.Series(y_hat, index=self.target_cols)


def make_strategy(name: str, cfg: Dict[str, Any], target_cols: List[str]) -> Strategy:
    """
    Factory for building strategies from configuration.

    name:
        - "OLS", "RIDGE", "LASSO", "RF", "ETR"
        - "ARM", "CSARM"
        - "MLP", "SDMLP"

    cfg:
        strategy-specific configuration dictionary.

    target_cols:
        list of target column names.
    """
    key = name.upper()
    params = cfg.get("params", {})

    # basic sklearn regressors
    if key in ("OLS", "LINEAR"):
        model = LinearRegression(**params)
        feature_cols = cfg.get("feature_cols", None)
        return SklearnRegStrategy(target_cols, model, feature_cols=feature_cols)

    if key == "RIDGE":
        model = Ridge(**params)
        feature_cols = cfg.get("feature_cols", None)
        return SklearnRegStrategy(target_cols, model, feature_cols=feature_cols)

    if key == "LASSO":
        model = Lasso(**params)
        feature_cols = cfg.get("feature_cols", None)
        return SklearnRegStrategy(target_cols, model, feature_cols=feature_cols)

    if key in ("RF", "RANDOMFOREST"):
        model = RandomForestRegressor(**params)
        feature_cols = cfg.get("feature_cols", None)
        return SklearnRegStrategy(target_cols, model, feature_cols=feature_cols)

    if key in ("ETR", "EXTRATREES"):
        model = ExtraTreesRegressor(**params)
        feature_cols = cfg.get("feature_cols", None)
        return SklearnRegStrategy(target_cols, model, feature_cols=feature_cols)

    # ARM (base + residual)
    if key == "ARM":
        return ARMStrategy(
            target_cols=target_cols,
            base_cols=cfg["base_cols"],
            feature_cols=cfg.get("feature_cols", []),
            base_type=cfg.get("base_type", "ridge"),
            base_params=cfg.get("base_params", None),
            residual_kind=cfg.get("residual_kind", None),
            residual_params=cfg.get("residual_params", None),
        )

    # CSARM
    if key == "CSARM":
        return CSARMStrategy(
            target_cols=target_cols,
            slope_map=cfg["slope_map"],
            feature_cols=cfg.get("feature_cols", []),
            residual_kind=cfg.get("residual_kind", None),
            residual_params=cfg.get("residual_params", None),
        )

    # Plain MLP (Torch)
    if key == "MLP":
        from .sdmlp import TorchPlainMLPStrategy

        return TorchPlainMLPStrategy(
            target_cols=target_cols,
            feature_cols=cfg.get("feature_cols", None),
            hidden_sizes=params.get("hidden_sizes", [64, 64]),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 0.0),
            max_epochs=params.get("max_epochs", 200),
            batch_size=params.get("batch_size", 64),
            dropout=params.get("dropout", 0.0),
            device=params.get("device", "auto"),
            target_weights=params.get("target_weights", None),
        )

    # SDMLP (Torch slope-direct MLP)
    if key == "SDMLP":
        from .sdmlp import TorchSDMLPStrategy

        return TorchSDMLPStrategy(
            target_cols=target_cols,
            slope_map=cfg["slope_map"],
            feature_cols=cfg.get("feature_cols", []),
            hidden_sizes=params.get("hidden_sizes", [64, 64]),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 0.0),
            max_epochs=params.get("max_epochs", 200),
            batch_size=params.get("batch_size", 64),
            dropout=params.get("dropout", 0.0),
            slope_scale=params.get("slope_scale", 1.0),
            slope_lr=params.get("slope_lr", None),
            device=params.get("device", "auto"),
            target_weights=params.get("target_weights", None),
        )

    raise ValueError(f"Unknown strategy name: {name}")