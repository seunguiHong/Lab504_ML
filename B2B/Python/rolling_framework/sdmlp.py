from __future__ import annotations
from typing import Dict, List, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from .strategies import Strategy


class MLPNet(nn.Module):
    """
    Fully-connected MLP for multi-output regression.

    Architecture:
        input_dim -> hidden_sizes[0] -> ... -> hidden_sizes[-1] -> out_dim
        with ReLU activations and optional dropout after each hidden layer.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: List[int], dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Backwards-compatible alias
SDMLPNet = MLPNet


class TorchPlainMLPStrategy(Strategy):
    """
    Plain PyTorch MLP strategy for multi-output regression.

    This strategy uses a single MLP y = f(X) without explicit slope structure.
    It is used as:
        - stand-alone "MLP" model on all features, or
        - residual model inside ARM / CSARM (using selected feature_cols).

    All feature scaling must be handled outside this class.
    """

    def __init__(
        self,
        target_cols: List[str],
        feature_cols: List[str] | None,
        hidden_sizes: List[int],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_epochs: int = 200,
        batch_size: int = 64,
        dropout: float = 0.0,
        device: str = "auto",
        target_weights: Union[None, List[float], np.ndarray] = None,
    ):
        super().__init__(target_cols)
        self.feature_cols = list(feature_cols) if feature_cols else None
        self.hidden_sizes = list(hidden_sizes)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.dropout = float(dropout)
        self.device_str = device

        # target-wise weights (shape: [n_targets]), normalized to sum=1
        if target_weights is None:
            self.target_weights = None
        else:
            w = np.asarray(target_weights, dtype=float)
            if w.shape[0] != len(self.target_cols):
                raise ValueError("target_weights length must match number of target_cols")
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            s = w.sum()
            self.target_weights = None if s == 0.0 else (w / s)

    def _resolve_device(self) -> torch.device:
        """Resolve torch device based on device_str."""
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        """
        Fit MLP on (X_tr, y_tr) and predict for x_te.

        If feature_cols is not None, only those columns are used as inputs.
        All scaling or transformations of X should be done before calling this method.
        """
        if X_tr.shape[0] == 0:
            raise ValueError("training window is empty in TorchPlainMLPStrategy")

        # select feature block
        if self.feature_cols is not None and len(self.feature_cols) > 0:
            X_block = X_tr[self.feature_cols]
            x_block = x_te[self.feature_cols]
        else:
            X_block = X_tr
            x_block = x_te

        X_np = X_block.to_numpy(float)
        Y_np = y_tr[self.target_cols].to_numpy(float)
        x_np = np.asarray(x_block, dtype=float).reshape(1, -1)

        n, in_dim = X_np.shape
        out_dim = len(self.target_cols)

        device = self._resolve_device()

        X_tensor = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        Y_tensor = torch.as_tensor(Y_np, dtype=torch.float32, device=device)
        x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=device)

        net = MLPNet(in_dim, out_dim, hidden_sizes=self.hidden_sizes, dropout=self.dropout).to(device)
        loss_fn = nn.MSELoss(reduction="none")
        opt = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        net.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = net(xb)  # (batch, n_targets)
                if self.target_weights is None:
                    loss = loss_fn(pred, yb).mean()
                else:
                    mse_per_target = loss_fn(pred, yb).mean(dim=0)  # (n_targets,)
                    w = torch.as_tensor(self.target_weights, dtype=torch.float32, device=device)
                    loss = (w * mse_per_target).sum()
                loss.backward()
                opt.step()

        net.eval()
        with torch.no_grad():
            y_hat = net(x_tensor).cpu().numpy()[0]

        return pd.Series(y_hat, index=self.target_cols)


class TorchSDMLPStrategy(Strategy):
    """
    PyTorch slope-direct MLP strategy with learnable slope parameters.

    Model structure for each target j:
        y_hat_{t,j} = a_j + b_j * s_{t,j} + f_j(z_t),

    where:
        - s_{t,j} is the slope regressor specified by slope_map[y_j],
        - z_t are additional features in feature_cols,
        - a_j, b_j are learnable parameters (initialized by per-target OLS
          and optionally scaled by slope_scale),
        - f_j is the MLPNet output for target j.

    Training minimizes MSE over the full prediction:
        L = (1/N) * sum_t || y_t - y_hat_t ||^2

    All scaling must be handled outside this class.
    """

    def __init__(
        self,
        target_cols: List[str],
        slope_map: Dict[str, str],
        feature_cols: List[str],
        hidden_sizes: List[int],
        lr: float = 1e-3,              # MLP learning rate
        weight_decay: float = 0.0,
        max_epochs: int = 200,
        batch_size: int = 64,
        dropout: float = 0.0,
        slope_scale: Union[float, Dict[str, float]] = 1.0,
        slope_lr: float | None = None, # learning rate for a,b (None => same as lr)
        device: str = "auto",
        target_weights: Union[None, List[float], np.ndarray] = None,
    ):
        super().__init__(target_cols)
        self.slope_map = dict(slope_map)
        self.feature_cols = list(feature_cols)
        self.hidden_sizes = list(hidden_sizes)

        self.lr = float(lr)
        self.slope_lr = float(slope_lr) if slope_lr is not None else float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.dropout = float(dropout)
        self.slope_scale = slope_scale
        self.device_str = device

        if target_weights is None:
            self.target_weights = None
        else:
            w = np.asarray(target_weights, dtype=float)
            if w.shape[0] != len(self.target_cols):
                raise ValueError("target_weights length must match number of target_cols")
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            s = w.sum()
            self.target_weights = None if s == 0.0 else (w / s)

    # ---------------- internal helpers ----------------

    @staticmethod
    def _ols_ab(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        """
        Compute intercept and slope for y ~ 1 + x via least squares.
        """
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0])
        b = float(beta[1])
        return a, b

    def _scale_for(self, y_col: str) -> float:
        """
        Return slope scaling factor k_j for target y_col (used only in initialization).
        """
        if isinstance(self.slope_scale, dict):
            return float(self.slope_scale.get(y_col, 1.0))
        return float(self.slope_scale)

    def _resolve_device(self) -> torch.device:
        """
        Resolve torch device based on device_str.
        """
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    # ---------------- main API ----------------

    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        """
        Fit SDMLP on training window and predict for a single test point.

        Inputs:
            X_tr : full feature DataFrame (including slope columns)
            y_tr : target DataFrame with columns target_cols
            x_te : single-row Series from X (same columns as X_tr)

        Returns:
            Series of predictions indexed by target_cols.
        """
        n = X_tr.shape[0]
        T = len(self.target_cols)
        K = len(self.feature_cols)

        if n == 0:
            raise ValueError("training window is empty in TorchSDMLPStrategy")

        # slope matrix S_tr (n x T) and test slope vector s_te (1 x T)
        S_tr = np.zeros((n, T), dtype=float)
        s_te = np.zeros((1, T), dtype=float)
        for j, y_col in enumerate(self.target_cols):
            s_col = self.slope_map[y_col]
            S_tr[:, j] = X_tr[s_col].to_numpy(float)
            s_te[0, j] = float(x_te[s_col])

        # feature matrix Z_tr (n x K) and test feature z_te (1 x K)
        if self.feature_cols:
            Z_tr = X_tr[self.feature_cols].to_numpy(float)
            z_te = x_te[self.feature_cols].to_numpy(float).reshape(1, -1)
        else:
            Z_tr = np.zeros((n, 0), dtype=float)
            z_te = np.zeros((1, 0), dtype=float)

        # target matrix
        Y_tr = y_tr[self.target_cols].to_numpy(float)

        # initialize a,b by per-target OLS with optional slope_scale
        a_init = np.zeros(T, dtype=float)
        b_init = np.zeros(T, dtype=float)
        for j, y_col in enumerate(self.target_cols):
            s_col = self.slope_map[y_col]
            a_j, b_j = self._ols_ab(
                y_tr[y_col].to_numpy(float),
                X_tr[s_col].to_numpy(float),
            )
            k_j = self._scale_for(y_col)
            a_init[j] = a_j
            b_init[j] = k_j * b_j

        device = self._resolve_device()

        S_tr_t = torch.as_tensor(S_tr, dtype=torch.float32, device=device)
        Z_tr_t = torch.as_tensor(Z_tr, dtype=torch.float32, device=device)
        Y_tr_t = torch.as_tensor(Y_tr, dtype=torch.float32, device=device)

        s_te_t = torch.as_tensor(s_te, dtype=torch.float32, device=device)
        z_te_t = torch.as_tensor(z_te, dtype=torch.float32, device=device)

        # inner model: learnable a,b plus MLP residual
        class SDMLPModel(nn.Module):
            def __init__(
                self,
                n_targets: int,
                n_features: int,
                hidden_sizes: List[int],
                dropout: float,
                a_init: np.ndarray,
                b_init: np.ndarray,
            ):
                super().__init__()
                # slope parameters per target
                self.a = nn.Parameter(torch.as_tensor(a_init, dtype=torch.float32))
                self.b = nn.Parameter(torch.as_tensor(b_init, dtype=torch.float32))
                # residual MLP block
                if n_features > 0:
                    self.mlp = MLPNet(n_features, n_targets, hidden_sizes=hidden_sizes, dropout=dropout)
                else:
                    self.mlp = None

            def forward(self, S: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
                """
                S: (batch, n_targets), Z: (batch, n_features) or empty.
                """
                slope_part = self.a + self.b * S
                if self.mlp is not None:
                    resid_part = self.mlp(Z)
                else:
                    resid_part = 0.0
                return slope_part + resid_part

        model = SDMLPModel(
            n_targets=T,
            n_features=K,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            a_init=a_init,
            b_init=b_init,
        ).to(device)

        # optimizer with separate param group for (a,b) and for MLP
        param_groups = []
        # a,b group (slope parameters)
        param_groups.append({
            "params": [model.a, model.b],
            "lr": self.slope_lr,
            "weight_decay": 0.0,
        })
        # MLP group (residual network)
        if model.mlp is not None:
            param_groups.append({
                "params": model.mlp.parameters(),
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            })

        opt = optim.Adam(param_groups)
        loss_fn = nn.MSELoss(reduction="none")

        dataset = torch.utils.data.TensorDataset(S_tr_t, Z_tr_t, Y_tr_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.max_epochs):
            for S_b, Z_b, Y_b in loader:
                opt.zero_grad()
                Y_hat_b = model(S_b, Z_b)  # (batch, n_targets)
                if self.target_weights is None:
                    loss = loss_fn(Y_hat_b, Y_b).mean()
                else:
                    mse_per_target = loss_fn(Y_hat_b, Y_b).mean(dim=0)
                    w = torch.as_tensor(self.target_weights, dtype=torch.float32, device=device)
                    loss = (w * mse_per_target).sum()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            Y_hat_te = model(s_te_t, z_te_t).cpu().numpy()[0]

        return pd.Series(Y_hat_te, index=self.target_cols)