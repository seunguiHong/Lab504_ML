"""
PyTorch-기반 전략 (ARM + Simple DNN)
────────────────────────────────────────────────────────────────────
• TorchMLP              : 단순 feed-forward MLP
• MultiBranchNet        : N-branch encoder + softmax gate + head
• AdditiveResidualModel : Base(선형) + Residual(MLP)
• ARMStrategy           : Residual/Hybrid 전략
• TorchDNNStrategy      : 순수 DNN 전략
"""

from __future__ import annotations
from .strategies import BaseStrategy

from typing import Any, Dict, List, Tuple, Optional, Iterable, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, Callback
from skorch.dataset import ValidSplit
from skorch.utils import to_device

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _first(grid_values, fallback):
    if grid_values is None:
        return fallback
    if isinstance(grid_values, (list, tuple)):
        return grid_values[0] if len(grid_values) > 0 else fallback
    return grid_values

def _defaults_from_grid(grid: Dict[str, list], prefix: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    plen = len(prefix)
    for k, v in grid.items():
        if k.startswith(prefix):
            sk = k[plen:]
            out[sk] = _first(v, defaults.get(sk))
    for k, v in defaults.items():
        out.setdefault(k, v)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 1) TorchMLP / SafeNet
# ─────────────────────────────────────────────────────────────────────────────
class TorchMLP(nn.Module):
    def __init__(self, num_feat: int, num_out: int, hidden: Tuple[int, ...] = (32, 16), dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = num_feat
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, num_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SafeNet(NeuralNetRegressor):
    def fit(self, X, y=None, **kw):                         # type: ignore[override]
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy(dtype=np.float32, copy=False)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype=np.float32, copy=False)
        return super().fit(X, y, **kw)

    def __getstate__(self):
        state = super().__getstate__()
        for k in ["history","_dataset_train","_dataset_valid","_optimizer","_criterion","_callbacks"]:
            state.pop(k, None)
        return state

# ─────────────────────────────────────────────────────────────────────────────
# 2) MultiBranchNet / SafeNetNBranchLR  (현재 ARM은 MLP-residual만 사용)
# ─────────────────────────────────────────────────────────────────────────────
def _mlp_with_dim(in_dim: int, hidden: Tuple[int, ...], drop: float) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
        d = h
    return nn.Sequential(*layers), (d if hidden else in_dim)

class MultiBranchNet(nn.Module):
    def __init__(self, branches: List[Dict], n_out: int, d_merge: Optional[int] = None, head_hidden: int = 16, head_drop: float = 0.0):
        super().__init__()
        self.n_out = n_out

        self.branch_idx:   List[List[int]] = []
        self.branch_encoders = nn.ModuleList()
        self.branch_projs    = nn.ModuleList()
        out_dims: List[int]  = []

        for br in branches:
            idx   = br["idx"]
            hid   = tuple(br.get("hidden", ()))
            drop  = float(br.get("drop", 0.0))
            enc, d_out = _mlp_with_dim(len(idx), hid, drop)
            self.branch_idx.append(idx)
            self.branch_encoders.append(enc)
            out_dims.append(d_out)

        if len(out_dims) == 0:
            self.d_merge = int(d_merge or n_out)
            self.alpha   = None
        else:
            self.d_merge = int(d_merge or max(out_dims))
            for d in out_dims:
                self.branch_projs.append(nn.Identity() if d == self.d_merge else nn.Linear(d, self.d_merge))
            self.alpha = nn.Parameter(torch.zeros(len(branches)))

        self.head = nn.Sequential(
            nn.Linear(self.d_merge, head_hidden), nn.ReLU(), nn.Dropout(head_drop),
            nn.Linear(head_hidden, n_out),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        dev = next(self.parameters()).device
        if X.device != dev:
            X = X.to(dev, non_blocking=True)

        if len(self.branch_encoders) == 0:
            h = X.new_zeros(X.size(0), self.d_merge)
        else:
            zs = []
            for enc, proj, idx in zip(self.branch_encoders, self.branch_projs, self.branch_idx):
                zi = proj(enc(X[:, idx]))
                zs.append(zi)
            w = torch.softmax(self.alpha, dim=0)
            h = torch.stack(zs, 0)
            h = torch.einsum("r, rbd -> bd", w, h)
        return self.head(h)

class SafeNetNBranchLR(NeuralNetRegressor):
    def __init__(self, *args, lr_br: Optional[List[float]] = None, lr_head: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_br   = lr_br
        self.lr_head = lr_head

    def fit(self, X, y=None, **kw):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy(dtype=np.float32, copy=False)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype=np.float32, copy=False)
        return super().fit(X, y, **kw)

    def initialize_optimizer(self):
        super().initialize_module()
        mdl: MultiBranchNet = self.module_  # type: ignore[assignment]
        opt_kw = self.get_params_for("optimizer__").copy()
        base_lr = opt_kw.pop("lr", 1e-3)

        groups = []
        n_br = len(mdl.branch_encoders)
        lrs = self.lr_br if (self.lr_br and len(self.lr_br) > 0) else [base_lr] * n_br
        if len(lrs) == 1 and n_br > 1:
            lrs = lrs * n_br
        for i in range(n_br):
            params_i = list(mdl.branch_encoders[i].parameters()) + list(mdl.branch_projs[i].parameters())
            if params_i:
                groups.append({"params": params_i, "lr": float(lrs[i])})

        head_params = list(mdl.head.parameters())
        if getattr(mdl, "alpha", None) is not None:
            head_params += [mdl.alpha]
        groups.append({"params": head_params, "lr": float(self.lr_head or base_lr)})

        if isinstance(self.optimizer, tuple):
            opt_cls, user_kw = self.optimizer
            all_kw = {**opt_kw, **user_kw}
        else:
            opt_cls = self.optimizer
            all_kw = opt_kw
        all_kw.pop("lr", None)
        self.optimizer_ = opt_cls(groups, **all_kw)
        return self

    def __getstate__(self):
        state = super().__getstate__()
        for k in ["history","_dataset_train","_dataset_valid","_optimizer","_criterion","_callbacks"]:
            state.pop(k, None)
        return state

    def infer(self, x, **fit_params):
        x = to_device(x, self.device)
        return super().infer(x, **fit_params)

# ─────────────────────────────────────────────────────────────────────────────
# 3) AdditiveResidualModel
# ─────────────────────────────────────────────────────────────────────────────
class AdditiveResidualModel(BaseEstimator, RegressorMixin):
    def __init__(self,
                 base_on: bool = True,
                 base_cols: Optional[List[str]] = None,
                 target_cols: Optional[List[str]] = None,
                 residual_model: Optional[Any] = None,
                 base_model: Optional[Any] = None,
                 feature_cols: Optional[List[str]] = None,
                 standardize_res: bool = True,
                 seed: int = 0):
        self.base_on = base_on
        self.base_cols = base_cols
        self.target_cols = target_cols
        self.residual_model = residual_model
        self.base_model = base_model
        self.feature_cols = feature_cols
        self.standardize_res = standardize_res
        self.seed = seed

        self._scaler: Optional[StandardScaler] = None
        self._feat_cols_used: List[str] = []
        self._target_dim: int = 0
        self._base_model_fitted_ = None
        self._residual_model_fitted_ = None

    @staticmethod
    def _to_2d(a) -> np.ndarray:
        if isinstance(a, (pd.DataFrame, pd.Series)):
            a = a.to_numpy()
        a = np.asarray(a)
        if a.ndim == 1: a = a.reshape(-1, 1)
        return a.astype(np.float32, copy=False)

    @staticmethod
    def _numeric_cols(df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[str]:
        base = (cols if cols is not None and len(cols) > 0 else list(df.columns))
        return [c for c in base if c in df.columns and np.issubdtype(df[c].dtype, np.number)]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if isinstance(y, pd.Series): y = y.to_frame()

        target_cols = list(self.target_cols or y.columns)
        base_cols   = list(self.base_cols or [])
        feat_cols   = list(self.feature_cols) if self.feature_cols else None

        np.random.seed(int(self.seed))
        torch.manual_seed(int(self.seed))

        Y = self._to_2d(y[target_cols])
        self._target_dim = Y.shape[1]

        if self.base_on:
            Xb = self._to_2d(X[self._numeric_cols(X, base_cols)])
        else:
            Xb = np.zeros((len(X), 1), dtype=np.float32)

        if feat_cols is None:
            self._feat_cols_used = self._numeric_cols(X)
        else:
            self._feat_cols_used = self._numeric_cols(X, feat_cols)
        Xres = self._to_2d(X[self._feat_cols_used]) if self._feat_cols_used else np.zeros((len(X), 1), np.float32)

        if self.base_on:
            base_model = self.base_model if self.base_model else LinearRegression()
            base_model.fit(Xb, Y)
            self._base_model_fitted_ = base_model
            U = Y - base_model.predict(Xb).astype(np.float32)
        else:
            self._base_model_fitted_ = None
            U = Y

        if bool(self.standardize_res) and Xres.shape[1] > 0:
            self._scaler = StandardScaler().fit(Xres)
            Z = self._scaler.transform(Xres)
        else:
            self._scaler = None
            Z = Xres

        if self.residual_model is not None and Z.shape[1] > 0:
            self.residual_model.fit(Z, U)
            self._residual_model_fitted_ = self.residual_model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.base_on and self._base_model_fitted_:
            Xb = self._to_2d(X[self._numeric_cols(X, list(self.base_cols or []))])
            Y_base = self._base_model_fitted_.predict(Xb).astype(np.float32)
        else:
            Y_base = np.zeros((len(X), self._target_dim), dtype=np.float32)

        if self._residual_model_fitted_ and self._feat_cols_used:
            Xr = self._to_2d(X[self._feat_cols_used])
            Z = self._scaler.transform(Xr) if self._scaler else Xr
            Y_res = self._residual_model_fitted_.predict(Z).astype(np.float32)
        else:
            Y_res = np.zeros((len(X), self._target_dim), dtype=np.float32)
        return Y_base + Y_res

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        target_cols = list(self.target_cols or y.columns)  # ← 안전 처리
        Ytrue = y[target_cols].to_numpy(dtype=np.float32)
        Yhat = self.predict(X)
        return -float(np.mean((Ytrue - Yhat) ** 2))

# ─────────────────────────────────────────────────────────────────────────────
# 4) ARM Strategy  (Residual = MLP, Grid는 유효 파라미터만)
# ─────────────────────────────────────────────────────────────────────────────
class ARMStrategy(BaseStrategy):
    _DEFAULT_GRID = {
        "arm__residual_model__module__hidden":          [(32, 16), (64, 32)],
        "arm__residual_model__module__dropout":         [0.1, 0.2],
        "arm__residual_model__optimizer__lr":           [1e-3, 5e-4],
        "arm__residual_model__optimizer__weight_decay": [0.0, 5e-5, 5e-4],
        "arm__residual_model__batch_size":              [32],
        "arm__residual_model__max_epochs":              [100],
        "arm__residual_model__train_split":             [ValidSplit(0.2, stratified=False)],
    }

    def __init__(self, core, option: Dict, params_grid=None):
        super().__init__(core, option, params_grid)
        self.option = option

    def build_pipeline(self) -> Pipeline:
        opt: Dict[str, Any] = self.option or {}
        # grid -> 기본값 사전
        self.grid = getattr(self, "params_grid", None) or self._DEFAULT_GRID
        hp = _defaults_from_grid(self.grid, "arm__residual_model__", {
            "module__hidden": (32, 16),
            "module__dropout": 0.1,
            "optimizer__lr": 1e-3,
            "optimizer__weight_decay": 0.0,
            "batch_size": 32,
            "max_epochs": 100,
            "train_split": ValidSplit(0.2, stratified=False),
        })

        # patience는 grid 대상 아님(콜백 내부 값) → option으로 제어
        patience = int(opt.get("patience", 10))

        n_out = 1 if self.core.y.ndim == 1 else self.core.y.shape[1]
        feature_cols: List[str] = list(opt.get("feature_cols", []))
        feat_cols = feature_cols if feature_cols else [
            c for c in self.core.X.columns if np.issubdtype(self.core.X[c].dtype, np.number)
        ]

        residual_model = SafeNet(
            module=TorchMLP,
            module__num_feat=len(feat_cols),
            module__num_out=n_out,
            module__hidden=hp["module__hidden"],
            module__dropout=float(hp["module__dropout"]),
            optimizer=torch.optim.Adam,
            optimizer__lr=float(hp["optimizer__lr"]),
            optimizer__weight_decay=float(hp["optimizer__weight_decay"]),
            criterion=nn.MSELoss,
            batch_size=int(hp["batch_size"]),
            max_epochs=int(hp["max_epochs"]),
            train_split=hp["train_split"],
            callbacks=[EarlyStopping(monitor="valid_loss",
                                     patience=patience,
                                     threshold=1e-5,
                                     lower_is_better=True)],
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0,
        )

        arm = AdditiveResidualModel(
            base_on=bool(opt.get("base_on", True)),
            base_cols=list(opt.get("base_cols", [])),
            target_cols=list(opt.get("target_cols", [])),
            residual_model=residual_model,
            base_model=opt.get("base_model", None),
            feature_cols=(feature_cols if feature_cols else None),
            standardize_res=bool(opt.get("standardize_res", True)),
            seed=int(opt.get("seed", 0)),
        )

        return Pipeline([("arm", arm)])

# ─────────────────────────────────────────────────────────────────────────────
# 5) TorchDNN Strategy  (Grid는 유효 파라미터만)
# ─────────────────────────────────────────────────────────────────────────────
class TorchDNNStrategy(BaseStrategy):
    _DEFAULT_GRID = {
        "dnn__module__hidden":          [(32, 16), (64, 32)],
        "dnn__module__dropout":         [0.1, 0.2],
        "dnn__optimizer__lr":           [1e-3, 5e-4],
        "dnn__optimizer__weight_decay": [0.0, 5e-5, 5e-4],
        "dnn__batch_size":              [32],
        "dnn__max_epochs":              [100],
        "dnn__train_split":             [ValidSplit(0.2, stratified=False)],
    }

    def __init__(self, core, option: Dict, params_grid=None):
        super().__init__(core, option, params_grid)
        self.option = option

    @staticmethod
    def _to32():
        return FunctionTransformer(
            lambda z: (
                z.to_numpy(dtype=np.float32, copy=False)
                if isinstance(z, (pd.DataFrame, pd.Series))
                else z.astype(np.float32, copy=False)
            ),
            validate=False,
        )

    def build_pipeline(self) -> Pipeline:
        opt: Dict[str, Any] = self.option or {}
        # grid -> 기본값 사전
        self.grid = getattr(self, "params_grid", None) or self._DEFAULT_GRID
        hp = _defaults_from_grid(self.grid, "dnn__", {
            "module__hidden": (32, 16),
            "module__dropout": 0.1,
            "optimizer__lr": 1e-3,
            "optimizer__weight_decay": 0.0,
            "batch_size": 32,
            "max_epochs": 100,
            "train_split": ValidSplit(0.2, stratified=False),
        })

        # patience는 grid 대상 아님 → option으로 제어
        patience = int(opt.get("patience", 10))

        n_out = 1 if self.core.y.ndim == 1 else self.core.y.shape[1]

        scaler_choice = str(opt.get("scaler", "standard")).lower()
        if scaler_choice == "standard":
            scaler = StandardScaler()
        elif scaler_choice == "minmax":
            scaler = MinMaxScaler(opt.get("minmax_range", (-1, 1)))
        else:
            scaler = None

        net = SafeNet(
            module=TorchMLP,
            module__num_feat=self.core.X.shape[1],
            module__num_out=n_out,
            module__hidden=hp["module__hidden"],
            module__dropout=float(hp["module__dropout"]),
            optimizer=torch.optim.Adam,
            optimizer__lr=float(hp["optimizer__lr"]),
            optimizer__weight_decay=float(hp["optimizer__weight_decay"]),
            criterion=nn.MSELoss,
            batch_size=int(hp["batch_size"]),
            max_epochs=int(hp["max_epochs"]),
            train_split=hp["train_split"],
            callbacks=[EarlyStopping(monitor="valid_loss",
                                     patience=patience,
                                     threshold=1e-5,
                                     lower_is_better=True)],
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0,
        )

        steps = [("to32", self._to32())]
        if scaler is not None:
            steps.append(("sc", scaler))
        steps.append(("dnn", net))

        return Pipeline(steps)