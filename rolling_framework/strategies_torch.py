"""
PyTorch-기반 전략 모음
────────────────────────────────────────────────────────────────────
• TorchMLP                    : 단순 feed-forward MLP
• TorchDNNStrategy            : 개별 MLP (GridSearch 대상)
• TorchUnifiedStrategy       : N branch MLP + Direct Version
"""

from __future__ import annotations
from .strategies import BaseStrategy          # 기존 파일의 BaseStrategy

# ─────────────────────────  공통 IMPORT  ─────────────────────────
import copy
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, EarlyStopping
from skorch.dataset import ValidSplit
from sklearn.linear_model import LinearRegression
from skorch.utils import to_device
# ─────────── ① MLP 모듈 ───────────
class TorchMLP(nn.Module):
    """Multi-layer perceptron (ReLU + Dropout)."""
    def __init__(
        self,
        num_feat: int,
        num_out:  int,
        hidden:   Tuple[int, ...] = (32, 16),
        dropout:  float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d = num_feat
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, num_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


# ─────────── ② skorch 래퍼 ───────────
class SafeNet(NeuralNetRegressor):
    """pandas → float32 NumPy 변환을 자동으로 처리하는 래퍼."""
    def fit(self, X, y=None, **kw):                         # type: ignore[override]
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy(dtype=np.float32, copy=False)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype=np.float32, copy=False)
        return super().fit(X, y, **kw)
    
    def __getstate__(self):
        state = super().__getstate__()      # skorch 기본 state
        # 학습이 끝난 뒤에만 존재하는 무거운 객체를 제거
        purge = [
            "history",           # pandas DataFrame
            "_dataset_train", "_dataset_valid",
            "_optimizer", "_criterion", "_callbacks",
        ]
        for k in purge:
            state.pop(k, None)
        return state


# ─────────── ③ 단일-MLP 전략 ───────────
class TorchDNNStrategy(BaseStrategy):
    """단일 MLP + Adam + EarlyStopping(valid_loss)"""

    _DEFAULT_GRID = {
        "dnn__module__hidden":          [(32, 16), (64, 32)],
        "dnn__module__dropout":         [0.0, 0.2],
        "dnn__optimizer__lr":           [1e-3, 5e-4],
        "dnn__optimizer__weight_decay": [0.0, 5e-4],   # L2
    }

    def build_pipeline(self) -> Pipeline:
        n_feat = self.core.X.shape[1]
        n_out  = 1 if self.core.y.ndim == 1 else self.core.y.shape[1]

        net = SafeNet(
            module                = TorchMLP,
            module__num_feat      = n_feat,
            module__num_out       = n_out,
            module__hidden        = (32, 16),
            module__dropout       = 0.1,
            optimizer             = torch.optim.Adam,
            optimizer__lr         = 1e-3,
            optimizer__weight_decay = 0.0,
            criterion             = nn.MSELoss,
            batch_size            = 16,
            max_epochs            = 100,
            train_split           = ValidSplit(0.2, stratified=False),
            callbacks             = [
                EarlyStopping(monitor="valid_loss",
                              patience=10,
                              threshold=1e-5,
                              lower_is_better=True)
            ],
            device                = "cuda" if torch.cuda.is_available() else "cpu",
            verbose               = 0,
        )

        pipe = Pipeline([
            ("to32", FunctionTransformer(
        lambda z: (
            z.to_numpy(dtype=np.float32, copy=False)         # DataFrame/Series → np.ndarray
            if isinstance(z, (pd.DataFrame, pd.Series))      # 아니면 그대로
            else z.astype(np.float32, copy=False)
        ),
        validate=False   # <- DataFrame 도 허용
        )),
            ("sc",   MinMaxScaler((-1, 1))),
            ("dnn",  net),
        ])
        # Grid 설정
        self.grid = getattr(self, "params_grid", None) or self._DEFAULT_GRID
        return pipe

# ─────────────────────────────────────────────────────────────────────────────
# Direct Mapping + Pretrained OLS + N Branch (Gate merge)
# ─────────────────────────────────────────────────────────────────────────────
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LinearRegression

from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, EarlyStopping
from skorch.dataset import ValidSplit
from skorch.utils import to_device

from .strategies import BaseStrategy  # ← 꼭 존재해야 함


# MLP 유틸
def _mlp_with_dim(in_dim: int, hidden: Tuple[int, ...], drop: float) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
        d = h
    return nn.Sequential(*layers), (d if hidden else in_dim)


# Direct map OLS 초기화
class PretrainDirectMap(Callback):
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        if not hasattr(net, "module_") or net.module_ is None:
            net.initialize()
        mdl = net.module_

        if mdl.direct_in_idx is None or len(mdl.direct_in_idx) == 0:
            return

        def _to_np(a):
            if hasattr(a, "to_numpy"):
                return a.to_numpy(dtype=np.float32, copy=False)
            return np.asarray(a, dtype=np.float32)

        Xnp, ynp = _to_np(X), _to_np(y)
        # 단일 타깃 대비(선택)
        y2d = ynp if ynp.ndim == 2 else ynp.reshape(-1, 1)

        M = len(mdl.direct_in_idx)
        for j in range(M):
            in_j  = mdl.direct_in_idx[j]
            out_j = mdl.direct_out_idx[j]
            xj = Xnp[:, [in_j]]
            yj = y2d[:, out_j]
            lr = LinearRegression().fit(xj, yj)
            w, b = float(lr.coef_.ravel()[0]), float(lr.intercept_)
            with torch.no_grad():
                mdl.w_direct.data[j] = w
                mdl.b_direct.data[j] = b

        if mdl.freeze_direct:
            mdl.w_direct.requires_grad_(False)
            mdl.b_direct.requires_grad_(False)


# N-Branch + Gate + Head + Direct Residual
class MultiBranchNet(nn.Module):
    def __init__(
        self,
        branches: List[Dict],
        n_out: int,
        direct_in_idx: Optional[List[int]] = None,
        direct_out_idx: Optional[List[int]] = None,
        freeze_direct: bool = False,
        d_merge: Optional[int] = None,
        head_hidden: int = 16,
        head_drop: float = 0.0,
    ):
        super().__init__()
        assert len(branches) >= 1, "at least one branch is required"
        self.n_out = n_out
        self.freeze_direct = freeze_direct

        # branches
        self.branch_idx: List[List[int]] = []
        encoders: List[nn.Sequential] = []
        out_dims: List[int] = []
        for br in branches:
            idx   = br["idx"]
            hid   = tuple(br.get("hidden", (32, 16)))
            drop  = float(br.get("drop", 0.1))
            enc, d_out = _mlp_with_dim(len(idx), hid, drop)
            encoders.append(enc)
            self.branch_idx.append(idx)
            out_dims.append(d_out)
        self.branch_encoders = nn.ModuleList(encoders)

        # merge proj
        self.d_merge = d_merge if d_merge is not None else int(max(out_dims))
        projs = []
        for d in out_dims:
            projs.append(nn.Identity() if d == self.d_merge else nn.Linear(d, self.d_merge))
        self.branch_projs = nn.ModuleList(projs)

        # gate (global, input-independent)
        self.alpha = nn.Parameter(torch.zeros(len(branches)))

        # head
        self.head = nn.Sequential(
            nn.Linear(self.d_merge, head_hidden), nn.ReLU(), nn.Dropout(head_drop),
            nn.Linear(head_hidden, n_out),
        )

        # direct map
        self.direct_in_idx  = list(direct_in_idx or [])
        self.direct_out_idx = list(direct_out_idx or [])
        if len(self.direct_in_idx) != len(self.direct_out_idx):
            raise ValueError("direct_in_idx와 direct_out_idx 길이가 같아야 합니다.")

        if len(self.direct_in_idx) > 0:
            M = len(self.direct_in_idx)
            self.w_direct = nn.Parameter(torch.zeros(M, dtype=torch.float32), requires_grad=not self.freeze_direct)
            self.b_direct = nn.Parameter(torch.zeros(M, dtype=torch.float32), requires_grad=not self.freeze_direct)
        else:
            self.w_direct = None
            self.b_direct = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        dev = next(self.parameters()).device
        if X.device != dev:
            X = X.to(dev, non_blocking=True)

        # branches → proj → stack
        zs: List[torch.Tensor] = []
        for enc, proj, idx in zip(self.branch_encoders, self.branch_projs, self.branch_idx):
            zi = proj(enc(X[:, idx]))                  # (B, d_merge)
            zs.append(zi)
        w = torch.softmax(self.alpha, dim=0)           # (R,)
        h = torch.stack(zs, dim=0)                     # (R, B, d_merge)
        h = torch.einsum("r, rbd -> bd", w, h)         # (B, d_merge)

        y_head = self.head(h)                          # (B, n_out)

        # direct residual
        if self.w_direct is not None and len(self.direct_in_idx) > 0:
            Xsel = X[:, self.direct_in_idx]            # (B, M)
            z = Xsel * self.w_direct + self.b_direct   # (B, M)
            y_direct = X.new_zeros(X.shape[0], self.n_out)
            out_idx = torch.tensor(self.direct_out_idx, device=X.device, dtype=torch.long)
            y_direct.index_copy_(1, out_idx, z)
            return y_head + y_direct

        return y_head


# Skorch 래퍼: 그룹별/헤드/직결 학습률
class SafeNetNBranchLR(NeuralNetRegressor):
    def __init__(self, *args, lr_br: Optional[List[float]] = None,
                 lr_head: Optional[float] = None,
                 lr_direct: Optional[float] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_br     = lr_br
        self.lr_head   = lr_head
        self.lr_direct = lr_direct

    def fit(self, X, y=None, **kw):  # pandas→float32
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
        # branches
        n_br = len(mdl.branch_encoders)
        lrs = self.lr_br if (self.lr_br and len(self.lr_br) > 0) else [base_lr] * n_br
        if len(lrs) == 1 and n_br > 1:
            lrs = lrs * n_br
        for i in range(n_br):
            params_i = list(mdl.branch_encoders[i].parameters()) + list(mdl.branch_projs[i].parameters())
            if params_i:
                groups.append({"params": params_i, "lr": float(lrs[i])})

        # head + gate(alpha)
        head_params = list(mdl.head.parameters()) + [mdl.alpha]
        groups.append({"params": head_params, "lr": float(self.lr_head or base_lr)})

        # direct (동결 아닐 때만)
        if (mdl.w_direct is not None) and (not mdl.freeze_direct):
            groups.append({"params": [mdl.w_direct, mdl.b_direct],
                           "lr": float(self.lr_direct or base_lr)})

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
        for k in ["history", "_dataset_train", "_dataset_valid", "_optimizer", "_criterion", "_callbacks"]:
            state.pop(k, None)
        return state

    def infer(self, x, **fit_params):
        x = to_device(x, self.device)
        return super().infer(x, **fit_params)


# 전략
class TorchMultiBranchStrategy(BaseStrategy):
    _DEFAULT_GRID = {
        "dnn__optimizer__lr": [1e-3],
        "dnn__optimizer__weight_decay": [1e-4],
        "dnn__lr_br": [[1e-3]],
        "dnn__lr_head": [1e-3],
        "dnn__lr_direct": [5e-4],
        "dnn__module__head_hidden": [16],
    }

    def __init__(self, core, option: Dict, params_grid=None):
        super().__init__(core, option, params_grid)
        cols_X = list(core.X.columns)
        cols_y = list(core.y.columns) if core.y.ndim > 1 else [core.y.name]
        self.n_out = len(cols_y)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # branches
        branches_cfg = option.get("branches", [])
        assert branches_cfg, "option['branches']는 최소 1개 필요합니다."
        self.branches: List[Dict] = []
        for br in branches_cfg:
            c = br["cols"]
            idx = [cols_X.index(col) for col in c]
            self.branches.append({
                "idx": idx,
                "hidden": tuple(br.get("hidden", (32, 16))),
                "drop": float(br.get("drop", 0.1)),
            })

        # direct map
        direct_pairs = option.get("direct_map", []) or []
        self.direct_in_idx: List[int] = []
        self.direct_out_idx: List[int] = []
        for in_name, out_name in direct_pairs:
            if in_name not in cols_X:
                raise ValueError(f"direct_map 입력 '{in_name}' 이 X 컬럼에 없습니다.")
            if out_name not in cols_y:
                raise ValueError(f"direct_map 타깃 '{out_name}' 이 y 컬럼에 없습니다.")
            self.direct_in_idx.append(cols_X.index(in_name))
            self.direct_out_idx.append(cols_y.index(out_name))

        # etc
        self.freeze_direct = bool(option.get("freeze_direct", False))
        self.head_hidden   = int(option.get("head_hidden", 16))
        self.d_merge       = option.get("d_merge", None)

    def build_pipeline(self) -> Pipeline:
        net = SafeNetNBranchLR(
            module=MultiBranchNet,
            module__branches=self.branches,
            module__n_out=self.n_out,
            module__direct_in_idx=self.direct_in_idx,
            module__direct_out_idx=self.direct_out_idx,
            module__freeze_direct=self.freeze_direct,
            module__d_merge=self.d_merge,
            module__head_hidden=self.head_hidden,
            optimizer=torch.optim.Adam,
            optimizer__lr=1e-3,
            optimizer__weight_decay=1e-4,
            criterion=nn.MSELoss,
            batch_size=16,
            max_epochs=100,
            train_split=ValidSplit(0.2, stratified=False),
            callbacks=[
                PretrainDirectMap(),
                EarlyStopping(monitor="valid_loss", patience=10, lower_is_better=True),
            ],
            device=self.device,
            verbose=0,
        )

        # 학습 전 로그 확인 : Default False (Just for Checking, Can eliminate later)
        if self.option.get("log_direct", False):
            cols_X = list(self.core.X.columns)
            cols_y = list(self.core.y.columns) if self.core.y.ndim > 1 else [self.core.y.name]
            pairs = [(cols_X[i], cols_y[j]) for i, j in zip(self.direct_in_idx, self.direct_out_idx)]
            print("\n[DirectMap: before fit]")
            print("  in_idx :", self.direct_in_idx)
            print("  out_idx:", self.direct_out_idx)
            print("  pairs  :", pairs, "\n")
        #--------

        self.grid = getattr(self, "params_grid", None) or self._DEFAULT_GRID

        return Pipeline([
            ("sc",   StandardScaler()),
            ("to32", FunctionTransformer(
                lambda z: (
                    z.to_numpy(dtype=np.float32, copy=False)
                    if isinstance(z, (pd.DataFrame, pd.Series))
                    else z.astype(np.float32, copy=False)
                ),
                validate=False,
            )),
            ("dnn",  net),
        ])