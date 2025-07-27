"""
PyTorch-기반 전략 모음
────────────────────────────────────────────────────────────────────
• TorchMLP                    : 단순 feed-forward MLP
• TorchDNNStrategy            : 개별 MLP (GridSearch 대상)
• NetEnsemble                 : 여러 seed → valid_loss Top-k 평균
• TorchDNNEnsembleStrategy    : NetEnsemble 래퍼
• TorchDNNDualStrategy        : 두 Feature-그룹을 별도 MLP → 결합
"""

from __future__ import annotations

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

from .strategies import BaseStrategy          # 기존 파일의 BaseStrategy

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


#-------------Unified Dual (Static + Dual-branch) Strategy----------------
# ─────────────────────────────────────────────────────────────────────────────
# Unified Dual (Static + Dual-branch w/ per-branch LR)
# ─────────────────────────────────────────────────────────────────────────────
from typing import List, Tuple, Optional, Dict
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from skorch import NeuralNetRegressor
from skorch.dataset import ValidSplit
from skorch.callbacks import Callback, EarlyStopping
from sklearn.linear_model import LinearRegression

# (1) 작은 유틸: MLP 빌더
def _mlp(in_dim: int, hidden: Tuple[int, ...], drop: float) -> tuple[nn.Sequential, int]:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
        d = h
    return nn.Sequential(*layers), d

# (2) 통합 모듈: Static + Dual
class UnifiedDualMLP(nn.Module):
    """
    통합 듀얼 브랜치 모듈
      • static_slope_idx가 있으면: 브랜치1=고정 선형(buffer w_s, b_s)
      • 없으면: 브랜치1=MLP(hidden1, drop1)
      • merge ∈ {'concat','sum','gate'}
    """
    def __init__(self,
                 idx2: List[int],
                 out_dim: int = 1,
                 merge: str = "concat",
                 # learned-branch1
                 idx1: Optional[List[int]] = None,
                 hidden1: Tuple[int, ...] = (3,),
                 drop1: float = 0.0,
                 # static-branch1
                 static_slope_idx: Optional[int] = None,
                 # branch2
                 hidden2: Tuple[int, ...] = (64,),
                 drop2: float = 0.0):
        super().__init__()
        assert merge in {"concat", "sum", "gate"}
        self.merge = merge
        self.idx2  = idx2
        self.static_slope_idx = static_slope_idx

        # branch1
        if static_slope_idx is None:
            assert idx1 is not None and len(idx1) > 0, "idx1 must be provided for dual-branch mode"
            self.idx1 = idx1
            self.br1, d1 = _mlp(len(idx1), hidden1, drop1)
            self.is_static = False
        else:
            self.idx1 = None
            self.register_buffer('w_s', torch.tensor(0., dtype=torch.float32))
            self.register_buffer('b_s', torch.tensor(0., dtype=torch.float32))
            d1 = 1
            self.is_static = True

        # branch2
        self.br2, d2 = _mlp(len(idx2), hidden2, drop2)

        # merge
        if merge == "concat":
            head_in = d1 + d2
            self.proj1 = nn.Identity()
            self.proj2 = nn.Identity()
            self.alpha = None
        else:
            d_merge = max(d1, d2)
            self.proj1 = nn.Identity() if d1 == d_merge else nn.Linear(d1, d_merge)
            self.proj2 = nn.Identity() if d2 == d_merge else nn.Linear(d2, d_merge)
            self.alpha = nn.Parameter(torch.tensor(0.0)) if merge == "gate" else None
            head_in = d_merge

        self.head = nn.Sequential(nn.Linear(head_in, 16), nn.ReLU(),
                                  nn.Linear(16, out_dim))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # branch1
        if self.is_static:
            x_s = X[:, self.static_slope_idx]      # (B,)
            z1  = (x_s * self.w_s + self.b_s).unsqueeze(1)  # (B,1)
        else:
            z1 = self.br1(X[:, self.idx1])         # (B,d1)

        # branch2
        z2 = self.br2(X[:, self.idx2])             # (B,d2)

        # merge
        if self.merge == "concat":
            h = torch.cat([z1, z2], dim=1)
        elif self.merge == "sum":
            h = self.proj1(z1) + self.proj2(z2)
        else:  # gate
            a = torch.sigmoid(self.alpha)
            h = (1 - a) * self.proj1(z1) + a * self.proj2(z2)

        return self.head(h)

# (3) Static 사전회귀 콜백
class PretrainSlope(Callback):
    """slope 단일 회귀로 w_s, b_s 초기화 + freeze (static 모드에서만 사용)"""
    def __init__(self, slope_col_idx: int): self.slope_col_idx = slope_col_idx
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        if not hasattr(net, "module_") or net.module_ is None:
            net.initialize()
        to_np = lambda a: (a.to_numpy(dtype=np.float32, copy=False)
                           if hasattr(a, "to_numpy")
                           else np.asarray(a, dtype=np.float32))
        Xnp, ynp = to_np(X), to_np(y)
        lr = LinearRegression().fit(Xnp[:, [self.slope_col_idx]], ynp)
        w_s = float(lr.coef_.ravel()[0])
        b_s = float(np.mean(lr.intercept_)) if np.ndim(lr.intercept_) else float(lr.intercept_)
        mdl = net.module_
        mdl.w_s.data.fill_(w_s); mdl.b_s.data.fill_(b_s)
        for p in [mdl.w_s, mdl.b_s]: p.requires_grad = False

# (4) SafeNet + 브랜치별 learning rate
class SafeNetDualLR(NeuralNetRegressor):
    """
    pandas→float32 변환 + 브랜치별 learning rate 지원
    GridSearch에서 'dnn__lr_br1', 'dnn__lr_br2', 'dnn__lr_head' 로 튜닝 가능
    """
    def __init__(self, *args, lr_br1=None, lr_br2=None, lr_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_br1 = lr_br1
        self.lr_br2 = lr_br2
        self.lr_head = lr_head

    def fit(self, X, y=None, **kw):                         # type: ignore[override]
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy(dtype=np.float32, copy=False)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype=np.float32, copy=False)
        return super().fit(X, y, **kw)

    def initialize_optimizer(self):
        # 모듈 초기화 먼저
        super().initialize_module()
        mdl = self.module_

        # base kwargs (예: weight_decay 등)
        opt_cls = self.optimizer
        opt_kw = self._get_params_for_optimizer(opt_cls)

        # 학습 파라미터 그룹 구성
        groups = []
        if hasattr(mdl, "br1") and any(p.requires_grad for p in mdl.br1.parameters()):
            groups.append({"params": mdl.br1.parameters(), "lr": self.lr_br1 or opt_kw.pop("lr", 1e-3)})
        if hasattr(mdl, "br2"):
            groups.append({"params": mdl.br2.parameters(), "lr": self.lr_br2 or opt_kw.get("lr", 1e-3)})
        groups.append({"params": mdl.head.parameters(), "lr": self.lr_head or opt_kw.get("lr", 1e-3)})

        # optimizer 생성
        self.optimizer_ = opt_cls(groups, **opt_kw)
        return self

    def __getstate__(self):
        state = super().__getstate__()
        for k in ["history","_dataset_train","_dataset_valid","_optimizer","_criterion","_callbacks"]:
            state.pop(k, None)
        return state

# (5) 통합 전략
from .strategies import BaseStrategy  # 프로젝트 내 BaseStrategy

class TorchDualUnifiedStrategy(BaseStrategy):
    """
    option:
      • Static: {"slope":"slope_col", "grp2":[...], "merge":"concat|sum|gate"}
      • Dual  : {"grp1":[...],       "grp2":[...], "merge":"concat|sum|gate"}
    Grid 예시:
      {
        "dnn__module__merge": ["concat","sum","gate"],
        "dnn__module__hidden2": [(32,16),(64,32)],
        "dnn__module__drop2": [0.0, 0.2],
        "dnn__lr_br1": [1e-3],      # dual 모드일 때만 유효
        "dnn__lr_br2": [5e-4,1e-3],
        "dnn__lr_head": [1e-3],
        "dnn__optimizer__weight_decay": [0.0, 5e-4],
      }
    """
    _DEFAULT_GRID = {
        "dnn__module__merge": ["concat","sum","gate"],
        "dnn__module__hidden2": [(32,16),(64,32)],
        "dnn__module__drop2": [0.0, 0.2],
        "dnn__lr_br2": [1e-3],
        "dnn__lr_head": [1e-3],
        "dnn__optimizer__weight_decay": [0.0, 5e-4],
    }

    def __init__(self, core, option: Dict, params_grid=None):
        super().__init__(core, option, params_grid)
        cols = list(core.X.columns)

        self.is_static = "slope" in option and option.get("slope") is not None
        self.merge = option.get("merge", "concat")

        if self.is_static:
            self.static_slope_idx = cols.index(option["slope"])
            self.idx1 = None
        else:
            grp1 = option["grp1"]; assert grp1 and isinstance(grp1, (list, tuple))
            self.idx1 = [cols.index(c) for c in grp1]
            self.static_slope_idx = None

        grp2 = option["grp2"]; assert grp2 and isinstance(grp2, (list, tuple))
        self.idx2 = [cols.index(c) for c in grp2]

        self.out_dim = 1 if self.core.y.ndim == 1 else self.core.y.shape[1]
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"

    def build_pipeline(self) -> Pipeline:
        net = SafeNetDualLR(
            module                   = UnifiedDualMLP,
            # 모듈 args
            module__idx2             = self.idx2,
            module__out_dim          = self.out_dim,
            module__merge            = self.merge,
            module__idx1             = self.idx1,
            module__static_slope_idx = self.static_slope_idx,
            module__hidden1          = (3,),       # 필요시 grid로
            module__drop1            = 0.0,
            module__hidden2          = (32,16),
            module__drop2            = 0.1,
            # optimizer & train
            optimizer                = torch.optim.Adam,
            optimizer__lr            = 1e-3,       # 기본 lr (그룹 lr이 없으면 fallback)
            optimizer__weight_decay  = 0.0,
            criterion                = nn.MSELoss,
            batch_size               = 16,
            max_epochs               = 100,
            train_split              = ValidSplit(0.2, stratified=False),
            callbacks                = (
                [PretrainSlope(self.static_slope_idx),
                 EarlyStopping(monitor="valid_loss", patience=10, lower_is_better=True)]
                if self.is_static else
                [EarlyStopping(monitor="valid_loss", patience=10, lower_is_better=True)]
            ),
            device                   = self.device,
            verbose                  = 0,
        )

        self.grid = getattr(self, "params_grid", None) or self._DEFAULT_GRID

        return Pipeline([
            ("to32", FunctionTransformer(
                lambda z: (z.to_numpy(dtype=np.float32, copy=False)
                           if isinstance(z, (pd.DataFrame, pd.Series))
                           else z.astype(np.float32, copy=False)),
                validate=False
            )),
            ("sc", StandardScaler()),
            ("dnn", net),
        ])