"""
PyTorch-기반 전략 (ARM: Additive Residual Model)
────────────────────────────────────────────────────────────────────
• TorchMLP              : 단순 feed-forward MLP (residual learner로 사용)
• MultiBranchNet        : N-branch encoder + softmax gate + head (residual learner)
• AdditiveResidualModel : Base(선형) + Residual(MLP/Multi-branch)
• ARMStrategy           : 외부 인터페이스(옵션으로 Base on/off, residual 종류 선택)
"""

from __future__ import annotations
from .strategies import BaseStrategy          # 외부 프레임워크의 베이스 전략

# ─────────────────────────  공통 IMPORT  ─────────────────────────
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from skorch.utils import to_device

# ─────────── ① MLP (Residual learner) ───────────
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
    """pandas/Series → float32 NumPy 자동 변환 래퍼."""
    def fit(self, X, y=None, **kw):                         # type: ignore[override]
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy(dtype=np.float32, copy=False)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype=np.float32, copy=False)
        return super().fit(X, y, **kw)

    def __getstate__(self):
        state = super().__getstate__()      # skorch 기본 state
        # 학습 끝난 뒤 무거운 객체 제거(직렬화 안정성)
        for k in ["history","_dataset_train","_dataset_valid","_optimizer","_criterion","_callbacks"]:
            state.pop(k, None)
        return state


# ─────────── ③ Multi-branch residual net ───────────
def _mlp_with_dim(in_dim: int, hidden: Tuple[int, ...], drop: float) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
        d = h
    return nn.Sequential(*layers), (d if hidden else in_dim)

class MultiBranchNet(nn.Module):
    """
    N-브랜치 인코더 + softmax gate 병합 + head
    - branches        : List[dict]  ── 각 dict = {"idx": [...], "hidden": (...), "drop": float}
    - d_merge         : Optional[int] 병합 차원(없으면 브랜치 출력 차원 max, 0-branch면 n_out)
    - head_hidden     : int         ── head 은닉 크기
    """
    def __init__(
        self,
        branches: List[Dict],
        n_out: int,
        d_merge: Optional[int] = None,
        head_hidden: int = 16,
        head_drop: float = 0.0,
    ):
        super().__init__()
        self.n_out = n_out

        # 1) 브랜치 인코더
        self.branch_idx:   List[List[int]] = []
        self.branch_encoders = nn.ModuleList()
        self.branch_projs    = nn.ModuleList()
        out_dims: List[int]  = []

        for br in branches:       # 0개도 허용
            idx   = br["idx"]
            hid   = tuple(br.get("hidden", ()))
            drop  = float(br.get("drop", 0.0))
            enc, d_out = _mlp_with_dim(len(idx), hid, drop)
            self.branch_idx.append(idx)
            self.branch_encoders.append(enc)
            out_dims.append(d_out)

        # 2) 병합 차원 & projection
        if len(out_dims) == 0:       # 0-branch 특수처리
            self.d_merge = int(d_merge or n_out)
            self.alpha   = None
        else:
            self.d_merge = int(d_merge or max(out_dims))
            for d in out_dims:
                self.branch_projs.append(nn.Identity() if d == self.d_merge
                                          else nn.Linear(d, self.d_merge))
            self.alpha = nn.Parameter(torch.zeros(len(branches)))  # softmax gate

        # 3) head
        self.head = nn.Sequential(
            nn.Linear(self.d_merge, head_hidden), nn.ReLU(), nn.Dropout(head_drop),
            nn.Linear(head_hidden, n_out),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        dev = next(self.parameters()).device
        if X.device != dev:
            X = X.to(dev, non_blocking=True)

        # (A) 브랜치 없으면 0-tensor, 있으면 gate 합
        if len(self.branch_encoders) == 0:
            h = X.new_zeros(X.size(0), self.d_merge)
        else:
            zs = []
            for enc, proj, idx in zip(self.branch_encoders, self.branch_projs, self.branch_idx):
                zi = proj(enc(X[:, idx]))
                zs.append(zi)                                 # (B, d_merge)
            w = torch.softmax(self.alpha, dim=0)             # (R,)
            h = torch.stack(zs, 0)                           # (R,B,d_merge)
            h = torch.einsum("r, rbd -> bd", w, h)           # gate 합

        return self.head(h)                                  # (B, n_out)


class SafeNetNBranchLR(NeuralNetRegressor):
    """MultiBranchNet용: 브랜치/헤드별 학습률 그룹화."""
    def __init__(self, *args, lr_br: Optional[List[float]] = None,
                 lr_head: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_br   = lr_br
        self.lr_head = lr_head

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
            params_i = list(mdl.branch_encoders[i].parameters())
            params_i += list(mdl.branch_projs[i].parameters())
            if params_i:
                groups.append({"params": params_i, "lr": float(lrs[i])})

        # head + gate(alpha)
        head_params = list(mdl.head.parameters())
        if getattr(mdl, "alpha", None) is not None:
            head_params += [mdl.alpha]
        groups.append({"params": head_params, "lr": float(self.lr_head or base_lr)})

        # skorch 표준 초기화
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


# ─────────── ④ ARM: Base + Residual ───────────
class AdditiveResidualModel(BaseEstimator, RegressorMixin):
    """
    y_hat = [base_on ? f_base(X[base_cols]) : 0] + f_residual(X[feature_cols])
    - base_on=True  → residual은 잔차 U=Y - Y_base 를 학습 (CS-Resi/Hybrid)
    - base_on=False → residual이 Y 자체를 학습 (일반 DNN)
    - residual_model: 멀티타깃 회귀기(SafeNet 또는 SafeNetNBranchLR)
    - feature_cols: residual 입력 컬럼의 '순서'(미지정 시 X의 수치형 전체)
    강건성:
      * 필수 컬럼 검사
      * NaN 행 동기 마스킹(훈련), 추론 시 NaN 대치
      * 잔차 입력만 표준화(StandardScaler)
    """
    def __init__(self,
                 base_on: bool = True,
                 base_cols: Optional[List[str]] = None,
                 target_cols: Optional[List[str]] = None,
                 residual_model: Optional[Any] = None,
                 base_model: Optional[Any] = None,
                 feature_cols: Optional[List[str]] = None,
                 standardize_res: bool = True,
                 seed: int = 0):
        self.base_on = bool(base_on)
        self.base_cols = list(base_cols or [])
        self.target_cols = list(target_cols or [])
        self.residual_model = residual_model
        self.base_model = base_model or LinearRegression()
        self.feature_cols = list(feature_cols or [])
        self.standardize_res = bool(standardize_res)
        self.seed = int(seed)

        # state
        self._scaler: Optional[StandardScaler] = None
        self._feat_cols_used: List[str] = []
        self._target_dim: int = 0
        self._nan_info_: Dict[str, int] = {}

    # utils
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

    def _check_required_cols(self, X: pd.DataFrame, y: pd.DataFrame):
        miss_t = [c for c in self.target_cols if c not in y.columns]
        if miss_t:
            raise ValueError(f"[ARM] target_cols missing: {miss_t}")
        if self.base_on:
            miss_b = [c for c in self.base_cols if c not in X.columns]
            if miss_b:
                raise ValueError(f"[ARM] base_cols missing: {miss_b}")

    def _pick_features(self, X: pd.DataFrame) -> List[str]:
        if self.feature_cols:
            feats = self._numeric_cols(X, self.feature_cols)
        else:
            feats = self._numeric_cols(X)
        return feats

    def _nan_mask_joint(self, arrays: List[np.ndarray]) -> np.ndarray:
        mask = np.ones(arrays[0].shape[0], dtype=bool)
        for arr in arrays:
            bad = ~np.isfinite(arr).all(axis=1)
            mask &= ~bad
        return mask

    # fit
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "X must be DataFrame"
        if isinstance(y, pd.Series): y = y.to_frame()
        self._check_required_cols(X, y)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        Y_df = y[self.target_cols].copy()
        Y = self._to_2d(Y_df)
        self._target_dim = Y.shape[1]

        # base features
        if self.base_on:
            Xb_cols = self._numeric_cols(X, self.base_cols)
            Xb = self._to_2d(X[Xb_cols])
        else:
            Xb_cols = []
            Xb = np.zeros((len(X), 1), dtype=np.float32)  # 마스크 동기화용 dummy

        # residual features (ordered)
        self._feat_cols_used = self._pick_features(X)
        Xres = self._to_2d(X[self._feat_cols_used]) if self._feat_cols_used else np.zeros((len(X), 1), np.float32)

        # NaN joint mask
        m = self._nan_mask_joint([Xb, Y, Xres])
        n_drop = int((~m).sum())
        self._nan_info_ = {"dropped_rows": n_drop}
        if n_drop > 0:
            Xb, Y, Xres = Xb[m], Y[m], Xres[m]

        # base fit & residual target
        if self.base_on:
            self.base_model.fit(Xb, Y)
            Y_base = self.base_model.predict(Xb).astype(np.float32, copy=False)
            U = Y - Y_base
        else:
            U = Y

        # scale residual features
        if self.standardize_res and Xres.shape[1] > 0:
            self._scaler = StandardScaler(with_mean=True, with_std=True)
            Z = self._scaler.fit_transform(Xres)
        else:
            self._scaler = None
            Z = Xres

        # residual fit
        if (self.residual_model is None) or (Z.shape[1] == 0):
            self.residual_model = None
        else:
            self.residual_model.fit(Z, U)

        return self

    # predict
    def _predict_base(self, X: pd.DataFrame) -> np.ndarray:
        if not self.base_on:
            return np.zeros((len(X), self._target_dim), dtype=np.float32)
        Xb = self._to_2d(X[self._numeric_cols(X, self.base_cols)])
        if not np.isfinite(Xb).all():
            Xb = np.nan_to_num(Xb, copy=False)
        return self.base_model.predict(Xb).astype(np.float32, copy=False)

    def _predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        if (self.residual_model is None) or (len(self._feat_cols_used) == 0):
            return np.zeros((len(X), self._target_dim), dtype=np.float32)
        Xr = self._to_2d(X[self._feat_cols_used])
        if not np.isfinite(Xr).all():
            Xr = np.nan_to_num(Xr, copy=False)
        Z = self._scaler.transform(Xr) if self._scaler is not None else Xr
        P = self.residual_model.predict(Z).astype(np.float32, copy=False)
        if P.ndim == 1: P = P.reshape(-1, 1)
        return P

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_base(X) + self._predict_residual(X)

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        Ytrue = (y if isinstance(y, pd.DataFrame) else y.to_frame())[self.target_cols].to_numpy(dtype=np.float32, copy=False)
        Yhat = self.predict(X)
        return -float(np.mean((Ytrue - Yhat) ** 2))

#------------Vanilla DNN Strategy ----------------
class TorchDNNStrategy(BaseStrategy):
    """
    가장 단순한 DNN:
      X -> (scale) -> TorchMLP(SafeNet) -> y
    - 옵션:
        scaler: 'standard' | 'minmax' | None (default='standard')
        hidden: tuple (예: (64,32))
        dropout: float
        lr, wd, bs, epochs, patience
    """
    _DEFAULT_GRID = {
        "dnn__module__hidden":          [(32, 16), (64, 32)],
        "dnn__module__dropout":         [0.0, 0.2],
        "dnn__optimizer__lr":           [1e-3, 5e-4],
        "dnn__optimizer__weight_decay": [0.0, 5e-4],
    }

    @staticmethod
    def _to32():
        # pandas/Series -> float32 ndarray (sklearn FunctionTransformer용)
        return FunctionTransformer(
            lambda z: (
                z.to_numpy(dtype=np.float32, copy=False)
                if isinstance(z, (pd.DataFrame, pd.Series))
                else z.astype(np.float32, copy=False)
            ),
            validate=False,
        )

    def build_pipeline(self) -> Pipeline:
        opt = self.option or {}
        n_out = 1 if self.core.y.ndim == 1 else self.core.y.shape[1]

        # --- 스케일러 선택 ---
        scaler_choice = str(opt.get("scaler", "standard")).lower()
        if scaler_choice == "standard":
            scaler = StandardScaler()
        elif scaler_choice == "minmax":
            rng = opt.get("minmax_range", (-1, 1))
            scaler = MinMaxScaler(rng)
        else:
            scaler = None  # 스케일러 미사용

        # --- DNN 본체 ---
        net = SafeNet(
            module=TorchMLP,
            module__num_feat=self.core.X.shape[1],   # 고정 크기 사용(명시적)
            module__num_out=n_out,
            module__hidden=tuple(opt.get("hidden", (32, 16))),
            module__dropout=float(opt.get("dropout", 0.1)),
            optimizer=torch.optim.Adam,
            optimizer__lr=float(opt.get("lr", 1e-3)),
            optimizer__weight_decay=float(opt.get("wd", 0.0)),
            criterion=nn.MSELoss,
            batch_size=int(opt.get("bs", 32)),
            max_epochs=int(opt.get("epochs", 100)),
            train_split=ValidSplit(0.2, stratified=False),
            callbacks=[EarlyStopping(monitor="valid_loss",
                                     patience=int(opt.get("patience", 10)),
                                     threshold=1e-5,
                                     lower_is_better=True)],
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0,
        )

        steps = [("to32", self._to32())]
        if scaler is not None:
            steps.append(("sc", scaler))
        steps.append(("dnn", net))

        self.grid = getattr(self, "params_grid", None) or self._DEFAULT_GRID
        return Pipeline(steps)


# ─────────── ⑤ ARM Strategy (외부 진입점) ───────────
class ARMStrategy(BaseStrategy):
    """
    Additive Residual Model Strategy
      - base_on True/False (선형 베이스 on/off)
      - residual_kind: 'mlp' 또는 'multibranch'
      - feature_cols: residual 입력 컬럼 순서(선택, 미지정 시 X의 수치형 전체)
      - target_cols, base_cols 필수(베이스 off라면 base_cols 생략 가능)
    """
    _DEFAULT_GRID = {
        # MLP residual 예시 그리드(접두: arm__residual_model__...)
        "arm__residual_model__module__hidden": [(32, 16), (64, 32)],
        "arm__residual_model__module__dropout": [0.0, 0.2],
        "arm__residual_model__optimizer__lr": [1e-3, 5e-4],
        "arm__residual_model__optimizer__weight_decay": [0.0, 5e-4],
    }

    def build_pipeline(self) -> Pipeline:
        opt: Dict[str, Any] = self.option or {}

        base_on: bool = bool(opt.get("base_on", True))
        base_cols: List[str] = list(opt.get("base_cols", []))
        target_cols: List[str] = list(opt.get("target_cols", []))
        feature_cols: List[str] = list(opt.get("feature_cols", []))
        base_model: Any = opt.get("base_model", LinearRegression())
        seed: int = int(opt.get("seed", 0))

        # 멀티타깃 차원
        n_out = 1 if self.core.y.ndim == 1 else self.core.y.shape[1]

        # residual model 선택
        resid_kind = str(opt.get("residual_kind", "mlp")).lower()

        if resid_kind == "mlp":
            residual_model = SafeNet(
                module=TorchMLP,
                module__num_feat=None,              # 첫 배치에서 자동 유추
                module__num_out=n_out,
                module__hidden=tuple(opt.get("mlp_hidden", (32, 16))),
                module__dropout=float(opt.get("mlp_dropout", 0.1)),
                optimizer=torch.optim.Adam,
                optimizer__lr=float(opt.get("mlp_lr", 1e-3)),
                optimizer__weight_decay=float(opt.get("mlp_wd", 0.0)),
                criterion=nn.MSELoss,
                batch_size=int(opt.get("mlp_bs", 32)),
                max_epochs=int(opt.get("mlp_epochs", 100)),
                train_split=ValidSplit(0.2, stratified=False),
                callbacks=[EarlyStopping(monitor="valid_loss",
                                         patience=int(opt.get("mlp_patience", 10)),
                                         threshold=1e-5,
                                         lower_is_better=True)],
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=0,
            )

        elif resid_kind == "multibranch":
            # feature_cols 기준으로 branch 인덱스 매핑
            if feature_cols:
                feat_cols = feature_cols
            else:
                feat_cols = [c for c in self.core.X.columns if np.issubdtype(self.core.X[c].dtype, np.number)]
            name2idx = {c: i for i, c in enumerate(feat_cols)}

            branches_cfg: List[Dict[str, Any]] = opt.get("branches", [])
            assert branches_cfg, "residual_kind='multibranch' requires option['branches']"

            branches_idx = []
            for br in branches_cfg:
                idx = [name2idx[c] for c in br["cols"] if c in name2idx]
                branches_idx.append({
                    "idx": idx,
                    "hidden": tuple(br.get("hidden", (64, 32))),
                    "drop": float(br.get("drop", 0.1)),
                })

            residual_model = SafeNetNBranchLR(
                module=MultiBranchNet,
                module__branches=branches_idx,
                module__n_out=n_out,
                module__d_merge=opt.get("d_merge", None),
                module__head_hidden=int(opt.get("head_hidden", 16)),
                optimizer=torch.optim.Adam,
                optimizer__lr=float(opt.get("mb_lr", 1e-3)),
                optimizer__weight_decay=float(opt.get("mb_wd", 1e-4)),
                criterion=nn.MSELoss,
                batch_size=int(opt.get("mb_bs", 32)),
                max_epochs=int(opt.get("mb_epochs", 100)),
                train_split=ValidSplit(0.2, stratified=False),
                callbacks=[EarlyStopping(monitor="valid_loss",
                                         patience=int(opt.get("mb_patience", 10)),
                                         lower_is_better=True)],
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=0,
            )
        else:
            raise ValueError("residual_kind must be 'mlp' or 'multibranch'")

        arm = AdditiveResidualModel(
            base_on=base_on,
            base_cols=base_cols,
            target_cols=target_cols,
            residual_model=residual_model,
            base_model=base_model,
            feature_cols=(feature_cols if feature_cols else None),
            standardize_res=bool(opt.get("standardize_res", True)),
            seed=seed,
        )

        # Grid 접두: arm__residual_model__...
        self.grid = getattr(self, "params_grid", None) or self._DEFAULT_GRID
        return Pipeline([("arm", arm)])