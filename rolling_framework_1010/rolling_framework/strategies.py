# rolling_framework/strategies.py
from __future__ import annotations
from typing import Dict, List, Optional, Iterable, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# xgboost는 선택적 의존성
try:
    from xgboost import XGBRegressor   # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# ======================================================================
# Base API
# ======================================================================
class Strategy:
    """Minimal interface: fit on (X_tr, y_tr); predict for x_te; return Series."""
    def __init__(self, target_cols: List[str]):
        self.target_cols = list(target_cols)

    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        raise NotImplementedError


# ======================================================================
# Helpers
# ======================================================================
def _build_weights(
    target_cols: List[str],
    weights: Optional[Union[Dict[str, float], List[float], np.ndarray]]
) -> Optional[np.ndarray]:
    """
    target_cols 순서에 맞는 가중치 벡터 생성:
    - dict: {target_name: weight}
    - list/ndarray: 타깃 수와 길이 일치 필요
    - None: 가중치 미사용
    """
    if weights is None:
        return None
    if isinstance(weights, dict):
        w = np.array([float(weights.get(t, 1.0)) for t in target_cols], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(target_cols):
            raise ValueError("target_weights length must match number of target_cols")
    # 비정상 값 방지
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    return w


# ======================================================================
# Cross-Validation Utilities (hold-out / time-series CV)
# ======================================================================
def _as_prefixed_grid(grid: Dict[str, Iterable], prefix: str = "model__") -> Dict[str, Iterable]:
    """
    Ensure parameter keys are correctly namespaced for Pipeline step 'model'.
    Accepts either raw estimator keys ('alpha') or pipeline keys ('model__alpha').
    """
    out: Dict[str, Iterable] = {}
    for k, v in grid.items():
        out[k if k.startswith(prefix) else (prefix + k)] = v
    return out


def _holdout_split(n: int, val_ratio: float = 0.2) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Single contiguous hold-out split: first (n - v) train, last v validation.
    Returns a list with one (train_idx, val_idx) tuple.
    """
    v = max(int(round(n * val_ratio)), 1)
    t = max(n - v, 1)
    train_idx = np.arange(0, t, dtype=int)
    val_idx = np.arange(t, n, dtype=int)
    return [(train_idx, val_idx)]


def _tscv_splits(n: int, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    간단한 expanding-origin time-series CV (min_train/test_size 없이).
    각 폴드는 동일 길이의 검증구간을 갖고, 학습구간은 시작에서 폴드 끝까지 확장.
    """
    n_splits = int(max(1, n_splits))
    fold = max(1, n // (n_splits + 1))  # 대략 균등 검증 길이
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_splits + 1):
        train_end = fold * k
        val_start = train_end
        val_end = min(n, val_start + fold)
        if val_end - val_start <= 0 or train_end <= 0:
            continue
        splits.append((np.arange(0, train_end, dtype=int), np.arange(val_start, val_end, dtype=int)))
    return splits if splits else _holdout_split(n, 0.2)


def _cv_score_mse(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    멀티아웃풋 MSE 스코어.
    - weights=None: 전체(샘플×타깃) 평균 MSE
    - weights!=None: 타깃별 MSE(샘플 평균)를 가중평균(가중치 합으로 정규화)
    """
    err = (y_true - y_pred) ** 2
    if weights is None:
        return float(np.mean(err))
    per_target = err.mean(axis=0)  # shape: (n_targets,)
    w = weights[: per_target.shape[0]].astype(float)
    s = float(w.sum())
    return float(np.dot(per_target, w) / s) if s > 0 else float(per_target.mean())


def _run_cv_make_best_pipeline(
    make_pipeline_fn,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv_cfg: Dict,
    weights: Optional[np.ndarray] = None,
) -> Pipeline:
    """
    Generic grid-search over a pipeline factory.
    - make_pipeline_fn(params_dict) -> Pipeline
    - cv_cfg keys:
        mode: "holdout" | "tscv"
        grid: Dict[str, List]
        val_ratio: float        (holdout only, default 0.2)
        n_splits: int           (tscv only, default 5)
    Returns: best-fit Pipeline retrained on the FULL training data with best params.
    """
    # 데이터가 너무 적으면 CV 생략
    n = len(X)
    if n < 5 or not cv_cfg or "grid" not in cv_cfg or not cv_cfg["grid"]:
        pipe = make_pipeline_fn({})
        pipe.fit(X, y)
        return pipe

    mode = str(cv_cfg.get("mode", "holdout")).lower()
    grid = _as_prefixed_grid(cv_cfg.get("grid", {}), prefix="model__")

    if mode == "holdout":
        splits = _holdout_split(n, val_ratio=float(cv_cfg.get("val_ratio", 0.2)))
    elif mode in ("tscv", "time-series", "time_series"):
        splits = _tscv_splits(n, n_splits=int(cv_cfg.get("n_splits", 5)))
    else:
        raise ValueError("cv.mode must be 'holdout' or 'tscv'")

    best_score = np.inf
    best_params: Optional[Dict[str, object]] = None

    # Grid-search over splits
    for params in ParameterGrid(grid):
        fold_scores: List[float] = []
        for tr_idx, va_idx in splits:
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            pipe = make_pipeline_fn(params)
            pipe.fit(X_tr, y_tr)
            y_hat = pipe.predict(X_va)
            fold_scores.append(_cv_score_mse(y_va.to_numpy(float), np.asarray(y_hat, dtype=float), weights=weights))
        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    pipe = make_pipeline_fn(best_params or {})
    pipe.fit(X, y)
    return pipe


# ======================================================================
# Simple sklearn strategies (with optional CV + target weighting)
# ======================================================================
class SklearnRegStrategy(Strategy):
    """Generic sklearn regressor (with StandardScaler, MultiOutput if needed). Supports optional CV and target weighting."""
    def __init__(
        self,
        target_cols: List[str],
        base_estimator,
        scale: bool = True,
        cv: Optional[Dict] = None,
        target_weights: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
    ):
        super().__init__(target_cols)
        self.base_estimator = base_estimator
        self.scale = scale
        self.cv = cv
        self._weights = _build_weights(target_cols, target_weights)
        self._is_multi = len(target_cols) > 1
        self.pipe: Optional[Pipeline] = None

    def _make_pipeline(self, param_overrides: Dict) -> Pipeline:
        """
        Build a pipeline = [optional scaler] -> model, applying param overrides.
        param_overrides keys can be 'alpha' or 'model__alpha'; both are accepted.
        """
        est = clone(self.base_estimator).set_params(
            **{k.replace("model__", ""): v for k, v in param_overrides.items()}
        )
        if self._is_multi:
            est = MultiOutputRegressor(est, n_jobs=-1)
        steps = ([("scaler", StandardScaler())] if self.scale else []) + [("model", est)]
        return Pipeline(steps)

    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        # CV 스코어링 시 타깃 가중치 사용 (학습 자체는 각 모델 기본 손실)
        if self.cv is not None:
            self.pipe = _run_cv_make_best_pipeline(
                lambda params: self._make_pipeline(params),
                X_tr,
                y_tr[self.target_cols],
                self.cv,
                weights=self._weights,
            )
        else:
            self.pipe = self._make_pipeline({})
            self.pipe.fit(X_tr, y_tr[self.target_cols])

        yhat = self.pipe.predict(x_te.to_frame().T)[0]
        return pd.Series(yhat, index=self.target_cols)


# ======================================================================
# ARM strategy (two-stage: base + residual), with optional CV per stage + target weighting
# ======================================================================
class ARMStrategy(Strategy):
    """Two-stage ARM: y = base(X_base) + res(X_feat). Supports optional CV for each stage and weighted scoring."""
    def __init__(
        self,
        target_cols: List[str],
        base_cols: List[str],
        feature_cols: List[str],
        base_kind: str = "ols",
        base_params: Optional[Dict] = None,
        residual_kind: str = "mlp",
        residual_params: Optional[Dict] = None,
        scale_base: bool = True,
        scale_res: bool = True,
        base_cv: Optional[Dict] = None,
        res_cv: Optional[Dict] = None,
        target_weights: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
    ):
        super().__init__(target_cols)
        self.base_cols = list(base_cols)
        self.feature_cols = list(feature_cols)
        self._weights = _build_weights(target_cols, target_weights)

        # --- Base estimator
        bk = base_kind.lower()
        base_params = base_params or {}
        if bk == "ols":
            base_est = LinearRegression(**base_params)
        elif bk == "ridge":
            base_est = Ridge(**base_params)
        elif bk == "lasso":
            base_est = Lasso(**base_params)
        elif bk == "enet":
            base_est = ElasticNet(**base_params)
        elif bk == "mlp":
            base_est = MLPRegressor(**base_params)
        elif bk == "rf":
            base_est = RandomForestRegressor(**base_params)
        elif bk == "et":
            base_est = ExtraTreesRegressor(**base_params)
        elif bk == "xgb":
            if not _HAS_XGB:
                raise RuntimeError("xgboost not installed")
            base_est = XGBRegressor(**base_params)  # type: ignore
        else:
            raise ValueError("Unsupported base_kind")

        # --- Residual estimator
        rk = residual_kind.lower()
        residual_params = residual_params or {}
        if rk == "mlp":
            res_est = MLPRegressor(**residual_params)
        elif rk == "ridge":
            res_est = Ridge(**residual_params)
        elif rk == "lasso":
            res_est = Lasso(**residual_params)
        elif rk == "enet":
            res_est = ElasticNet(**residual_params)
        elif rk == "rf":
            res_est = RandomForestRegressor(**residual_params)
        elif rk == "et":
            res_est = ExtraTreesRegressor(**residual_params)
        elif rk == "xgb":
            if not _HAS_XGB:
                raise RuntimeError("xgboost not installed")
            res_est = XGBRegressor(**residual_params)  # type: ignore
        else:
            raise ValueError("Unsupported residual_kind")

        # Multi-output wrapping
        base_est = MultiOutputRegressor(base_est, n_jobs=-1) if len(target_cols) > 1 else base_est
        res_est  = MultiOutputRegressor(res_est,  n_jobs=-1) if len(target_cols) > 1 else res_est

        self.base_cv = base_cv
        self.res_cv  = res_cv

        self.base_pipe = Pipeline(
            [("scaler", StandardScaler())] * int(scale_base) + [("model", base_est)]
        )
        self.res_pipe = Pipeline(
            [("scaler", StandardScaler())] * int(scale_res) + [("model", res_est)]
        )

    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        # 1) Base stage (CV + weighting)
        if self.base_cv is not None:
            def _mk_base(params: Dict) -> Pipeline:
                est = clone(self.base_pipe.named_steps["model"]).set_params(
                    **{k.replace("model__", ""): v for k, v in params.items()}
                )
                steps = ([("scaler", StandardScaler())] if "scaler" in self.base_pipe.named_steps else []) + [("model", est)]
                return Pipeline(steps)
            self.base_pipe = _run_cv_make_best_pipeline(
                _mk_base, X_tr[self.base_cols], y_tr[self.target_cols], self.base_cv, weights=self._weights
            )
        else:
            self.base_pipe.fit(X_tr[self.base_cols], y_tr[self.target_cols])

        base_tr = pd.DataFrame(
            self.base_pipe.predict(X_tr[self.base_cols]),
            index=X_tr.index, columns=self.target_cols
        )
        resid_tr = (y_tr[self.target_cols] - base_tr)

        # 2) Residual stage (CV + weighting)
        if self.res_cv is not None:
            def _mk_res(params: Dict) -> Pipeline:
                est = clone(self.res_pipe.named_steps["model"]).set_params(
                    **{k.replace("model__", ""): v for k, v in params.items()}
                )
                steps = ([("scaler", StandardScaler())] if "scaler" in self.res_pipe.named_steps else []) + [("model", est)]
                return Pipeline(steps)
            self.res_pipe = _run_cv_make_best_pipeline(
                _mk_res, X_tr[self.feature_cols], resid_tr, self.res_cv, weights=self._weights
            )
        else:
            self.res_pipe.fit(X_tr[self.feature_cols], resid_tr)

        # 3) One-step-ahead prediction for x_te
        pred = (
            pd.Series(self.base_pipe.predict(x_te[self.base_cols].to_frame().T)[0], index=self.target_cols)
            + pd.Series(self.res_pipe.predict(x_te[self.feature_cols].to_frame().T)[0], index=self.target_cols)
        )
        return pred

# ==== NEW: CSARMStrategy (per-maturity CS base + generic residual) ==========
class CSARMStrategy(Strategy):
    """
    Base: 각 만기 y_j ~ s_j  (Campbell–Shiller: 단일 slope, OLS)
    Residual: 임의의 특징(feature_cols)로 회귀 (ridge/lasso/enet/mlp/rf/et/xgb 등)
              feature_cols가 비어있으면 잔차단을 생략(순수 CS로 동작).
    """
    def __init__(
        self,
        target_cols: List[str],
        slope_map: Dict[str, str],                 # {"xr_2":"s_2", ...}
        feature_cols: Optional[List[str]] = None,  # 잔차용 특징(없으면 None/[])
        residual_kind: str = "ridge",
        residual_params: Optional[Dict] = None,
        scale_res: bool = True,
        res_cv: Optional[Dict] = None,             # {"mode":"tscv"/"holdout", "n_splits":..., "grid":{...}}
    ):
        super().__init__(target_cols)
        missing = [t for t in target_cols if t not in slope_map]
        if missing:
            raise ValueError(f"slope_map lacks targets: {missing}")
        self.slope_map = dict(slope_map)
        self.feature_cols = list(feature_cols) if feature_cols else []
        self.scale_res = bool(scale_res)
        self.res_cv = res_cv

        # 잔차 추정기 선택
        rk = residual_kind.lower()
        params = residual_params or {}
        if rk == "ridge":
            self.res_est = Ridge(**params)
        elif rk == "lasso":
            self.res_est = Lasso(**params)
        elif rk == "enet":
            self.res_est = ElasticNet(**params)
        elif rk == "mlp":
            self.res_est = MLPRegressor(**params)
        elif rk == "rf":
            self.res_est = RandomForestRegressor(**params)
        elif rk == "et":
            self.res_est = ExtraTreesRegressor(**params)
        elif rk == "xgb":
            if not _HAS_XGB:
                raise RuntimeError("xgboost not installed")
            self.res_est = XGBRegressor(**params)  # type: ignore
        else:
            raise ValueError("Unsupported residual_kind")

        # fitted 객체(디버깅용)
        self._base_models: Dict[str, LinearRegression] = {}
        self._res_pipes: Dict[str, Pipeline] = {}

    def _make_res_pipe(self, param_overrides: Dict) -> Pipeline:
        # grid 키가 'alpha' 또는 'model__alpha' 등이어도 허용
        est = clone(self.res_est).set_params(
            **{k.replace("model__", ""): v for k, v in param_overrides.items()}
        )
        steps = ([("scaler", StandardScaler())] if self.scale_res else []) + [("model", est)]
        return Pipeline(steps)

    def fit_predict(self, X_tr: pd.DataFrame, y_tr: pd.DataFrame, x_te: pd.Series) -> pd.Series:
        preds: Dict[str, float] = {}

        # 잔차 입력(공통). 특징이 없으면 None 처리
        if self.feature_cols:
            Xf_tr = X_tr[self.feature_cols]
            xf_te = x_te[self.feature_cols].to_frame().T
        else:
            Xf_tr = None
            xf_te = None

        for ycol in self.target_cols:
            scol = self.slope_map[ycol]

            # 1) Base: y_j ~ s_j (OLS)
            base = LinearRegression()
            base.fit(X_tr[[scol]], y_tr[[ycol]])
            base_tr = base.predict(X_tr[[scol]]).ravel()
            resid_tr = pd.DataFrame(
                y_tr[[ycol]].to_numpy(float).ravel() - base_tr,
                index=X_tr.index, columns=[ycol]
            )
            self._base_models[ycol] = base

            # 2) Residual (선택)
            if Xf_tr is not None and not Xf_tr.empty:
                if self.res_cv is not None:
                    res_pipe = _run_cv_make_best_pipeline(
                        lambda params: self._make_res_pipe(params),
                        Xf_tr, resid_tr, self.res_cv, weights=None
                    )
                else:
                    res_pipe = self._make_res_pipe({})
                    res_pipe.fit(Xf_tr, resid_tr)
                self._res_pipes[ycol] = res_pipe
                res_te = float(np.asarray(res_pipe.predict(xf_te)).ravel()[0])
            else:
                res_te = 0.0

            # 3) 예측 = base + residual
            base_te = float(base.predict(x_te[[scol]].to_numpy(float).reshape(1, -1)).ravel()[0])
            preds[ycol] = base_te + res_te

        return pd.Series(preds, index=self.target_cols)
    
# ======================================================================
# Factory
# ======================================================================
def make_strategy(name: str, **cfg) -> Strategy:
    """
    name ∈ {"OLS","Ridge","Lasso","ElasticNet","RF","ExtraTrees","XGB","MLP","ARM"}
    Common cfg:
        - target_cols: List[str]
        - params: Dict (estimator params)
        - scale: bool (default True for linear/MLP; False for trees recommended)
        - cv: Optional[Dict] (holdout/tscv config; see _run_cv_make_best_pipeline)
        - target_weights: Optional[Dict[str, float] | List[float] | np.ndarray]
    ARM cfg:
        - base_cols, feature_cols
        - base_kind, base_params, residual_kind, residual_params
        - scale_base, scale_res
        - base_cv, res_cv
        - target_weights
    """
    tgt = cfg.get("target_cols")
    if not tgt:
        raise ValueError("target_cols must be provided")

    tw = cfg.get("target_weights", None)

    # Linear family
    if name == "OLS":
        return SklearnRegStrategy(tgt, LinearRegression(**cfg.get("params", {})), scale=cfg.get("scale", True), cv=cfg.get("cv"), target_weights=tw)
    if name == "Ridge":
        return SklearnRegStrategy(tgt, Ridge(**cfg.get("params", {})), scale=cfg.get("scale", True), cv=cfg.get("cv"), target_weights=tw)
    if name == "Lasso":
        return SklearnRegStrategy(tgt, Lasso(**cfg.get("params", {})), scale=cfg.get("scale", True), cv=cfg.get("cv"), target_weights=tw)
    if name == "ElasticNet":
        return SklearnRegStrategy(tgt, ElasticNet(**cfg.get("params", {})), scale=cfg.get("scale", True), cv=cfg.get("cv"), target_weights=tw)

    # Trees / Ensembles / MLP / XGB
    if name == "RF":
        return SklearnRegStrategy(tgt, RandomForestRegressor(**cfg.get("params", {})), scale=cfg.get("scale", False), cv=cfg.get("cv"), target_weights=tw)
    if name == "ExtraTrees":
        return SklearnRegStrategy(tgt, ExtraTreesRegressor(**cfg.get("params", {})), scale=cfg.get("scale", False), cv=cfg.get("cv"), target_weights=tw)
    if name == "MLP":
        return SklearnRegStrategy(tgt, MLPRegressor(**cfg.get("params", {})), scale=cfg.get("scale", True), cv=cfg.get("cv"), target_weights=tw)
    if name == "XGB":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed")
        return SklearnRegStrategy(tgt, XGBRegressor(**cfg.get("params", {})), scale=cfg.get("scale", False), cv=cfg.get("cv"), target_weights=tw)

    # ARM (two-stage)
    if name == "ARM":
        return ARMStrategy(
            target_cols=tgt,
            base_cols=cfg["base_cols"],
            feature_cols=cfg["feature_cols"],
            base_kind=cfg.get("base_kind", "ols"),
            base_params=cfg.get("base_params", {}),
            residual_kind=cfg.get("residual_kind", "mlp"),
            residual_params=cfg.get("residual_params", {}),
            scale_base=cfg.get("scale_base", True),
            scale_res=cfg.get("scale_res", True),
            base_cv=cfg.get("base_cv"),
            res_cv=cfg.get("res_cv"),
            target_weights=tw,
        )
    
    if name in ("CSARM", "ARM_CS"):
        return CSARMStrategy(
            target_cols=cfg["target_cols"],
            slope_map=cfg["slope_map"],                 # {"xr_2":"s_2", ...}
            feature_cols=cfg.get("feature_cols", []),   # 임의 특징(없으면 [])
            residual_kind=cfg.get("residual_kind", "ridge"),
            residual_params=cfg.get("residual_params", {}),
            scale_res=cfg.get("scale_res", True),
            res_cv=cfg.get("res_cv"),
        )

    raise ValueError(f"Unknown strategy: {name}")