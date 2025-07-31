"""개별 모델 Strategy + 공통 BaseStrategy"""
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
import torch
import torch.nn as nn     




class BaseStrategy(ABC):
    def __init__(self, core, option: str, grid: Dict):
        self.core = core
        self.opt  = option
        self.grid = grid or {}

    # ── 공통 fit/predict 로직 ─────────────────────────────
    def fit_predict(self, ds: str):
        Xf, yf = self.core._get_train_full(ds)
        if Xf.empty:
            return

        X_tr, X_val, y_tr, y_val = self.core._split_train_val(Xf, yf)
        X_tv = pd.concat([X_tr, X_val]); y_tv = pd.concat([y_tr, y_val])

        pipe = self.build_pipeline()
        gs = GridSearchCV(
            pipe, self.grid,
            cv=self.core._cv_split(len(X_tr), len(X_val)),
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        gs.fit(X_tv, y_tv)
        best = gs.best_estimator_

        self.core.rec.save_model(ds, best, gs.best_params_)
        self.core.rec.save_train_r2(ds, y_tr, best.predict(X_tr))
        self.core.rec.save_pred(ds, best.predict(self.core.X.loc[[ds]])[0])

    # ── 서브클래스가 파이프라인 정의 ──────────────────────
    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        ...


# ───────────────────────────────── Strategy 구현 ─────────
class OLSStrategy(BaseStrategy):
    def build_pipeline(self):
        return Pipeline([('sc', StandardScaler()),
                         ('lr', LinearRegression())])


class PCRStrategy(BaseStrategy):
    def build_pipeline(self):
        steps = [('sc', StandardScaler()),
                 ('pca', PCA()),
                 ('lr',  LinearRegression())]
        if self.opt == 'squared':
            steps.insert(2, ('poly', PolynomialFeatures(degree=2, include_bias=False)))
            steps.insert(3, ('sc2', StandardScaler()))                                                              # 예상 문제지점
        return Pipeline(steps)


class PLSStrategy(BaseStrategy):
    def build_pipeline(self):
        return Pipeline([('sc', StandardScaler()),
                         ('pls', PLSRegression())])


class PenalizedStrategy(BaseStrategy):
    _BASES = {'ridge': Ridge, 'lasso': Lasso, 'elasticnet': ElasticNet}

    def build_pipeline(self):
        base_cls = self._BASES[self.opt]
        return Pipeline([('sc', StandardScaler()),
                         ('reg', base_cls(random_state=self.core.random_state))])


class TreeStrategy(BaseStrategy):
    _BASES = {
        "XGB":          XGBRegressor,
        "RandomForest": RandomForestRegressor,
        "ExtremeTrees": ExtraTreesRegressor,
        "EBM":          ExplainableBoostingRegressor,
    }

    def build_pipeline(self):
        base_cls = self._BASES[self.opt]

        # --- 1) 원본 추정기 ---------------------------------
        if self.opt == "XGB":
            base_est = base_cls(
                n_jobs      = -1,
                random_state= self.core.random_state,
                objective   = "reg:squarederror",
                eval_metric = "rmse",
            )
        else:  # RF, ET, EBM
            base_est = base_cls(
                n_jobs      = -1,
                random_state= self.core.random_state,
            )

        # --- 2) 다중 타깃 래핑 ------------------------------
        if self.core.y.shape[1] > 1:
            est = MultiOutputRegressor(base_est, n_jobs=-1)
        else:
            est = base_est

        # --- 3) 파이프라인 반환 -----------------------------
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", est),
        ])

from .strategies_torch import (
    TorchDNNStrategy,
    TorchMultiBranchStrategy,
)

# ── dispatcher ───────────────────────────────────────────
STRATEGIES = {
    'OLS':       OLSStrategy,
    'PCR':       PCRStrategy,
    'PLS':       PLSStrategy,
    'Penalized': PenalizedStrategy,
    'Tree':      TreeStrategy,
    'DNN':       TorchDNNStrategy,   # ← 별도 파일에서 정의됨 (strategies_torch.py)
    'DNN_NBR' : TorchMultiBranchStrategy,
}