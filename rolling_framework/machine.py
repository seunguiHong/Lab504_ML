from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from .analytics_mixin import AnalyticsMixin
from .dataset_mixin import DatasetMixin
from .recorder import Recorder
from .strategies import STRATEGIES


class Machine(DatasetMixin, AnalyticsMixin):
    """
    Rolling training/evaluation driver.

    Responsibilities:
      • Own the dataset windows and test dates (via DatasetMixin)
      • Hold a Recorder for per-date OOS predictions
      • Instantiate and run a Strategy by name
      • Expose simple analytics (R2OOS etc.)
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        model_type: str,
        option: Optional[Dict] = None,
        params_grid: Optional[Dict] = None,
        burn_in_start: str = "197108",
        burn_in_end:   str = "198801",
        period: List[str] = ["197108", "201712"],
        forecast_horizon: int = 12,
        val_ratio: float = 0.15,
        random_state: int = 42,
    ):
        super().__init__(X, y, burn_in_start, burn_in_end,
                         period, forecast_horizon, val_ratio)

        self.model_type   = model_type
        self.random_state = random_state
        self.targets = (list(y.columns)
                        if hasattr(y, "columns") else [y.name or 0])
        self.rec = Recorder(self.targets)

        if model_type not in STRATEGIES:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.strategy = STRATEGIES[model_type](self, option or {}, params_grid)

    # ──────────────────────────────────────────────────────────────────────
    # Main rolling loop
    # ──────────────────────────────────────────────────────────────────────
    def training(self):
        if not self.test_dates:
            raise ValueError("No test dates generated; check burn-in/period.")
        for ds in tqdm(self.test_dates,
                       total=len(self.test_dates),
                       desc=f"{self.model_type} rolling"):
            self.strategy.fit_predict(ds)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers to gather OOS truth & predictions (no shifting)
    # ──────────────────────────────────────────────────────────────────────
    def _collect_oos_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (Y_true, Y_pred) as DataFrames aligned on common index/columns.
        This reads OOS predictions from Recorder, one row per test date.
        No internal shifting — assumes external alignment is already done.
        """
        rows_true: List[pd.Series] = []
        rows_pred: List[pd.Series] = []

        for ds in self.test_dates:
            if ds not in self.y.index:
                continue
            if ds not in self.rec.oos_pred:   # Recorder stores per-date preds
                continue

            yt = self.y.loc[ds]
            yp_obj = self.rec.oos_pred[ds]

            # Accept either a Series or a single-row DataFrame
            if isinstance(yp_obj, pd.DataFrame):
                yp = yp_obj.loc[ds] if ds in yp_obj.index else yp_obj.squeeze()
            else:
                yp = yp_obj

            # Ensure Series with same index as targets
            yt = yt.reindex(self.targets)
            yp = yp.reindex(self.targets)

            rows_true.append(yt)
            rows_pred.append(yp)

        if not rows_true or not rows_pred:
            return pd.DataFrame(), pd.DataFrame()

        Y_true = pd.DataFrame(rows_true)
        Y_pred = pd.DataFrame(rows_pred)

        # Align on both axes defensively (should already match)
        idx = Y_true.index.intersection(Y_pred.index)
        cols = Y_true.columns.intersection(Y_pred.columns)
        return Y_true.loc[idx, cols], Y_pred.loc[idx, cols]

    def R2OOS(self, baseline: str = "naive", cs_path: str = None):
        """
        Minimal, robust R²_OOS by maturity:
        R² = 1 - Σ_t (y_t - ŷ_t)² / Σ_t (y_t - b_t)²

        baseline:
        - "naive"    : b_t = 0
        - "condmean" : b_t = (각 만기별 과거 평균 up to t-1) = expanding mean 후 shift(1)
        - "cs_yhat"  : b_t = 외부 cs_yhat[t, maturity] (csv 제공)
        Assumptions: training() 완료, Recorder에 OOS 예측 존재.
        """

        targets = list(self.targets)

        # 1) 수집: test_dates에서 실측/예측 둘 다 있는 행만 모은다.
        rows_true, rows_pred = [], []
        for ds in self.test_dates:
            k = str(ds)
            if (k not in self.y.index) or (k not in self.rec.oos_pred):
                continue

            yt = self.y.loc[k]
            yp_obj = self.rec.oos_pred[k]
            yp = yp_obj.loc[k] if isinstance(yp_obj, pd.DataFrame) and (k in yp_obj.index) else yp_obj
            yt = yt.reindex(targets)
            yp = yp.reindex(targets)

            # 실측/예측에 NaN이 섞인 행은 스킵 (필요시 제거)
            if yt.isna().any() or yp.isna().any():
                continue

            rows_true.append(yt.astype(float))
            rows_pred.append(yp.astype(float))

        if not rows_true:
            # 유효한 테스트 행이 하나도 없으면 만기별 NaN 반환
            return pd.Series(np.nan, index=targets)

        Y_true = pd.DataFrame(rows_true)
        Y_pred = pd.DataFrame(rows_pred)
        # 인덱스/컬럼 정렬(방어적)
        idx = Y_true.index
        cols = targets
        Y_pred = Y_pred.reindex(index=idx, columns=cols)

        # 2) baseline 구성
        if baseline == "naive":
            # 항상 유효
            B = pd.DataFrame(0.0, index=idx, columns=cols)
            valid_mask = pd.Series(True, index=idx)

        elif baseline == "condmean":
            # 각 만기별의 과거 평균 → shift(1)로 t시점 정보 누설 방지
            B_full = Y_true.expanding(min_periods=1).mean().shift(1)
            # baseline이 NaN인 초기 행 제거
            valid_mask = ~B_full.isna().any(axis=1)
            if not valid_mask.any():
                return pd.Series(np.nan, index=cols)
            B = B_full.loc[valid_mask]
            Y_true = Y_true.loc[valid_mask]
            Y_pred = Y_pred.loc[valid_mask]

        elif baseline == "cs_yhat":
            if cs_path is None:
                raise ValueError("baseline='cs_yhat' requires cs_path")
            cs = pd.read_csv(cs_path, index_col="Time")

            # 인덱스/열 정규화
            cs.index = (cs.index.astype(str)
                                .str.replace(r"[\-\/\s]", "", regex=True))
            cs = cs[~cs.index.duplicated(keep="last")]
            cs.columns = [str(c).strip() for c in cs.columns]

            # y와 같은 축으로 reindex (결측은 NaN)
            cs = cs.reindex(index=idx, columns=cols).astype(float)

            # baseline이 모두 존재하는 행만 사용 (만기별 모두 유효)
            valid_mask = ~cs.isna().any(axis=1)
            if not valid_mask.any():
                return pd.Series(np.nan, index=cols)
            B = cs.loc[valid_mask]
            Y_true = Y_true.loc[valid_mask]
            Y_pred = Y_pred.loc[valid_mask]

        else:
            raise ValueError("baseline must be 'naive', 'condmean', or 'cs_yhat'")

        # 3) R² 계산 (만기별 Series 반환)
        num = ((Y_true - Y_pred) ** 2).sum(axis=0)
        den = ((Y_true - B)     ** 2).sum(axis=0)

        r2 = 1.0 - num / den.replace(0.0, np.nan)
        return r2

    # ──────────────────────────────────────────────────────────────────────
    # Convenience proxies to Recorder
    # ──────────────────────────────────────────────────────────────────────
    @property
    def best_model(self):  return self.rec.best_model

    @property
    def best_params(self): return self.rec.best_param

    @property
    def y_train_r2(self):  return self.rec.train_r2

    @property
    def y_test_pred(self): return self.rec.oos_pred