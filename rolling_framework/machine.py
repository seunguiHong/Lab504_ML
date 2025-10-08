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

    # ──────────────────────────────────────────────────────────────────────
    # Out-of-sample R² with clean baselines (no shift)
    # ──────────────────────────────────────────────────────────────────────
    def R2OOS(
        self,
        baseline: str = "naive",                 # "naive" | "cs_yhat" | "condmean"
        cs_path: Optional[str] = None,           # data/cs_yhat.csv 경로
        cs_df: Optional[pd.DataFrame] = None,    # 이미 로드해 넘길 수도 있음
        per_maturity: bool = True,               # True면 만기별 Series 반환, False면 평균 스칼라
    ):
        """
        R2_OOS = 1 - sum_t ||y_t - yhat_t||^2 / sum_t ||y_t - b_t||^2
          - baseline='naive'   : b_t = 0
          - baseline='cs_yhat' : b_t = cs_yhat[t, :]
          - baseline='condmean': b_t = mean_j cs_yhat[t, j] (단면 평균, 모든 만기에 동일값)
        가정: 데이터는 사전에 시간 정렬/정합(alignment) 되어 있음.
        """
        targets = self.targets                     # ex) ["xr_2", ...]
        # ----- 1) cs_yhat 준비 (필요할 때만) -----
        cs = None
        if baseline in ("cs_yhat", "condmean"):
            if cs_df is not None:
                cs = cs_df.copy()
            elif cs_path is not None:
                cs = pd.read_csv(cs_path, index_col="Time")
            else:
                # 경로/데이터 미제공이면 cs-기반 R2는 계산 불가 → 전부 NaN 반환
                return pd.Series(np.nan, index=targets) if per_maturity else float("nan")

            # 인덱스/열 정합: 인덱스는 문자열 YYYYMM, 열은 targets 순서
            cs.index = cs.index.astype(str)
            y_index = self.y.index.astype(str)
            cs = cs.reindex(index=y_index, columns=targets)

        # ----- 2) 누적 제곱합 초기화 -----
        ss_res_tot = pd.Series(0.0, index=targets)  # 분자: (y - yhat)^2
        ss_tot_tot = pd.Series(0.0, index=targets)  # 분모: (y - baseline)^2

        # ----- 3) 테스트 구간 루프 -----
        for ds in self.test_dates:
            ds = str(ds)  # 날짜 키를 문자열로 통일

            # 예측/실측 꺼내기
            if ds not in self.rec.oos_pred or ds not in self.y.index:
                continue
            yt = self.y.loc[ds]                    # Series (targets)
            yp = self.rec.oos_pred[ds].loc[ds]     # Series (targets)

            # baseline 벡터 만들기
            if baseline == "naive":
                bench = pd.Series(0.0, index=targets)
            elif baseline == "cs_yhat":
                if ds not in cs.index:
                    # 해당 날짜 cs가 없으면 스킵
                    continue
                bench = cs.loc[ds]                  # Series (targets)
            elif baseline == "condmean":
                if ds not in cs.index:
                    continue
                m = float(np.nanmean(cs.loc[ds].to_numpy(dtype=float)))  # 단면 평균
                bench = pd.Series(m, index=targets)
            else:
                raise ValueError("baseline must be 'naive', 'cs_yhat', or 'condmean'.")

            # 증분 제곱오차 계산 (NaN은 0으로 무시하고 누적)
            inc_res = ((yt - yp) ** 2).reindex(targets).astype(float).fillna(0.0)
            inc_tot = ((yt - bench) ** 2).reindex(targets).astype(float).fillna(0.0)

            ss_res_tot = ss_res_tot.add(inc_res, fill_value=0.0)
            ss_tot_tot = ss_tot_tot.add(inc_tot, fill_value=0.0)

        # ----- 4) R2_OOS 계산 -----
        denom = ss_tot_tot.replace(0.0, np.nan)  # 분모 0이면 NaN 처리
        r2 = 1.0 - (ss_res_tot / denom)

        return r2 if per_maturity else float(np.nanmean(r2.to_numpy(dtype=float)))

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