"""DatasetMixin: 공통 데이터 헬퍼"""
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit


class DatasetMixin:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        burn_in_start: str,
        burn_in_end: str,
        period: list,
        forecast_horizon: int,
        val_ratio: float,
    ):
        self.X = X.copy(); self.X.index = self.X.index.astype(str)
        self.y = y.copy(); self.y.index = self.y.index.astype(str)

        self.burn_in_start = burn_in_start
        self.burn_in_end = burn_in_end
        self.forecast_horizon = forecast_horizon
        self.val_ratio = val_ratio

        start_dt = pd.to_datetime(burn_in_end, format='%Y%m')
        end_dt   = pd.to_datetime(period[1], format='%Y%m') + pd.offsets.MonthEnd(0)
        self.test_dates = pd.date_range(start_dt, end_dt, freq='M').strftime('%Y%m').tolist()

    # ── 학습 구간 추출 ──────────────────────────────────────
    def _get_train_full(self, test_date: str):
        dt = pd.to_datetime(test_date, format='%Y%m')
        train_end = (dt - pd.DateOffset(months=self.forecast_horizon)).strftime('%Y%m')
        Xf = self.X.loc[self.burn_in_start:train_end]
        yf = self.y.loc[self.burn_in_start:train_end]
        return Xf, yf

    # ── train / val 분리 ──────────────────────────────────
    def _split_train_val(self, Xf: pd.DataFrame, yf: pd.DataFrame):
        n_tr = max(int(len(Xf) * (1 - self.val_ratio)), 1)
        return Xf.iloc[:n_tr], Xf.iloc[n_tr:], yf.iloc[:n_tr], yf.iloc[n_tr:]

    # ── PredefinedSplit 생성 ──────────────────────────────
    @staticmethod
    def _cv_split(n_tr: int, n_val: int):
        return PredefinedSplit(np.concatenate([np.zeros(n_tr), np.ones(n_val)]))