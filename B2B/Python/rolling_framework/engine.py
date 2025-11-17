# rolling_framework/engine.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.io import savemat

from .utils import add_months, r2_oos
from .strategies import Strategy
from tqdm.auto import tqdm


class ExpandingRunner:
    """
    Expanding-window forecasting engine with an initial embargo after burn-in.

    Design
    ------
    - X, y are assumed to be pre-aligned by the user.
    - period = [start, end] in "YYYYMM" strings.
    - burn_in_end = "YYYYMM": last date included in the burn-in.
    - horizon: initial embargo (in months) AND rolling embargo between train and test.

      First test date:
          t_0 = max(burn_in_end, period_start) shifted by +horizon months.

      For each test date t:
          - if horizon > 0: training uses dates <= add_months(t, -horizon)
          - if horizon = 0: training uses dates < t
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        strategy: Strategy,
        period: List[str],            # ["YYYYMM","YYYYMM"]
        burn_in_end: str,             # "YYYYMM"
        horizon: int = 0,             # initial embargo in months (e.g., 12)
    ):
        # ensure string index for time
        self.X = X.copy()
        self.X.index = self.X.index.astype(str)
        self.y = y.copy()
        self.y.index = self.y.index.astype(str)

        self.strategy = strategy

        self.start = str(period[0])
        self.end = str(period[1])
        self.burn_in_end = str(burn_in_end)
        self.horizon = int(horizon)

        # intersection and clipping to period
        idx = self.X.index.intersection(self.y.index).astype(str)
        idx = idx[(idx >= self.start) & (idx <= self.end)]
        self.times = sorted(idx)

        # first test time after burn-in and horizon embargo
        first = max(self.burn_in_end, self.start)
        first_test = add_months(first, self.horizon)
        self.test_times = [t for t in self.times if t >= first_test]

        # container for OOS predictions (dict: time -> Series)
        self.oos_pred: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # core expanding walk
    # ------------------------------------------------------------------
    def fit_walk(self, progress: bool = True, desc: str = "Expanding walk") -> "ExpandingRunner":
        """
        Run expanding-window forecasting with an optional horizon embargo.

        If self.horizon > 0, the training window at test date t uses observations
        up to add_months(t, -self.horizon), so that the realized return window
        for horizon-step targets does not overlap between train and test.

        If self.horizon == 0, this reduces to the usual expanding scheme using
        all dates strictly earlier than t.
        """
        bar = tqdm(
            self.test_times,
            total=len(self.test_times),
            desc=desc,
            unit="step",
            disable=not progress,
        )

        for t in bar:
            # training window according to horizon embargo
            if self.horizon > 0:
                cutoff = add_months(t, -self.horizon)
                tr = [s for s in self.times if s <= cutoff]
            else:
                tr = [s for s in self.times if s < t]

            if not tr:
                continue

            X_tr = self.X.loc[tr]
            y_tr = self.y.loc[tr, self.strategy.target_cols]
            x_te = self.X.loc[t]

            if progress:
                bar.set_postfix_str(f"t={t} | train={len(tr)}")

            # delegate to strategy
            self.oos_pred[t] = self.strategy.fit_predict(X_tr, y_tr, x_te)

        return self

    # ------------------------------------------------------------------
    # collect outputs
    # ------------------------------------------------------------------
    def collect_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (Y_true, Y_pred) as aligned DataFrames restricted to realized test dates.
        """
        idx = [t for t in self.test_times if t in self.oos_pred]
        Y_true = self.y.loc[idx, self.strategy.target_cols]
        Y_pred = pd.DataFrame([self.oos_pred[t] for t in idx], index=idx)
        return Y_true, Y_pred

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def R2OOS(
        self,
        baseline: str = "condmean",                 # "condmean" | "naive" | "custom"
        benchmark: Optional[pd.DataFrame] = None,   # used if baseline == "custom"
    ) -> pd.Series:
        """
        Compute out-of-sample R^2 for each target column.
        """
        Y_true, Y_pred = self.collect_frames()
        return r2_oos(Y_true, Y_pred, baseline=baseline, benchmark=benchmark)

    # Authors Comments : It doesn't need to split, but it's fine as is.
    def MSE_frame(self) -> pd.DataFrame:
        """
        Return squared errors for each time and target:
        SE[t, j] = (Y_true[t, j] - Y_pred[t, j])^2.
        """
        Y_true, Y_pred = self.collect_frames()
        se = (Y_pred - Y_true) ** 2
        return se

    def MSE_time(self, avg_over_targets: bool = True):
        """
        MSE over time.

        If avg_over_targets is True:
            returns pd.Series indexed by time:
                MSE_t = mean_j SE[t, j]
        Else:
            returns full SE DataFrame (time × targets).
        """
        se = self.MSE_frame()
        if avg_over_targets:
            return se.mean(axis=1)
        return se
    # ------------------------------------------------------------------
    # export to MATLAB .mat
    # ------------------------------------------------------------------
    def to_mat(self, path: str, baseline: str = "condmean", benchmark: Optional[pd.DataFrame] = None):
        Y_true, Y_pred = self.collect_frames()
        out = {
            "Y_true": Y_true.to_numpy(float),
            "Y_pred": Y_pred.to_numpy(float),
            "dates":  np.array(Y_true.index.tolist(), dtype=object),
            "maturities": np.array(Y_true.columns.tolist(), dtype=object),
            "horizon": np.array([self.horizon]),
            "burn_in_end": np.array([self.burn_in_end], dtype=object),
        }

        # R^2_oos
        r2 = self.R2OOS(baseline=baseline, benchmark=benchmark)
        out["R2OOS"] = r2.to_numpy(float)

        # Squared errors and time-wise MSE
        se = (Y_pred - Y_true) ** 2
        out["SE"] = se.to_numpy(float)                    # shape: (T, J)
        out["MSE_t"] = se.mean(axis=1).to_numpy(float)    # shape: (T,)

        # custom baseline (e.g., CS OLS)
        if benchmark is not None:
            bench = benchmark.loc[Y_true.index, Y_true.columns].astype(float)
            out["Y_cs_hat"] = bench.to_numpy(float)

        savemat(path, out)