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
    Expanding window engine with an initial embargo after burn-in.
    - No internal shifting of y (assume X, y already aligned by user).
    - First test date = max(burn_in_end, period_start) advanced by `horizon` months.
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
        self.X = X.copy(); self.X.index = self.X.index.astype(str)
        self.y = y.copy(); self.y.index = self.y.index.astype(str)
        self.strategy = strategy

        self.start = str(period[0]); self.end = str(period[1])
        self.burn_in_end = str(burn_in_end)
        self.horizon = int(horizon)

        idx = self.X.index.intersection(self.y.index).astype(str)
        idx = idx[(idx >= self.start) & (idx <= self.end)]
        self.times = sorted(idx)

        first = max(self.burn_in_end, self.start)
        first_test = add_months(first, self.horizon)
        self.test_times = [t for t in self.times if t >= first_test]

        self.oos_pred: Dict[str, pd.Series] = {}

    # ---------------- core walk ----------------
    def fit_walk(self, progress: bool = True, desc: str = "Expanding walk"):
        bar = tqdm(self.test_times, total=len(self.test_times), desc=desc, unit="step", disable=not progress)
        for t in bar:
            tr = [s for s in self.times if s < t]
            if not tr:
                continue
            X_tr, y_tr, x_te = self.X.loc[tr], self.y.loc[tr], self.X.loc[t]
            if progress:
                bar.set_postfix_str(f"t={t} | train={len(tr)}")
            self.oos_pred[t] = self.strategy.fit_predict(X_tr, y_tr, x_te)
        return self

    # ---------------- frames -------------------
    def collect_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        idx = [t for t in self.test_times if t in self.oos_pred]
        Y_true = self.y.loc[idx, self.strategy.target_cols]
        Y_pred = pd.DataFrame([self.oos_pred[t] for t in idx], index=idx)
        return Y_true, Y_pred

    # ---------------- metrics ------------------
    def R2OOS(
        self,
        baseline: str = "condmean",                 # "condmean" | "naive" | "custom"
        benchmark: Optional[pd.DataFrame] = None,   # caller-provided benchmark if custom
    ) -> pd.Series:
        Y_true, Y_pred = self.collect_frames()
        return r2_oos(Y_true, Y_pred, baseline=baseline, benchmark=benchmark)

    # ---------------- MATLAB export -----------
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
        r2 = self.R2OOS(baseline=baseline, benchmark=benchmark)
        out["R2OOS"] = r2.to_numpy(float)

        # ---- NEW: custom baseline 저장 (예: CS OLS) ----
        if benchmark is not None:
            bench = benchmark.loc[Y_true.index, Y_true.columns].astype(float)
            out["Y_cs_hat"] = bench.to_numpy(float)   # ← 추가

        savemat(path, out)