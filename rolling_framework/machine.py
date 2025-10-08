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
        per_maturity: bool = True,
        baseline: str = "naive",
        cs_path: Optional[str] = None,
    ):
        """
        Compute out-of-sample R²:

            R2_OOS = 1 - SSE(model) / SSE(baseline)

        Baselines:
          • "naive"    : denominator = || y ||^2
          • "cs_yhat"  : denominator = || y - cs_yhat ||^2
          • "condmean" : denominator = || y - mean_t(cs_yhat) ||^2
                          (cross-sectional mean at each time t, repeated across maturities)

        Assumptions:
          • If baseline uses cs_yhat, the CSV at `cs_path` is pre-aligned to OOS dates
            and has identical columns to y (no internal shifting here).
        """
        Y_true, Y_pred = self._collect_oos_frames()
        if Y_true.empty or Y_pred.empty:
            return np.nan if not per_maturity else pd.Series(dtype=float, index=self.targets)

        # Ensure same ordering as self.targets for readability
        Y_true = Y_true.reindex(columns=self.targets)
        Y_pred = Y_pred.reindex(columns=self.targets)

        # Numerator
        sse_model = ((Y_true - Y_pred) ** 2).sum(axis=0, min_count=1)

        b = str(baseline).lower()
        if b == "naive" or cs_path is None:
            # denominator = || y ||^2
            sse_base = (Y_true ** 2).sum(axis=0, min_count=1)
        else:
            cs = pd.read_csv(cs_path, index_col="Time")
            # Hard align; no shift or offset applied
            cs = cs.reindex(index=Y_true.index, columns=Y_true.columns)

            if b == "cs_yhat":
                base = cs
            elif b == "condmean":
                mu = cs.mean(axis=1)  # time-t cross-sectional mean
                base = pd.DataFrame(
                    np.repeat(mu.values.reshape(-1, 1), len(Y_true.columns), axis=1),
                    index=Y_true.index, columns=Y_true.columns
                )
            else:
                # Fallback to naive if unknown baseline
                base = pd.DataFrame(0.0, index=Y_true.index, columns=Y_true.columns)

            sse_base = ((Y_true - base) ** 2).sum(axis=0, min_count=1)

        r2 = 1.0 - (sse_model / sse_base.replace(0.0, np.nan))
        return r2 if per_maturity else float(np.nanmean(r2.values))

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