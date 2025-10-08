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
        cs_path: Optional[str] = None,           # path to data/cs_yhat.csv (for baseline="cs_yhat")
        cs_df: Optional[pd.DataFrame] = None,    # or pass a preloaded DataFrame
        per_maturity: bool = True,               # True: return Series by maturity, False: scalar mean
    ):
        """
        R2_OOS = 1 - sum_t ||y_t - ŷ_t||^2 / sum_t ||y_t - b_t||^2

        baselines:
          - naive    : b_t = 0  (vector of zeros)
          - condmean : b_t = time-series mean of y up to t-1 (per-maturity historical mean)
          - cs_yhat  : b_t = external cross-sectional baseline at time t (load from cs_yhat)

        Assumptions:
          - Data are already aligned externally (no internal shifting here).
        """
        targets = self.targets  # e.g., ["xr_2", "xr_3", ...]
        y_all = self.y

        # ----- (optional) cs_yhat ready only if needed -----
        cs = None
        if baseline == "cs_yhat":
            if cs_df is not None:
                cs = cs_df.copy()
            elif cs_path is not None:
                cs = pd.read_csv(cs_path, index_col="Time")
            else:
                # No cs provided → cannot compute CS baseline
                return pd.Series(np.nan, index=targets) if per_maturity else float("nan")

            # align to y: same index (as str) and same columns (targets)
            cs.index = cs.index.astype(str)
            y_index = y_all.index.astype(str)
            cs = cs.reindex(index=y_index, columns=targets)

        # ----- accumulators -----
        ss_res_tot = pd.Series(0.0, index=targets)  # numerator: sum ||y - ŷ||^2
        ss_tot_tot = pd.Series(0.0, index=targets)  # denom   : sum ||y - b||^2

        # ----- loop over test dates -----
        for ds in self.test_dates:
            ds = str(ds)
            if ds not in y_all.index or ds not in self.rec.oos_pred:
                continue

            # y_t
            yt = y_all.loc[ds].reindex(targets).astype(float)

            # ŷ_t (accept Series or 1-row DataFrame)
            yp_obj = self.rec.oos_pred[ds]
            if isinstance(yp_obj, pd.DataFrame):
                if ds in yp_obj.index:
                    yp = yp_obj.loc[ds]
                else:
                    yp = yp_obj.squeeze()
            else:
                yp = yp_obj
            yp = pd.Series(yp, index=targets, dtype=float)

            # baseline b_t
            if baseline == "naive":
                bench = pd.Series(0.0, index=targets)
            elif baseline == "condmean":
                # historical mean up to t-1, per maturity
                hist = y_all.loc[:ds].iloc[:-1]  # strictly before t
                if hist.empty:
                    # no history yet → skip this t
                    continue
                bench = hist.mean().reindex(targets).astype(float)
            elif baseline == "cs_yhat":
                if cs is None or ds not in cs.index:
                    continue
                bench = cs.loc[ds].reindex(targets).astype(float)
            else:
                raise ValueError("baseline must be 'naive', 'condmean', or 'cs_yhat'.")

            # accumulate squared errors (ignore NaN by filling 0)
            inc_res = ((yt - yp) ** 2).fillna(0.0)
            inc_tot = ((yt - bench) ** 2).fillna(0.0)

            ss_res_tot = ss_res_tot.add(inc_res, fill_value=0.0)
            ss_tot_tot = ss_tot_tot.add(inc_tot, fill_value=0.0)

        # ----- final R2 -----
        denom = ss_tot_tot.replace(0.0, np.nan)
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