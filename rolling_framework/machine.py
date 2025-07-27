from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
from tqdm.auto import tqdm

from .analytics_mixin import AnalyticsMixin
from .dataset_mixin import DatasetMixin
from .recorder import Recorder
from .strategies import STRATEGIES


class Machine(DatasetMixin, AnalyticsMixin):
    """고수준 롤링-윈도우 학습 엔진."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        model_type: str,
        option: Optional[str] = None,
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
        self.strategy = STRATEGIES[model_type](self, option, params_grid)

    # ── 메인 학습 루프 ───────────────────────────────────
    def training(self):
        if not self.test_dates:
            raise ValueError("No test dates generated; check burn-in/period.")
        for ds in tqdm(self.test_dates,
                       total=len(self.test_dates),
                       desc=f"{self.model_type} rolling"):
            self.strategy.fit_predict(ds)

    # ── Out-of-Sample R² ────────────────────────────────
    def R2OOS(self, baseline: str = "condmean", use_global_mean: bool = False):
        ss_res_tot = pd.Series(0.0, index=self.targets)
        ss_tot_tot = pd.Series(0.0, index=self.targets)
        g_mean = self.y.mean() if use_global_mean else None

        for ds in self.test_dates:
            if ds not in self.rec.oos_pred or ds not in self.y.index:
                continue
            yt = self.y.loc[ds]
            yp = self.rec.oos_pred[ds].loc[ds]

            ss_res = (yt - yp) ** 2
            if baseline == "condmean":
                bench = self.y.loc[:ds].iloc[:-1].mean()
            elif baseline == "naive":
                bench = g_mean if use_global_mean else self.y.loc[:ds].mean()
            else:
                bench = pd.Series(0.0, index=self.targets)
            ss_tot = (yt - bench) ** 2

            ss_res_tot += ss_res
            ss_tot_tot += ss_tot

        return 1 - ss_res_tot / ss_tot_tot

    # ── Recorder alias ─────────────────────────────────
    @property
    def best_model(self):  return self.rec.best_model
    @property
    def best_params(self): return self.rec.best_param
    @property
    def y_train_r2(self):  return self.rec.train_r2
    @property
    def y_test_pred(self): return self.rec.oos_pred