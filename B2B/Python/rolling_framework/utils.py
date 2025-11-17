# rolling_framework/utils.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def add_months(yyyymm: str, k: int) -> str:
    """Add k months to a YYYYMM string and return new YYYYMM string."""
    y = int(yyyymm[:4])
    m = int(yyyymm[4:])
    total = (y * 12 + (m - 1)) + k
    ny, nm = divmod(total, 12)
    return f"{ny:04d}{nm + 1:02d}"


def r2_oos(
    Y_true: pd.DataFrame,
    Y_pred: pd.DataFrame,
    baseline: str = "condmean",                  # "naive" | "condmean" | "custom"
    benchmark: Optional[pd.DataFrame] = None,    # used only when baseline == "custom"
) -> pd.Series:
    """
    Compute out-of-sample R^2 for each column.

    R2_oos_j = 1 - sum_t (y_tj - yhat_tj)^2 / sum_t (y_tj - b_tj)^2,
    where b_tj is the baseline forecast (naive, condmean, or custom benchmark).
    """
    Y_true = Y_true.astype(float)
    Y_pred = Y_pred.astype(float)

    idx = Y_true.index
    cols = Y_true.columns

    if baseline == "naive":
        # zero-return baseline
        B = pd.DataFrame(0.0, index=idx, columns=cols)
    elif baseline == "condmean":
        # expanding historical mean of returns, one-step lagged
        B = Y_true.expanding(min_periods=1).mean().shift(1)
    elif baseline == "custom":
        if benchmark is None:
            raise ValueError("baseline='custom' requires benchmark=DataFrame")
        # assume caller aligned it already; just clip to idx/cols
        B = benchmark.reindex(index=idx, columns=cols).astype(float)
    else:
        raise ValueError("baseline must be 'naive' | 'condmean' | 'custom'")

    num = ((Y_true - Y_pred) ** 2).sum(axis=0)
    den = ((Y_true - B) ** 2).sum(axis=0).replace(0.0, np.nan)
    return 1.0 - (num / den)