# rolling_framework/analytics_mixin.py
from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.api as sm
from scipy.stats import t as tstat

class AnalyticsMixin:
    # Machine.rec (Recorder) 와 self.y 연결이 있다고 가정
    # -----------------------------------------------------------
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
            else:                           # baseline == 'zero'
                bench = pd.Series(0.0, index=self.targets)

            ss_tot = (yt - bench) ** 2
            ss_res_tot += ss_res
            ss_tot_tot += ss_tot

        return 1 - ss_res_tot / ss_tot_tot
    
    def compare(self, target_col=None) -> pd.DataFrame:
        cols = self.targets
        if target_col is None: target_col = cols[0]
        records = []
        for ds in self.test_dates:
            if ds in self.rec.oos_pred:
                yt = self.y.loc[ds, target_col]
                yp = self.rec.oos_pred[ds].loc[ds, target_col]
                records.append({"date": pd.to_datetime(ds, format='%Y%m'),
                                "y_true": yt, "y_pred": yp})
        return pd.DataFrame(records).set_index("date")

    def CumSSE_series(self, target_col=None, baseline='condmean',
                      use_global_mean=False) -> pd.Series:
        if target_col is None: target_col = self.targets[0]
        errs_m, errs_b, idx = [], [], []
        gmean = self.y.mean() if use_global_mean else None
        for ds in self.test_dates:
            if ds not in self.rec.oos_pred: continue
            yt = self.y.loc[ds, target_col]
            yp = self.rec.oos_pred[ds].loc[ds, target_col]
            if baseline == 'condmean':
                bench = self.y.loc[:ds, target_col].iloc[:-1].mean()
            elif baseline == 'naive':
                bench = gmean[target_col] if use_global_mean else self.y.loc[:ds, target_col].mean()
            else:
                bench = 0.0
            errs_m.append((yt - yp)**2); errs_b.append((yt - bench)**2); idx.append(ds)
        cm = pd.Series(errs_m, index=pd.to_datetime(idx, format='%Y%m')).cumsum()
        cb = pd.Series(errs_b, index=cm.index).cumsum()
        return cb - cm

    def RSZ_Signif(self, target=None):
        cols = [target] if target else self.targets
        pvals = {}
        for col in cols:
            tv, pv = [], []
            for ds in self.test_dates:
                if ds in self.rec.oos_pred:
                    tv.append(self.y.loc[ds, col])
                    pv.append(self.rec.oos_pred[ds].loc[ds, col])
            tv, pv = np.asarray(tv), np.asarray(pv)
            valid = (~np.isnan(tv)) & (~np.isnan(pv))
            if not valid.any(): pvals[col] = np.nan; continue
            diff = (tv - pv)**2
            bench = (tv - np.cumsum(tv)/np.arange(1,len(tv)+1))**2
            f = bench[valid] - diff[valid]
            ols = sm.OLS(f, np.ones_like(f)).fit(cov_type='HAC', cov_kwds={'maxlags':12})
            pvals[col] = 1 - tstat.cdf(float(ols.tvalues), int(ols.nobs)-1)
        return pvals[target] if target else pvals
    
    def compare_plot(
        self,
        target_col: str | None = None,
        start: str | None = None,
        end:   str | None = None,
        figsize=(10, 4),
        scatter=False,
        grid=True,
        title=None,
    ):
        """
        Parameters
        ----------
        target_col : str, 기본값 None
            타깃 변수(컬럼명). None이면 첫 번째 타깃 사용.
        start, end : '%Y%m' 문자열 또는 None
            플롯할 구간 필터.
        figsize    : tuple
            plt.figure 크기.
        scatter    : bool
            True 면 점(scatter) + 선, False 면 선만.
        grid       : bool
            True 면 격자 표시.
        title      : str
            그래프 제목. None 이면 자동 생성.
        """
        col = target_col or self.targets[0]

        # --- 데이터 수집 ----------------------------------------
        dates, yt, yp = [], [], []
        for ds in self.test_dates:
            if ds not in self.rec.oos_pred: continue
            if (start and ds < start) or (end and ds > end): continue
            dates.append(pd.to_datetime(ds, format='%Y%m'))
            yt.append(self.y.loc[ds, col])
            yp.append(self.rec.oos_pred[ds].loc[ds, col])

        if not dates:
            raise ValueError("해당 기간·컬럼에 OOS 예측이 없습니다.")

        # --- 시각화 --------------------------------------------
        plt.figure(figsize=figsize)
        if scatter:
            plt.scatter(dates, yt, label="y_true", s=28)
            plt.scatter(dates, yp, label="y_pred", s=28)
        else:
            plt.plot(dates, yt, label="y_true", lw=1.5)
            plt.plot(dates, yp, label="y_pred", lw=1.5)

        plt.xlabel("Date");  plt.ylabel(col)
        plt.title(title or f"True vs Predicted – {col}")
        if grid: plt.grid(alpha=0.3)
        plt.legend();  plt.tight_layout();  plt.show()


    # ─────────────────── ① 가중치 생성 ───────────────────
    def _make_weights(self,
                    cols: list[str],
                    scheme: str | pd.Series | np.ndarray | None = "equal"
                    ) -> pd.Series:
        """
        Returns
        -------
        pd.Series(index=cols, name='w')  –  항상 합계 1 로 정규화
        """
        # ── ① 균등 가중 ───────────────────────────────────────────────
        if scheme is None or (isinstance(scheme, str) and scheme == "equal"):
            w = pd.Series(1.0 / len(cols), index=cols, name="w")

        # ── ③ 사용자 지정(array-like / Series) ───────────────────────
        else:
            w = pd.Series(scheme, index=cols, dtype=float, name="w")
            w = w / w.sum()                               # 자동 정규화

        return w

    # ─────────────────── ② 포트폴리오 시계열 ───────────────────
    def portfolio_series(self,
                        cols: list[str] | None = None,
                        weights: str | pd.Series | np.ndarray | None = "equal",
                        kind: str = "pred",            # {'pred', 'true'}
                        name: str | None = None
                        ) -> pd.Series:
        """
        kind='pred' → OOS 예측(deep copy X)  ·  kind='true' → 실현 수익

        weights:
            "equal"  → 균등가중
            Series/ndarray  → 사용자 정의 가중치 (자동 정규화)
        """
        cols = cols or self.targets
        w    = self._make_weights(cols, weights)

        # --- 만기별 시계열 집계 -----------------------------------------
        if kind == "pred":
            df = pd.concat(
                {ds: self.rec.oos_pred[ds].loc[ds, cols]
                for ds in self.test_dates if ds in self.rec.oos_pred},
                axis=1,
            ).T.sort_index()
        else:                                    # kind == 'true'
            df = self.y.loc[self.test_dates, cols]

        port = df.mul(w, axis=1).sum(axis=1)

        # ── 이름 자동 지정 ------------------------------------------------
        if name is None:
            label = "EW" if (isinstance(weights, str) and weights == "equal") else "W"
            pre   = "yhat" if kind == "pred" else "ytrue"
            name  = f"{pre}_{label}"
        port.name = name
        return port


    # ─────────────────── ③ R²-OOS 계산 ───────────────────
    def r2_oos_portfolio(self,
                        cols=None,
                        weights="equal",
                        baseline="condmean",      # {'condmean','naive','zero'}
                        use_global_mean=False,
                        return_full=False):
        """
        baseline
            'condmean' : 과거 평균(rolling)
            'naive'    : 전체 평균 (use_global_mean=True) or expanding 평균
            'zero'     : 0
        """
        yhat = self.portfolio_series(cols, weights, kind="pred")
        ytru = self.portfolio_series(cols, weights, kind="true")

        # --- 벤치마크 ----------------------------------------------------
        if baseline == "condmean":
            bench = ytru.expanding(min_periods=2).mean().shift()
        elif baseline == "naive":
            bench = ytru.mean() if use_global_mean else ytru.expanding().mean()
        else:                               # 'zero'
            bench = 0.0

        # --- R²-OOS ------------------------------------------------------
        r2 = 1 - ((ytru - yhat)**2).sum() / ((ytru - bench)**2).sum()

        return (r2, ytru, yhat, bench) if return_full else r2 
    
    # --------Temp : recorder.py와 일부 겹치는 문제 / MSE 출력용---------

    def _oos_pred(self, cols=None):
        """
        Recorder(self.rec).oos_pred 에 저장된 테스트기간 예측들을 모아
        (yhat, ytru) 튜플로 반환.
          - yhat: pd.DataFrame (index=테스트 날짜, columns=만기)
          - ytru: pd.DataFrame (동일 index/columns 정렬)
        """
        # 1) 컬럼 결정
        if cols is None:
            cols = list(self.y.columns)
        else:
            cols = list(cols)

        # 2) 날짜별로 recorder에서 꺼내 붙이기
        frames = []
        for ds in getattr(self, "test_dates", []):
            block = getattr(self.rec, "oos_pred", {}).get(ds)
            if block is None:
                continue

            # 대부분 block.loc[ds, cols] 가 동작 (Series 또는 1xK DF)
            row = block.loc[ds, cols] if set(cols).issubset(block.columns) else block.loc[ds]

            # Series → 1xK DF 통일
            if isinstance(row, pd.Series):
                row = row.to_frame().T

            # 인덱스(시간) 부여
            row.index = pd.Index([ds], name=self.y.index.name or "Time")
            frames.append(row)

        if not frames:
            raise RuntimeError("Recorder.oos_pred에 테스트 예측이 없습니다. training() 실행 여부를 확인하세요.")

        # 3) 예측/실제 정렬
        yhat = pd.concat(frames, axis=0).sort_index()
        ytru = self.y.loc[yhat.index, cols]
        return yhat, ytru


    def _train_stats(self, cols):
        """테스트 첫 날짜 이전 구간을 train으로 보고 만기별 std/var 계산."""
        first_test = min(self.test_dates) if hasattr(self, "test_dates") else None
        if first_test is None:
            # fallback: 전체에서
            df_train = self.y.loc[:, cols]
        else:
            # 인덱스가 YYYYMM 문자열이라면 사전식 비교로도 정렬 가능(동일 포맷 가정)
            df_train = self.y.loc[self.y.index < first_test, cols]
            if df_train.empty:
                df_train = self.y.loc[:, cols]
        std = df_train.std(ddof=1)
        var = df_train.var(ddof=1)
        return std, var

    def MSEOOS(self, cols=None, rmse: bool=False, return_full: bool=False,
                normalize: str | None = None,  # 'std' | 'var' | 'ratio' | None
                naive: str = "mean"            # normalize='ratio'일 때 기준
                ):
        """
        테스트 기간 평균제곱오차(만기별).
          - rmse=True면 제곱근을 취함
          - normalize:
              None  : 원래대로 MSE / RMSE
              'var' : NMSE = MSE / Var(y_train)
              'std' : NRMSE = RMSE / Std(y_train) (rmse=False이어도 RMSE로 바꿔 std로 나눔)
              'ratio': MSE / MSE_naive  (naive='mean'이면 훈련구간 평균 예측)
        return_full=True면 (series, ytrue, yhat) 반환.
        """
        # 1) OOS 예측/실제
        yhat, ytru = self._oos_pred(cols)
        cols = list(ytru.columns)

        # 2) 기본 MSE/RMSE
        err = ytru - yhat
        mse = (err ** 2).mean(axis=0)  # Series (만기별)

        # 3) 정규화 옵션
        if normalize is None:
            out = np.sqrt(mse) if rmse else mse

        elif normalize == "var":
            # NMSE = MSE / Var(y_train)
            _, var = self._train_stats(cols)
            out = mse / var
            if rmse:
                # 요청이 rmse=True면 NRMSE = sqrt(MSE)/sqrt(Var) 로 반환
                out = np.sqrt(mse) / np.sqrt(var)

        elif normalize == "std":
            # NRMSE = RMSE / Std(y_train)
            std, _ = self._train_stats(cols)
            out = np.sqrt(mse) / std  # rmse 여부와 무관하게 RMSE로 정규화하는 게 일반적

        elif normalize == "ratio":
            # 모델 MSE를 '나이브' 대비 비율로
            if naive == "mean":
                # 훈련구간 평균(고정)을 예측으로 사용
                first_test = min(self.test_dates) if hasattr(self, "test_dates") else None
                if first_test is None:
                    y_train = self.y.loc[:, cols]
                else:
                    y_train = self.y.loc[self.y.index < first_test, cols]
                    if y_train.empty:
                        y_train = self.y.loc[:, cols]
                mu = y_train.mean(axis=0)
                mse_naive = ((ytru - mu) ** 2).mean(axis=0)
            else:
                raise ValueError("naive 옵션은 현재 'mean'만 지원합니다.")
            out = mse / mse_naive
            if rmse:
                out = np.sqrt(mse / mse_naive)
        else:
            raise ValueError("normalize must be one of {None,'var','std','ratio'}.")

        if return_full:
            return out, ytru, yhat
        return out
        
