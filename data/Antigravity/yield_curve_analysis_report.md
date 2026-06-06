# 금리 기간구조 시계열 분석 및 LaTeX 테이블 요약 보고서
(Yield Curve Time-Series Analysis & LaTeX Tables Summary Report)

본 보고서는 사용자님의 금리 기간구조 데이터셋([target_and_features.mat](file:///Users/ethan_hong/Library/CloudStorage/Dropbox/0_Lab_504/Codes/504_ML/NN/data/target_and_features.mat))을 대상으로 수행한 **기초 요약 통계량 산출**, **Augmented Dickey-Fuller (ADF) 단위근 검정**, 그리고 **KPSS 및 Phillips-Perron (PP) 공동 정상성 검정** 결과를 체계적으로 정리하고, 학술 논문 수준의 LaTeX 표를 손쉽게 복사해 사용할 수 있도록 정리한 최종 결과물입니다.

---

## 1. 생성 및 제공된 파일 내역 (Deliverables)

모든 파일은 워크스페이스 디렉터리 내에 생성 및 배치 완료되었습니다:

| 구분 | 파일명 | 기능 및 구성 항목 |
| :--- | :--- | :--- |
| **MATLAB 스크립트** | [generate_summary_latex.m](file:///Users/ethan_hong/Library/CloudStorage/Dropbox/0_Lab_504/Codes/504_ML/NN/data/Antigravity/generate_summary_latex.m) | 1년 초과수익률, 선도금리, 선도금리 변화량의 요약 통계량(평균, 표준편차, 최솟값, 최댓값, AR(1), 왜도, 첨도) 계산 및 내보내기 |
| | [generate_adf_latex.m](file:///Users/ethan_hong/Library/CloudStorage/Dropbox/0_Lab_504/Codes/504_ML/NN/data/Antigravity/generate_adf_latex.m) | AIC(Akaike Information Criterion)를 활용한 최적 시차(Lag) 자동 선택 기반 ADF 검정 수행 및 내보내기 |
| | [generate_kpss_pp_latex.m](file:///Users/ethan_hong/Library/CloudStorage/Dropbox/0_Lab_504/Codes/504_ML/NN/data/Antigravity/generate_kpss_pp_latex.m) | Newey-West 자동 시차/대역폭 선택 방식을 반영한 KPSS 정상성 검정 및 Phillips-Perron(PP) 단위근 공동 검정 수행 및 내보내기 |
| **LaTeX 표 파일** | [summary_table.tex](file:///Users/ethan_hong/Library/CloudStorage/Dropbox/0_Lab_504/Codes/504_ML/NN/data/Antigravity/summary_table.tex) | 실제 데이터를 사용하여 계산을 끝마친 고품질 요약 통계량 LaTeX 코드 |
| | [adf_table.tex](file:///Users/ethan_hong/Library/CloudStorage/Dropbox/0_Lab_504/Codes/504_ML/NN/data/Antigravity/adf_table.tex) | 실제 데이터를 사용하여 계산을 끝마친 고품질 ADF 단위근 검정 결과 LaTeX 코드 |
| | [kpss_pp_table.tex](file:///Users/ethan_hong/Library/CloudStorage/Dropbox/0_Lab_504/Codes/504_ML/NN/data/Antigravity/kpss_pp_table.tex) | 실제 데이터를 사용하여 계산을 끝마친 고품질 KPSS & PP joint 검정 결과 LaTeX 코드 |

---

## 2. 요약 통계량 결과 및 LaTeX 코드 (Summary Statistics)

본 분석에서는 초과수익률 `rx`, 선도금리 `fwd`, 12개월 선도금리 변화량 `yoy_fwd`에 대해 통계량을 구했습니다. 가독성을 위해 **선도금리와 변화량 변수는 백분율(\%) 단위로 스케일링** 하였습니다.

```latex
% Publication-quality Yield Curve Summary Statistics Table
\begin{table}[tbp]
  \centering
  \caption{Summary Statistics for Yield Curve Excess Returns and Forwards}
  \label{tab:yield_summary_stats}
  \begin{tabular}{lccccccccc}
    \toprule
    Maturity & 2Y & 3Y & 4Y & 5Y & 6Y & 7Y & 8Y & 9Y & 10Y \\
    \midrule
    \multicolumn{10}{l}{\textbf{Panel A: One-Year Excess Returns ($rx_t^{(n)}$, \%)}} \\
    \noalign{\vskip 2pt}
    Mean               &   0.3747 &   0.6697 &   0.9405 &   1.0426 &   1.2959 &   1.3138 &   1.8003 &   1.8719 &   1.8960 \\
    Std. Dev.          &   1.6579 &   3.0302 &   4.2268 &   5.2699 &   6.2950 &   7.1655 &   8.5626 &   9.5138 &  10.4300 \\
    Min                &  -5.7587 & -10.5006 & -13.8192 & -17.2995 & -19.9807 & -22.9390 & -25.8719 & -29.8960 & -32.7183 \\
    Max                &   5.8219 &   9.9048 &  13.4521 &  17.0696 &  22.3290 &  27.0048 &  31.1268 &  35.4532 &  38.3689 \\
    AR(1)              &   0.9345 &   0.9340 &   0.9329 &   0.9303 &   0.9299 &   0.9280 &   0.9284 &   0.9274 &   0.9260 \\
    Skewness           &   0.0425 &  -0.0409 &  -0.0164 &  -0.0331 &   0.0922 &   0.0854 &   0.0616 &   0.0811 &   0.0221 \\
    Kurtosis           &   4.0319 &   3.8742 &   3.7584 &   3.7014 &   3.7853 &   3.7666 &   3.6734 &   3.7679 &   3.6684 \\
    \midrule
    \multicolumn{10}{l}{\textbf{Panel B: Forward Rates ($f_t^{(n)}$, \%)}} \\
    \noalign{\vskip 2pt}
    Mean               &   5.1837 &   5.4722 &   5.7366 &   5.8398 &   6.0898 &   6.1098 &   6.3530 &   6.3936 &   6.3934 \\
    Std. Dev.          &   3.1989 &   3.0725 &   2.9736 &   2.8123 &   2.8550 &   2.7183 &   2.8797 &   2.8105 &   2.6074 \\
    Min                &   0.1132 &   0.1181 &   0.3200 &   0.4682 &   0.7231 &   0.7988 &   0.8841 &   0.8407 &   0.8904 \\
    Max                &  15.8766 &  15.2654 &  15.1938 &  14.3721 &  15.8587 &  14.7203 &  14.8801 &  15.3051 &  14.8905 \\
    AR(1)              &   0.9919 &   0.9924 &   0.9921 &   0.9922 &   0.9906 &   0.9886 &   0.9911 &   0.9895 &   0.9858 \\
    Skewness           &   0.5280 &   0.5283 &   0.6109 &   0.5771 &   0.7602 &   0.6701 &   0.5304 &   0.5457 &   0.2669 \\
    Kurtosis           &   3.0515 &   3.0301 &   3.1068 &   2.9963 &   3.3548 &   3.3462 &   2.9384 &   3.1537 &   2.8638 \\
    \midrule
    \multicolumn{10}{l}{\textbf{Panel C: YoY Change in Forward Rates ($\Delta f_t^{(n)}$, \%)}} \\
    \noalign{\vskip 2pt}
    Mean               &   0.0003 &  -0.0002 &  -0.0002 &   0.0000 &   0.0001 &  -0.0000 &  -0.0336 &  -0.0333 &  -0.0333 \\
    Std. Dev.          &   1.3710 &   1.2043 &   1.1730 &   1.0821 &   1.1224 &   1.2821 &   1.0345 &   1.2225 &   1.4812 \\
    Min                &  -4.0073 &  -3.8471 &  -4.5413 &  -4.4847 &  -5.6904 &  -6.0592 &  -3.9918 &  -6.0157 &  -8.9189 \\
    Max                &   4.5029 &   4.0090 &   4.2093 &   3.3168 &   3.8490 &   4.4468 &   4.0538 &   3.8017 &   6.5790 \\
    AR(1)              &   0.9128 &   0.9076 &   0.9001 &   0.8900 &   0.8900 &   0.8916 &   0.8525 &   0.8873 &   0.9131 \\
    Skewness           &  -0.0356 &  -0.0247 &  -0.1535 &  -0.2539 &  -0.8818 &  -0.6806 &  -0.1953 &  -0.7673 &  -0.8236 \\
    Kurtosis           &   3.7371 &   3.6964 &   4.2937 &   3.8319 &   6.5801 &   5.5653 &   4.0252 &   5.8599 &  11.6548 \\
    \bottomrule
  \end{tabular}
  \begin{tablenotes}[flushleft]
    \small
    \item \textit{Note:} This table presents the summary statistics for the yield curve variables computed from the monthly dataset.
    Panel A displays the one-year excess returns ($rx_t^{(n)} = n y_t^{(n)} - (n-1) y_{t+12}^{(n-1)} - y_t^{(1)}$) for annual maturities $n = 2, \dots, 10$.
    Panel B displays the annual forward rates ($f_t^{(n)} = n y_t^{(n)} - (n-1) y_t^{(n-1)}$).
    Panel C displays the trailing 12-month change in forward rates ($\Delta f_t^{(n)} = f_t^{(n)} - f_{t-12}^{(n)}$).
    All variables are expressed in percentage terms (\%). AR(1) represents the first-order autocorrelation coefficient.
  \end{tablenotes}
\end{table}
```

---

## 3. ADF 단위근 검정 결과 (Augmented Dickey-Fuller Test)

상수항(drift)을 포함한 ADF 모형에 대해 AIC 기준으로 최적 시차를 자동 선택해 검정한 결과입니다.
* **임계값 (Critical Values)**: 1% `-$3.4388$`, 5% `-$2.8653$`, 10% `-$2.5688$`

```latex
% Augmented Dickey-Fuller (ADF) Test Table
\begin{table}[tbp]
  \centering
  \caption{Augmented Dickey-Fuller (ADF) Unit Root Test Results}
  \label{tab:adf_test_results}
  \begin{tabular}{lccccccccc}
    \toprule
    Maturity & 2Y & 3Y & 4Y & 5Y & 6Y & 7Y & 8Y & 9Y & 10Y \\
    \midrule
    \multicolumn{10}{l}{\textbf{Panel A: Forward Rates ($f_t^{(n)}$)}} \\
    \noalign{\vskip 2pt}
    ADF Stat           &  -1.7550 &  -1.7048 &  -1.4254 &  -1.5750 &  -1.6784 &  -1.7856 &  -1.1823 &  -1.5198 &  -1.2334 \\
    p-value            &   0.4030 &   0.4286 &   0.5700 &   0.4961 &   0.4423 &   0.3876 &   0.6812 &   0.5237 &   0.6591 \\
    Lags (AIC)         &        2 &        0 &        7 &        2 &        6 &        2 &       11 &        2 &       19 \\
    Observations       &      772 &      774 &      767 &      772 &      768 &      772 &      641 &      650 &      633 \\
    \midrule
    \multicolumn{10}{l}{\textbf{Panel B: YoY Change in Forward Rates ($\Delta f_t^{(n)}$)}} \\
    \noalign{\vskip 2pt}
    ADF Stat           &  -5.9946 &  -5.7710 &  -6.1075 &  -5.8309 &  -5.9548 &  -6.1170 &  -3.9893 &  -6.1509 &  -6.3494 \\
    p-value            &   0.0000 &   0.0000 &   0.0000 &   0.0000 &   0.0000 &   0.0000 &   0.0015 &   0.0000 &   0.0000 \\
    Lags (AIC)         &       13 &       18 &       12 &       12 &       16 &       13 &       14 &       12 &       17 \\
    Observations       &      749 &      744 &      750 &      750 &      746 &      749 &      626 &      628 &      623 \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## 4. KPSS 및 Phillips-Perron (PP) 공동 검정 결과 (KPSS & PP Joint Test)

자기상관과 이분산성을 Newey-West 자동 선택 대역폭으로 통제한 KPSS(수준 정상성) 검정과 PP(상수항 포함 단위근) 검정 결과입니다.
* **KPSS 임계값 ($H_0$: 정상성)**: 1% `$0.7390$`, 5% `$0.4630$`, 10% `$0.3470$`
* **PP 임계값 ($H_0$: 단위근)**: 1% `-$3.4388$`, 5% `-$2.8653$`, 10% `-$2.5688$`

```latex
% KPSS and Phillips-Perron (PP) Joint Stationarity Test Table
\begin{table}[tbp]
  \centering
  \caption{KPSS and Phillips-Perron (PP) Stationarity Test Results}
  \label{tab:kpss_pp_results}
  \begin{tabular}{lccccccccc}
    \toprule
    Maturity & 2Y & 3Y & 4Y & 5Y & 6Y & 7Y & 8Y & 9Y & 10Y \\
    \midrule
    \multicolumn{10}{l}{\textbf{Panel A: Forward Rates ($f_t^{(n)}$)}} \\
    \noalign{\vskip 2pt}
    KPSS Stat          &   2.1323 &   2.1854 &   2.0842 &   2.0538 &   1.9235 &   1.9406 &   2.8182 &   2.7862 &   2.8193 \\
      p-value          &   0.0100 &   0.0100 &   0.0100 &   0.0100 &   0.0100 &   0.0100 &   0.0100 &   0.0100 &   0.0100 \\
      Lags (NW)        &       17 &       17 &       17 &       17 &       17 &       17 &       16 &       16 &       16 \\
    \noalign{\vskip 3pt}
    PP Stat            &  -1.7627 &  -1.5937 &  -1.5857 &  -1.5476 &  -1.6175 &  -1.8229 &  -1.2910 &  -1.5116 &  -1.8935 \\
      p-value          &   0.3991 &   0.4868 &   0.4907 &   0.5099 &   0.4740 &   0.3692 &   0.6333 &   0.5278 &   0.3351 \\
      Lags (NW)        &       21 &       21 &       21 &       21 &       21 &       21 &       20 &       20 &       20 \\
    \midrule
    \multicolumn{10}{l}{\textbf{Panel B: YoY Change in Forward Rates ($\Delta f_t^{(n)}$)}} \\
    \noalign{\vskip 2pt}
    KPSS Stat          &   0.1420 &   0.1720 &   0.1888 &   0.2226 &   0.1985 &   0.1484 &   0.2007 &   0.1403 &   0.0861 \\
      p-value          &   0.1000 &   0.1000 &   0.1000 &   0.1000 &   0.1000 &   0.1000 &   0.1000 &   0.1000 &   0.1000 \\
      Lags (NW)        &       16 &       16 &       16 &       16 &       16 &       16 &       15 &       15 &       15 \\
    \noalign{\vskip 3pt}
    PP Stat            &  -6.0303 &  -6.1026 &  -6.3389 &  -6.4693 &  -6.5764 &  -6.3385 &  -6.8875 &  -5.9156 &  -5.1980 \\
      p-value          &   0.0000 &   0.0000 &   0.0000 &   0.0000 &   0.0000 &   0.0000 &   0.0000 &   0.0000 &   0.0000 \\
      Lags (NW)        &       20 &       20 &       20 &       20 &       20 &       20 &       20 &       20 &       20 \\
    \bottomrule
  \end{tabular}
  \begin{tablenotes}[flushleft]
    \small
    \item \textit{Note:} This table reports the Kwiatkowsi-Phillips-Schmidt-Shin (KPSS) and Phillips-Perron (PP) stationarity test statistics and p-values.
    The null hypothesis ($H_0$) for the KPSS test is that the series is level-stationary, whereas the null hypothesis ($H_0$) for the PP test is that the series has a unit root.
    Panel A displays results for annual forward rates $f_t^{(n)}$, and Panel B displays results for the trailing 12-month change in forward rates $\Delta f_t^{(n)}$.
    All models include an intercept. Bandwidths/lags are selected automatically using the Newey-West (NW) automatic lag selection method.
    KPSS critical values for level-stationarity are 0.7390 (1\%), 0.4630 (5\%), and 0.3470 (10\%).
    PP critical values match the ADF critical values: $-$3.4388 (1\%), $-$2.8653 (5\%), and $-$2.5688 (10\%).
  \end{tablenotes}
\end{table}
```

---

## 5. 실증적 시계열 및 금융적 해석 요약

세 가지 시계열 분석 결과가 이론 및 이전 실증 연구들과 매우 높은 정합성을 보이고 있습니다:

### ① 선도금리 수준 ($f_t^{(n)}$) — Stochastic Trend의 존재 ($I(1)$ 비정상 시계열)
* **결과**: ADF 통계량(`-1.76 ~ -1.18`)과 PP 통계량(`-1.89 ~ -1.29`)은 모두 귀무가설(단위근 존재)을 기각하지 못했습니다. 반면, KPSS 검정(`1.92 ~ 2.82`)은 귀무가설(정상 시계열)을 **1% 유의수준에서 강력히 기각**했습니다.
* **해석**: 선도금리 수준(Level) 변수는 강한 지속성(Persistence)을 띄며 단위근을 포함한 **비정상성 시계열**입니다. 따라서 수준 변수간 단순 선형 회귀는 **가짜 회귀(Spurious Regression)** 오류를 범할 가능성이 매우 큽니다.

### ② 선도금리 12개월 변화량 ($\Delta f_t^{(n)}$) — Mean-Reversion 입증 ($I(0)$ 정상 시계열)
* **결과**: ADF(`-6.35 ~ -3.99`)와 PP(`-6.89 ~ -5.19`) 검정은 1% 수준에서 단위근을 강력 기각하였으며, KPSS(`0.086 ~ 0.223`) 검정은 정상성 귀무가설을 기각하지 못했습니다.
* **해석**: 이자율 시계열의 12개월 변화량은 stochastic trend가 성공적으로 소거된 **평균회귀적 정상 시계열**입니다. Cochrane-Piazzesi(2005)와 같이 미래 채권 수익률 예측 모형을 만들거나 다변량 VAR 모형을 구축할 때, 이 정상 차분 변수를 모델링하는 것이 계량경제학적으로 완벽히 타당합니다.

### ③ 꼬리 위험과 Kurtosis 특징
* **특이점**: 12개월 선도금리 변화량의 왜도(Skewness)는 음의 방향(`-0.82 ~ -0.03`)으로 기울어져 있어 하락 Shock가 더 자주 강하게 발생함을 뜻합니다. 특히, **10년 만기 선도금리 변화량의 첨도(Kurtosis)는 11.65**로 고성격의 **Fat-tail(두터운 꼬리)** 성격을 보여줍니다. 이는 금융 시장 위기 등에서 발생하는 급격한 금리 변동 쇼크가 정규분포 예측 대비 극도로 빈번하게 일어남을 보여주는 실증적 증거입니다.

---

## 6. MATLAB 스크립트 실행 가이드

각 스크립트는 워크스페이스에 저장되어 있으며, MATLAB 명령창에 다음 한 줄을 입력하면 테이블이 파일로 출력되고 화면에 렌더링됩니다:

```matlab
% 1. 요약 통계량 계산 및 summary_table.tex 출력
generate_summary_latex();

% 2. ADF 검정 계산 및 adf_table.tex 출력
generate_adf_latex();

% 3. KPSS & PP 검정 계산 및 kpss_pp_table.tex 출력
generate_kpss_pp_latex();
```
