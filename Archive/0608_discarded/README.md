# 0608 엔진 (폐기됨)

**폐기일:** 2026-06-13
**사유:** 성과 압박 하에 급조된 하이퍼파라미터 탐색 파이프라인. 이론적 결함으로 폐기.

## 무엇이었나

베이스라인 엔진(`../../Engine.py`, `../../NNBib.py`) 위에 만기별 가중손실,
조기중단(early-abort), 대규모 하이퍼파라미터 탐색을 얹은 개량판.

| 파일 | 내용 |
|---|---|
| `Engine_0608.py` | `Engine.py` + early-abort 로직 (`EarlyAborted`, `EarlyAbortStep`) |
| `NNBib_0608.py` | `NNBib.py` + `target_weights`(만기별 가중손실), `use_bias`, `regularize_output` |
| `config_0608.py` | 선형 NN(`archi=[]`) + output 정규화 설정 |
| `run_0608_nmc100_navg10_search.py` | nmc=100/navg=10 대규모 후보 탐색 드라이버 |
| `nmc100_navg10_search_0608/` | 위 탐색의 산출물 (.mat 후보 + 로그 + summary.csv) |

## 왜 버렸나 — 이론적 결함

핵심은 **탐색 레이어가 실현 OOS R²로 후보를 선별**한다는 점이다.

`Engine_0608.run_oos_forecast`는 평가구간의 실현수익률로 계산한 `r2_now`가
임계치 아래면 후보를 폐기(early-abort)하고, `run_0608_..._search.py`는 이
실현 OOS R²로 수십 개 후보를 가지치기·선별한다. 그 결과:

1. **테스트셋 누수 (data snooping).** 모델 선택 기준이 학습/검증 데이터가
   아니라 평가구간의 실현 R²다. 보고된 R²OOS는 상향 편향되고,
   Clark-West p-value의 귀무분포 가정이 깨져 **무효**가 된다.
2. **Pseudo-real-time 전제 위반.** 실시간 예측자는 시점 t에서 전체표본 R²를
   알 수 없다. 전체 OOS R²로 abort/선택하는 순간 OOS 실험의 의미가 사라진다.

엔진의 *예측 메커니즘*(embargo expanding window, 시드 앙상블, 검증기반
하이퍼파라미터 선택)은 건전했다. 결함은 이 탐색/조기중단 레이어에 있었다.

## 정상 대안

`../../Engine.py` + `../../NNBib.py` + `../../config*.py` 베이스라인을 사용할 것.
후보 선택은 OOS R²가 아니라 **검증손실/정보기준**으로 하고, 최종 보고 모델은
사전에 고정한다.

## 복구

활성 트리(`NN/`)에서 이 모듈들을 import하는 파일은 없으므로 폐기해도
베이스라인 파이프라인은 그대로 동작한다. 되살리려면 이 폴더의 파일들을
`NN/`로 다시 옮기면 된다.
