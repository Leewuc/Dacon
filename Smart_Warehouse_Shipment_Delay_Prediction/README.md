# DACON 창고 출하 지연 예측

DACON 대회 — 창고 로봇 운영 데이터로 향후 30분 평균 지연 시간 예측

| 항목 | 값 |
| ---- | -- |
| 평가 지표 | MAE (낮을수록 좋음) |
| 최종 점수 | **9.9866** |
| 1위 점수 | 9.69 |
| 환경 | Python 3.10, LightGBM, CatBoost, XGBoost |

> 이 저장소는 주요 코드만 추려 정리한 포트폴리오 버전입니다.

---

## 문제 정의

시나리오 × 레이아웃 × 시간 스텝(25 step) 구조. 각 시나리오는 특정 레이아웃에서 25개 시점 데이터를 가지며, 마지막 시점의 `avg_delay_minutes_next_30m` 예측.

주요 난점:

- Train/Test 레이아웃 불일치 → 분포 이동(covariate shift)
- 오른쪽 치우친 target 분포 → log1p transform 필요
- 25 step 시계열 내 형태(shape) 정보가 예측에 유효

---

## 최종 모델: 3-Pass Pseudo-label Pipeline

`ensemble/lead_norm_ensemble.py` 기반. 최고 LB: **9.9866**

```text
[Pseudo-label]
  temporal pseudo-label (iter5, bandwidth=0.20, catboost_weight=1.0)
    ↓
[Base Features: 567개]
  base(517) + lead_norm(35) + temporal_dynamics(15)
    ↓
[Pass 1]  log1p LGBM  ← train + test(pseudo)
  → pred1
    ↓
[Pass 2]  + 4 scenario features from pred1
            (scen_mean, scen_std, scen_max, vs_mean)
  → pred2
    ↓
[Pass 3]  Pass1 scenario features → Pass2 scenario features로 교체
  → pred3  (최종 LGBM 예측)
    ↓
[Ensemble]
  86% LGBM(pred3) + 14% CatBoost + layout/tail-XGB routing
```

### LGBM 파라미터

```python
objective      = "mae"
n_estimators   = 1200
learning_rate  = 0.03
num_leaves     = 63
subsample      = 0.65
seed           = 42
```

---

## 피처 엔지니어링

### Lead-Norm Features — `features/temporal_dynamics_ensemble.py` + `ensemble/lead_norm_ensemble.py`

원시 lead 사용 시 train/test 비율 차이(congestion ratio=1.27)로 외삽 문제 → **시나리오 평균으로 정규화**.

```text
lead1_rel       = lead1 / scenario_mean   ← 상대적 미래 수준
lead_diff1      = lead1 - current         ← 미래 변화량
future_mean_rel = future_mean / scenario_mean
```

### Temporal Dynamics — `features/temporal_dynamics_ensemble.py`

절대값이 아닌 **형태(shape)** 측정 → 분포 이동에 강건:

- slope (추세), CV (변동계수), peak timing, early-vs-late ratio

### Scenario Aggregation — `features/scenario_agg_features.py`

Pass 1/2 예측값 기반 시나리오 수준 집계 (타겟 누수 없이):
`scen_mean`, `scen_std`, `scen_max`, `vs_mean`

### Queueing Theory — `features/queuing_theory_features.py`

M/M/c 모델 기반 비선형 포화 지표:

- `mm1_queue_length(ρ) = ρ² / (1 - ρ)`
- `mm1_near_saturation(ρ) = ρ / (1 - ρ)`

---

## LB 점수 히스토리

| 모델 | LB MAE | 비고 |
| ---- | ------ | ---- |
| iter5 temporal pseudo | 10.0681 | |
| log1p + lead_norm + temporal | 10.0305 | |
| 2-pass (scenario feature) | 10.0179 | |
| **3-pass (iter5 pseudo, seed=42)** | **9.9866** | ★ 최고 |
| 4-pass | 10.0044 | 과적합 |
| 3-pass richscen (9 features) | 10.0155 | 과적합 |
| 3-pass v2 (3-pass pseudo 재사용) | 10.0762 | 발산 |
| target_leads | 10.0809 | 역효과 |

---

## 핵심 인사이트

1. **Pseudo-label 반복 → 항상 발산**: 자기 예측을 pseudo-label로 재사용하면 무조건 악화
2. **3-pass가 최적**: Pass 4부터 과적합, Pass 2보다 낫지만 Pass 4는 퇴보
3. **Scenario features는 4개가 최적**: 9개로 늘리면 과적합
4. **Target lead feature는 역효과**: train=실제값 / test=pseudo-label 품질 차이 → 분포 불일치
5. **Log1p transform 필수**: 오른쪽 치우친 target에 매우 효과적
6. **Multi-seed 앙상블 효과 미미**: 예측 상관관계 ρ=0.988 → 분산 감소 없음
7. **Lead 정규화 필수**: 원시 lead 사용 시 congestion_lead1 비율=1.27로 외삽 문제

---

## XAI 분석

`xai/generate_xai_report.py` 실행 결과 (holdout split 기준, 자세한 내용은 `xai/` 참고).

### 피처 중요도 — Permutation (MAE Drop)

| 피처 | Drop |
| ---- | ---- |
| congestion_score_exp_mean | 2.644 |
| low_battery_ratio_exp_std | 0.529 |
| pack_utilization_exp_mean | 0.477 |
| max_zone_density | 0.176 |

### 피처 패밀리별 Gain

| 패밀리 | Gain |
| ------ | ---- |
| traffic_and_congestion | 1,750,461 |
| sequence_history | 490,545 |
| order_and_pick_workload | 253,265 |
| battery_and_charging | 175,401 |
| layout_structure | 160,383 |

---

## 실험 흐름

| 기간 | 주요 실험 |
| ---- | -------- |
| 3월 말 ~ 4월 초 | 베이스라인, lag/rolling features, layout features, 튜닝 |
| 4월 초 ~ 중순 | layout specialist, congestion tail expert, pseudo-label v1~v3, adversarial validation |
| 4월 중순 ~ 말 | LSTM, distribution alignment, covariate shift ensemble, conditional gating |
| 4월 말 | lead features, temporal dynamics, **3-pass pipeline** (최고 달성) |
| 5월 초 | OOF 4-pass 실험, 최종 블렌딩 탐색 |

---

## 파일 구조

```text
dacon-warehouse-delay/
├── README.md
├── .gitignore
│
├── baseline/
│   └── pipeline_baseline.py          # LGBM 베이스라인 (lag features)
│
├── features/
│   ├── queuing_theory_features.py    # M/M/c 대기열 이론 피처
│   ├── scenario_agg_features.py      # 시나리오 집계 피처 (pass2/3 사용)
│   └── temporal_dynamics_ensemble.py # 시간적 동역학 shape 피처
│
├── models/
│   ├── catboost_categorical_experiments.py  # CatBoost 범주형
│   ├── xgboost_experiments.py               # XGBoost
│   ├── lstm_sequence_model.py               # LSTM 시퀀스 모델
│   ├── layout_specialist_experiments.py     # 레이아웃 전문화 모델
│   ├── pseudo_label_v4_weighted.py          # 가중 pseudo-label (도메인 어댑테이션)
│   └── adversarial_validation_cv.py         # 적대적 검증 (covariate shift 분석)
│
├── ensemble/
│   └── lead_norm_ensemble.py         # ★ 최고 모델 (3-pass, LB 9.9866)
│
├── blend/
│   └── blend_search.py               # 제출 블렌딩 가중치 탐색
│
└── xai/
    ├── generate_xai_report.py        # XAI 리포트 생성 스크립트
    ├── insight_summary.md            # 피처 중요도 분석 결과
    └── model_compare_summary.md      # LGBM vs CatBoost 비교
```

---

## 데이터

`train.csv`, `test.csv`, `sample_submission.csv`는 [DACON 대회 페이지](https://dacon.io)에서 다운로드.
