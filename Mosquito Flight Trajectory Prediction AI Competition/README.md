# DACON 모기 비행 궤적 예측 대회 솔루션

**최종 LB: 0.6906** | 평가: R-Hit@1cm (3D 유클리드 거리 ≤ 0.01m 비율)

---

## 문제 개요

- **입력**: 모기 3D 위치 (x,y,z) 11타임스텝 × 40ms 간격 (−400ms ~ 0ms)
- **예측**: +80ms 후 위치
- **훈련/테스트**: 각 10,000개 샘플, 별도 CSV

---

## 핵심 방법론

### 1. Azimuth 정규화
마지막 속도의 XY 성분을 +X 방향으로 회전 후, 마지막 위치를 원점으로 이동.
모델이 방향에 무관하게 궤적 패턴을 학습할 수 있게 함.

```python
vxy = X[:, -1, :2] - X[:, -2, :2]
ca, sa = vxy / |vxy|  # cos, sin of last heading
Xn = rotate(X - last, ca, sa)  # XY rotation only, Z unchanged
```

### 2. 잔차 예측 (Residual Learning)
등속도 외삽(cv = last_vel × 2)의 잔차만 학습.
타겟 분산이 ~4.8cm → 수 mm 수준으로 감소.

```python
# Phase 1: predict (y_norm - cv_norm)
loss = F.l1_loss(model(X), y_norm - cv_norm)
# Inference: pred = model(X) + cv_norm
```

### 3. MAE Pretraining (핵심 돌파구, +1.25%p)
Phase 1에서 MSE 대신 **L1(MAE)** 손실 사용.
L1이 다른 local optima를 찾아 더 높은 OOF + Phase 2 시작점 개선.

### 4. Smooth R-Hit Loss (Phase 2)
tau를 점진적으로 줄이며 1cm 경계에 집중.

```python
loss = -sigmoid((0.01 - dist) / tau).mean()
# tau: 0.003 → 0.0005 over 40 epochs
```

### 5. y축 반전 Augmentation + TTA
정규화 프레임에서 좌우 대칭. 학습 데이터 2배 + 추론 시 평균.

```python
MF = [1, -1, 1, 1, -1, 1]  # y, vy 부호 반전
pred = (model(X) + model(flip_y(X)) * [-1,1,1]) / 2
```

### 6. Pseudo-labeling
테스트 데이터에 구모델 예측값을 pseudo-label로 사용 (PSEUDO_W=0.5).
`pseudo = (v5_predictions + v9_predictions) / 2`

---

## 아키텍처

```
입력: (B, 11, 6) — [x,y,z, vx,vy,vz] azimuth 정규화
stem:   Conv1d(6→128, k=1) + BN + GELU
blocks: 6× ResBlock (Conv1d(128,3,pad=1) + BN + GELU) × 2 + skip
pool:   GlobalAvgPool + GlobalMaxPool → concat (B, 256)
head:   Linear(256→256) → GELU → Dropout(0.3) → Linear(256→64) → GELU → Linear(64→3)
```

**학습 설정**
- Phase 1: MAE, lr=1e-3, CosineAnnealing, 80 epoch
- Phase 2: Smooth R-Hit, lr=1e-4, CosineAnnealing, 40 epoch
- AdamW, weight_decay=1e-4, batch=256
- 10-fold CV, 5 seeds [42, 7, 2025, 13, 99] → 50 모델 앙상블
- 정규화 공간에서 평균 후 역변환

---

## 제출 이력 (주요)

| 설명 | OOF | LB |
|------|-----|----|
| CNN v5 잔차예측 (MSE, 3-seed) | 66.47% | 0.6854 |
| CNN v11 pseudo-label (MSE, 5-seed 10-fold) | 67.77% | 0.6890 |
| v17(K=4) 20% + v11 80% | 67.92% | 0.6896 |
| **CNN v27 MAE pretrain (5-seed 10-fold)** | **68.02%** | **0.6904** |
| v27_10s 25% + v22 30% + v11 45% | 68.16% | **0.6906** |

---

## 효과 없던 것들

| 방법 | 이유 |
|------|------|
| Transformer / BiLSTM / GRU | 10K 데이터에서 CNN보다 약함 |
| 9ch/10ch 피처 (가속도 등) | CNN이 이미 암묵적으로 활용, 노이즈만 추가 |
| 가우시안 노이즈 augmentation | 1cm 기준이 너무 tight |
| WTA multi-hypothesis (K=5) | OOF 낮고 full run에서 다양성 붕괴 |
| Phase 2 pseudo를 v27로 교체 | r_y=0.987, 과적합 후 LB 0.6874 |
| 3D 회전 정규화 | OOF -3.6%p, 2D azimuth가 최적 |
| Stacking / GBDT | OOF 57-62%, 블렌딩 불가 |
| ep1=120 (긴 MAE) | MSE 과최적화 후 R-Hit 전환 방해 |

---

## 데이터 분석 요약

- 오차 주원인: **y(횡방향) 46.6%**, z(수직) 14.7%, x(전진) 13.8%
- 급격한 방향 전환(yaw > 30°): hit rate 34.7% (평균 68.2% 대비 −33%p)
- 방향 완벽 예측 시 이론 상한: **~71.2%**
- 현재 gap: ~2.2%p — 급격한 방향 전환이 본질적 예측 불가 영역

---

## 핵심 코드

`train.py` 참고
