# Skills_advanced_analysis.md — 고급 투자 분석 규칙 정의

이 문서는 기본 6-Skills 분석을 넘어선 **고급 분석 모듈**의 계산 규칙, 시각화 기준, 해석 가이드를 정의한다.

---

## 1. Fama-French Factor Attribution (팩터 귀인 분석)

### 1.1 목적
포트폴리오 수익률을 **시장(Market), 사이즈(SMB), 가치(HML), 알파(Alpha)** 4가지 요인으로 분해하여,
"수익이 어디서 왔는가?"를 정량적으로 규명한다.

### 1.2 회귀 모델

```
R_p(t) - R_f(t) = α + β_mkt·(R_mkt - R_f) + β_smb·SMB(t) + β_hml·HML(t) + ε(t)
```

| 변수 | 의미 | 데이터 소스 |
|------|------|-------------|
| R_p | 포트폴리오 일간 수익률 | 사용자 입력 |
| R_f | 무위험 수익률 (일간) | Kenneth French Library or 연 3.5%/252 |
| R_mkt - R_f | 시장 초과수익률 | Kenneth French Library |
| SMB | Small Minus Big (소형주 프리미엄) | Kenneth French Library |
| HML | High Minus Low (가치주 프리미엄) | Kenneth French Library |
| α | Jensen's Alpha (팩터 설명 불가 초과수익) | 회귀 절편 |

### 1.3 OLS 회귀 구현 규칙

```python
# 의존성: numpy만 사용 (sklearn 금지 — 배포 경량화)
# 정규방정식: β = (X^T X)^(-1) X^T y
X = [ones, mkt_rf, smb, hml]   # (T × 4)
y = portfolio_excess_returns     # (T × 1)
beta = np.linalg.lstsq(X, y, rcond=None)[0]

alpha = beta[0]           # 일간 알파
beta_market = beta[1]
beta_smb = beta[2]
beta_hml = beta[3]

# 연환산
alpha_annual = alpha * 252
residual_std = std(y - X @ beta) * sqrt(252)

# 알파 유의성 (t-statistic)
se = residual_std / sqrt(T) / sqrt(252)
t_stat = alpha / se  # |t| > 2.0이면 95% 유의

# R² = 1 - SS_res / SS_tot
r_squared = 1 - sum(residuals^2) / sum((y - mean(y))^2)
```

### 1.4 수익률 기여도 분해 규칙

```python
# 각 팩터의 수익률 기여 (연환산)
contrib_market = beta_market * mean(mkt_rf) * 252
contrib_smb = beta_smb * mean(smb) * 252
contrib_hml = beta_hml * mean(hml) * 252
contrib_alpha = alpha * 252

# 퍼센트 분해 (절대값 기준)
total_abs = |contrib_market| + |contrib_smb| + |contrib_hml| + |contrib_alpha|
pct_market = |contrib_market| / total_abs * 100
pct_smb = |contrib_smb| / total_abs * 100
pct_hml = |contrib_hml| / total_abs * 100
pct_alpha = |contrib_alpha| / total_abs * 100
```

### 1.5 시각화 규칙

| 차트 | 타입 | 용도 |
|------|------|------|
| 팩터 기여 도넛 | `go.Pie(hole=0.5)` | 4팩터 비중 한눈에 파악 |
| 팩터 노출 바 | `go.Bar` + 0선 | Beta 크기/방향 비교 |
| 누적 기여 라인 | `go.Scatter(stackgroup)` | 시간별 팩터 기여 누적 |
| 알파 유의성 게이지 | `go.Indicator(mode="gauge")` | t-stat → 유의성 직관적 표현 |

### 1.6 해석 가이드 규칙

| 조건 | 진단 | 설명 |
|------|------|------|
| β_mkt > 1.0 | "공격적 포트폴리오" | 시장보다 높은 변동성 |
| β_mkt < 0.8 | "방어적 포트폴리오" | 시장 하락 시 방어 효과 |
| |t_alpha| > 2.0 | "통계적으로 유의한 알파" | 팩터로 설명 안 되는 진짜 초과수익 |
| R² > 0.8 | "시장 추종형" | 대부분 시장 요인에 의한 수익 |
| R² < 0.3 | "독자적 전략" | 시장과 독립적인 수익원 |

### 1.7 오프라인 대응 규칙
Kenneth French Library 접속 실패 시:
- 합성 팩터 데이터 자동 생성
- `mkt_rf ~ N(0.04%, 0.95%)`, `smb ~ N(0.01%, 0.5%)`, `hml ~ N(0.01%, 0.5%)`
- 포트폴리오 수익률과 0.6~0.8 상관관계 유지

---

## 2. Efficient Frontier (효율적 프론티어)

### 2.1 목적
Markowitz 평균-분산 최적화 프레임워크에서 현재 포트폴리오의 **효율성**을 평가한다.
"같은 리스크에서 더 높은 수익을, 같은 수익에서 더 낮은 리스크를 달성할 수 있는가?"

### 2.2 계산 규칙

```python
# 입력
prices: DataFrame (T × N)  # N개 종목의 일별 가격
weights: Dict[str, float]   # 현재 비중

# Step 1: 수익률 통계
daily_returns = prices.pct_change().dropna()
mu = daily_returns.mean() * 252          # 연환산 기대수익률 (N × 1)
cov = daily_returns.cov() * 252          # 연환산 공분산 행렬 (N × N)

# Step 2: 랜덤 포트폴리오 샘플링 (N_sim = 3,000~10,000)
for i in range(N_sim):
    w = np.random.dirichlet(np.ones(N))  # Dirichlet → 비중 합 = 1, 양수
    ret = w @ mu                          # 포트폴리오 기대수익률
    vol = sqrt(w @ cov @ w)              # 포트폴리오 변동성
    sharpe = (ret - rf) / vol            # 샤프비율

# Step 3: 최적 포트폴리오 식별
min_variance = argmin(volatility)
max_sharpe = argmax(sharpe_ratio)

# Step 4: 현재 포트폴리오 위치
current_ret = w_current @ mu
current_vol = sqrt(w_current @ cov @ w_current)
current_sharpe = (current_ret - rf) / current_vol
```

### 2.3 시각화 규칙

**효율적 프론티어 산점도:**
- X축: 연환산 변동성 (%), Y축: 연환산 기대수익률 (%)
- 색상: Sharpe Ratio (Viridis colorscale)
- 마커: ★ 최소분산 (yellow), ★ 최대샤프 (green), ◆ 현재 (red, 크게)
- 사이즈: Sharpe 비례 (2~8px)

**비중 비교 바:**
- 3그룹: 현재/최소분산/최대샤프
- Grouped bar, 종목별

### 2.4 해석 가이드

| 조건 | 진단 |
|------|------|
| 현재 Sharpe ≈ 최대 Sharpe (gap < 0.1) | "효율적 프론티어 근처에 위치 → 잘 최적화됨" |
| gap > 0.3 | "최적 대비 비효율적 → 비중 조정 권장" |
| 현재 vol > 최소분산 vol × 1.5 | "불필요한 리스크를 감수 중 → 방어적 재배분 검토" |

---

## 3. Stress Test (역사적 위기 시뮬레이션)

### 3.1 목적
포트폴리오가 과거 금융 위기를 **실시간으로 겪었다면** 어떤 손실을 입었을지 추정한다.

### 3.2 내장 시나리오

| 시나리오 | 시장 하락 | 기간 | 회복 기간 |
|----------|----------|------|----------|
| COVID-19 (2020) | -34% | 23거래일 | 148거래일 |
| 글로벌 금융위기 (2008) | -57% | 352거래일 | 1,020거래일 |
| 금리인상 충격 (2022) | -25% | 194거래일 | 310거래일 |
| 닷컴 버블 (2000) | -49% | 650거래일 | 1,730거래일 |

### 3.3 섹터 충격 전파 규칙

```python
# 각 시나리오에 섹터별 impact_multiplier 정의
# multiplier > 1.0: 시장보다 더 큰 타격 (예: 금융 in 2008)
# multiplier < 1.0: 시장보다 방어적 (예: 헬스케어 in COVID)

portfolio_loss = market_decline × Σ(weight_i × sector_impact_i)

# 일별 경로: GBM (Geometric Brownian Motion)
drift = portfolio_loss / duration_days
vol = |portfolio_loss| / sqrt(duration_days)
daily_returns ~ N(drift, vol)
```

### 3.4 시각화 규칙

| 차트 | 타입 | 특이사항 |
|------|------|----------|
| 시나리오 비교 바 | 가로 Bar | 포트폴리오 vs 시장 이중 바 |
| 일별 경로 | Line + Area | 최저점 마커, 100 기준선 |
| 섹터 영향 트리맵 | Treemap | 크기=비중, 색상=손실 강도 |
| 회복 타임라인 | Stacked Bar | 위기+회복 기간 스택 |

---

## 4. Geopolitical Scenario Engine (지정학 시나리오)

### 4.1 목적
**지정학적 위기 → 매크로 충격 → 섹터 전파 → 포트폴리오 영향**의 인과 체인을 시뮬레이션한다.
사용자가 직접 매크로 변수를 조합하여 커스텀 시나리오도 설계 가능.

### 4.2 매크로 충격 변수

| 변수 | 설명 | 범위 | 단위 |
|------|------|------|------|
| oil_price | 유가 (Brent) 변동 | -50 ~ +150 | % |
| interest_rate | 기준금리 변동 | -200 ~ +500 | bp |
| usd_krw | 원/달러 환율 변동 | -20 ~ +40 | % |
| supply_chain | 공급망 스트레스 지수 | 0 ~ 100 | index |
| geopolitical_risk | 지정학 리스크 프리미엄 | 0 ~ 100 | index |
| recession | 경기침체 강도 | 0 ~ 100 | index |

### 4.3 섹터 탄력성 매트릭스 (핵심 규칙)

각 매크로 변수에 대한 섹터별 **탄력성**(sensitivity)을 정의한다:
- **양수**: 같은 방향 (유가 상승 → 에너지주 수혜)
- **음수**: 반대 방향 (유가 상승 → 항공주 타격)

```python
# 예: 유가(oil_price) 탄력성
oil_sensitivity = {
    "에너지": +0.45,    # 유가와 강한 양의 상관
    "항공": -0.55,      # 유가 상승 → 유류비 증가 → 실적 악화
    "방산": +0.20,      # 지정학 긴장 동반 시 수혜
    "2차전지": +0.15,   # 대체에너지 수혜 기대
    "여행/레저": -0.35, # 운송비 + 소비 위축
    "화학": -0.25,      # 원자재 비용 증가
    ...
}
```

### 4.4 충격 전파 공식

```python
# Step 1: 매크로 충격 정규화
normalized_shock = shock_value / (range_max - range_min) × 2

# Step 2: 섹터별 총 영향
sector_impact = Σ(normalized_shock_i × elasticity_i,sector) × 100  # percent

# Step 3: 포트폴리오 영향
portfolio_impact = Σ(weight_ticker × sector_impact_of_ticker)

# Step 4: 일별 경로 (GBM with shock decay)
shock_decay = exp(-t / (T × 0.3))  # 초반 충격 후 점진 완화
daily_returns = N(drift, vol) × (0.3 + 0.7 × shock_decay)
```

### 4.5 내장 시나리오

| 시나리오 | 유가 | 금리 | 환율 | 공급망 | 지정학 | 경기침체 |
|----------|------|------|------|--------|--------|----------|
| 🇮🇷 호르무즈 봉쇄 | +80% | +50bp | +12% | 60 | 85 | 25 |
| 🇹🇼 대만해협 위기 | +35% | -25bp | +18% | 95 | 95 | 50 |
| 🇺🇦 NATO 확전 | +60% | -50bp | +15% | 70 | 90 | 45 |
| 🇰🇵 한반도 긴장 | +15% | -25bp | +10% | 30 | 75 | 15 |
| 💹 AI 버블 붕괴 | -15% | -75bp | +5% | 15 | 20 | 40 |
| 🦠 팬데믹 2.0 | -30% | -100bp | +8% | 80 | 40 | 65 |

### 4.6 시각화 규칙

| 차트 | 타입 | 용도 |
|------|------|------|
| 매크로 → 포트폴리오 워터폴 | `go.Waterfall` | 각 매크로 변수의 기여도 누적 |
| 섹터별 영향 바 | 수평 `go.Bar` | 빨강(타격)/초록(수혜) 바 |
| 30일 경로 | `go.Scatter` | 시나리오 발생 후 가치 추이 |
| 시나리오 비교 레이더 | `go.Scatterpolar` | 여러 시나리오의 충격 강도 비교 |
| 시나리오×섹터 히트맵 | `go.Heatmap` | 전체 시나리오-섹터 영향 한눈에 |

---

## 5. Correlation Network (상관관계 네트워크)

### 5.1 목적
종목 간 상관관계를 **네트워크 그래프**로 시각화하여 분산투자 효과를 직관적으로 파악한다.
높은 상관관계를 가진 종목 **클러스터**를 자동 탐지한다.

### 5.2 상관행렬 계산 규칙

```python
# 일간 수익률 기반 피어슨 상관계수
returns = prices.pct_change().dropna()
corr_matrix = returns.corr()

# 엣지 필터링: |correlation| > threshold (기본 0.3)
# 사용자가 슬라이더로 0.1 ~ 0.9 조절 가능
edges = [(i, j, corr) for i, j in pairs if |corr[i][j]| > threshold]
```

### 5.3 네트워크 레이아웃 규칙

```python
# Fruchterman-Reingold Force-Directed Layout
# 외부 라이브러리(networkx) 없이 순수 구현

repulsive_force = k² / distance      # 모든 노드 쌍
attractive_force = d² / k × |corr|   # 연결된 노드 쌍만
temperature *= 0.95                   # 매 반복마다 냉각
iterations = 150
```

### 5.4 클러스터링 규칙

```python
# Greedy agglomerative: 평균 상호 상관 > threshold인 노드들을 그룹화
# threshold = 0.5 (기본값)
# 결과: {cluster_id: [ticker1, ticker2, ...]}
```

### 5.5 시각화 규칙

| 요소 | 시각적 속성 | 인코딩 |
|------|------------|--------|
| 노드 크기 | 15~50px | 포트폴리오 비중 비례 |
| 노드 색상 | 클러스터별 팔레트 | 클러스터 멤버십 |
| 엣지 색상 | 초록/빨강 | 양/음의 상관 |
| 엣지 두께 | 0.5~4px | |상관계수| 비례 |
| 노드 라벨 | 종목 코드 | 항상 표시 |
| Hover | 종목명, 섹터, 비중, Top 3 상관 종목 | 상세 정보 |

### 5.6 보조 차트: 상관관계 히트맵

- Diverging colorscale: Red(-1) → White(0) → Blue(+1)
- 셀 내 수치 표시 (소수점 2자리)
- 대각선은 항상 1.0

### 5.7 분산투자 인사이트 규칙

| 조건 | 진단 |
|------|------|
| avg_corr > 0.6 | "종목들이 비슷하게 움직임 → 분산 효과 제한적" |
| avg_corr < 0.3 | "좋은 분산 효과" |
| 클러스터 ≤ 2개 (종목 5+ 이상) | "대부분 동일 클러스터 → 다른 섹터 편입 권장" |
| 음의 상관 엣지 존재 | "자연적 헤지 효과 발견 → 하락 방어에 유리" |

---

## 6. Multi-Portfolio Comparison (포트폴리오 비교)

### 6.1 목적
현재 포트폴리오를 **벤치마크 포트폴리오**(S&P 500, 글로벌 분산, 한국 대표, 성장주, 배당형)와
나란히 비교하여 상대적 강점/약점을 파악한다.

### 6.2 비교 지표

| 지표 | 계산식 | 비교 기준 |
|------|--------|----------|
| 총 수익률 | `(1+r).prod() - 1` | 높을수록 좋음 |
| Sharpe Ratio | `mean_excess / std × √252` | 높을수록 좋음 |
| Max Drawdown | `max(peak - trough) / peak` | 작을수록 좋음 (절대값) |
| 연환산 변동성 | `std × √252` | 낮을수록 안정적 |
| Overall Skill | 6-Skills 평균 | 높을수록 좋음 |
| 강점/약점 Skill | max/min of 6 Skills | 비교 포인트 |

### 6.3 시각화 규칙

| 차트 | 최대 비교 수 | 색상 할당 |
|------|-------------|----------|
| 오버레이 레이더 | 5개 | Indigo, Emerald, Amber, Red, Violet |
| 그룹 바 | 5개 | 동일 |
| 비교 테이블 | 무제한 | 조건부 서식 (최고=볼드) |

---

## 7. VaR & CVaR 계산 규칙

### 7.1 Value at Risk (VaR)

```python
# Historical Simulation 방식
# 95% VaR: 과거 수익률 하위 5%ile에 해당하는 손실
var_95 = np.percentile(simulated_final_values, 5)
var_99 = np.percentile(simulated_final_values, 1)

# 금액 기준
loss_95 = initial_value - var_95
```

### 7.2 Conditional VaR (Expected Shortfall)

```python
# VaR를 초과하는 손실의 평균
cvar_95 = mean(values[values <= var_95])
# "최악의 5% 시나리오에서 평균적으로 얼마를 잃는가?"
```

### 7.3 표시 규칙
- 금액 단위: 초기 투자금 $10,000 기준
- 소수점: 달러 단위 정수, 퍼센트 소수점 1자리
- 색상: 빨강 (손실 강조)

---

## 8. GARCH(1,1) 변동성 예측

### 8.1 목적
과거 수익률의 **변동성 클러스터링**(큰 변동 뒤에 큰 변동이 이어지는 현상)을 모델링하여,
미래 변동성을 예측하고 현재 변동성 국면(높음/보통/낮음)을 진단한다.

### 8.2 GARCH(1,1) 모델

```
σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
```

| 파라미터 | 의미 | 제약 조건 |
|----------|------|-----------|
| ω (omega) | 기저 분산 | > 0 |
| α (alpha) | 직전 충격 반응 | > 0 |
| β (beta) | 변동성 지속 | > 0 |
| α + β | 지속성 (persistence) | < 1 (안정 조건) |

### 8.3 파라미터 추정 규칙

```python
# MLE (Maximum Likelihood Estimation) via scipy.optimize.minimize
# 목적함수: -log-likelihood = 0.5 * Σ[log(σ²_t) + ε²_t / σ²_t]

# 초기값: ω=1e-6, α=0.08, β=0.88
# 경계: ω∈(1e-10, 1e-3), α∈(0.01, 0.5), β∈(0.3, 0.99)
# 제약: α + β < 0.999

# MLE 실패 시 → 그리드 서치 폴백
alpha_grid = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
beta_grid = [0.75, 0.80, 0.85, 0.88, 0.90, 0.92]
# 각 조합에서 log-likelihood 최대 선택
```

### 8.4 예측 공식

```python
# 장기 균형 변동성 (Long-run Volatility)
sigma_LR = sqrt(omega / (1 - alpha - beta)) * sqrt(252)

# 다단계 예측 (h일 후)
sigma2_h = omega * sum((alpha+beta)**i for i in range(h)) + (alpha+beta)**h * sigma2_t

# 반감기 (Volatility Half-life)
half_life = log(0.5) / log(alpha + beta)
```

### 8.5 변동성 국면 분류

| 조건 | 국면 | 의미 |
|------|------|------|
| σ_1d > 장기평균 × 1.5 | 🔴 HIGH | 변동성 급등 구간 |
| σ_1d > 장기평균 × 0.8 | 🟡 MEDIUM | 정상 범위 |
| σ_1d ≤ 장기평균 × 0.8 | 🟢 LOW | 안정 구간 |

### 8.6 시각화 규칙

| 차트 | 타입 | 용도 |
|------|------|------|
| 조건부 변동성 + 수익률 | `go.Scatter` 이중 축 | 변동성 클러스터링 확인 |
| 변동성 예측 퍼널 | `go.Funnel` | 30/60/90일 예측 비교 |
| GARCH 파라미터 도넛 | `go.Pie(hole=0.5)` | ω/α/β 비중 시각화 |

### 8.7 최소 데이터 요건
- 최소 50 거래일 이상의 일간 수익률 필요
- 50일 미만 시 에러 메시지 + 분석 스킵

---

## 9. Black-Litterman 포트폴리오 최적화

### 9.1 목적
Markowitz 효율적 프론티어의 **입력 민감성 문제**를 해결한다.
시장 균형 수익률(사전 분포)에 투자자의 **주관적 뷰**(사후 분포)를 결합하여 안정적인 최적 비중을 산출.

### 9.2 시장 균형 수익률 (Prior)

```python
# 역최적화: 시장 비중으로부터 내재 기대수익률 추출
π = δ × Σ × w_market

# δ: 위험회피계수 (기본 2.5)
# Σ: 연환산 공분산 행렬
# w_market: 시장 시가총액 비중 (또는 현재 포트폴리오 비중)
```

### 9.3 Black-Litterman 공식 (Posterior)

```python
# 투자자 뷰: Q = P × E[R] + ε
# P: 뷰 행렬 (K × N), K=뷰 수, N=종목 수
# Q: 뷰 기대수익률 (K × 1)
# Ω: 뷰 불확실성 (K × K, 대각)

# 사후 기대수익률
μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]

# 사후 공분산 (근사)
Σ_BL ≈ (1 + τ) × Σ

# τ: 스케일링 파라미터 (기본 0.05, 값이 클수록 뷰 반영 강화)
```

### 9.4 최적 비중 산출

```python
# 해석적 해: w* ∝ Σ_BL⁻¹ × (μ_BL - r_f)
raw_weights = np.linalg.solve(posterior_cov, posterior_returns - rf)
optimal_weights = max(0, raw_weights) / sum(max(0, raw_weights))  # long-only 제약
```

### 9.5 뷰 입력 규칙
- 절대 뷰: "AAPL이 연 15% 수익률을 낼 것"
- 상대 뷰: "AAPL이 MSFT보다 5% 더 좋을 것" (P 행렬 [+1, -1, 0, ...])
- 뷰 없으면: Prior(시장 균형)만 사용 → 시장 비중 ≈ 최적 비중

### 9.6 시각화 규칙

| 차트 | 타입 | 용도 |
|------|------|------|
| 사전/사후 수익률 비교 | `go.Bar` (그룹) | 뷰 반영 전후 비교 |
| 최적 비중 vs 현재 비중 | `go.Bar` (그룹) | 비중 조정 방향 파악 |
| 뷰 영향도 | `go.Bar` (수평) | 각 뷰가 수익률을 얼마나 바꿨는지 |
| 요약 테이블 | `go.Table` | 종목별 Prior/Posterior/비중 일람 |

### 9.7 파라미터 범위

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| δ (risk_aversion) | 2.5 | 0.5~10.0 | 높을수록 보수적 |
| τ (tau) | 0.05 | 0.01~0.5 | 높을수록 뷰 반영 강화 |

---

## 10. Rebalance Signal (리밸런싱 시그널)

### 10.1 목적
5가지 독립 트리거를 종합하여 **"지금 리밸런싱해야 하나?"**에 대한 정량적 긴급도(0~100)를 산출한다.

### 10.2 5-Trigger 구성

| 트리거 | 가중치 | 데이터 소스 |
|--------|--------|-------------|
| Regime Score | 30% | 시장 국면 (Bull/Bear/Sideways) |
| Skill Score | 25% | 6-Skills 점수 (특히 Risk Mgmt, Adaptability) |
| Volatility Score | 20% | 20일 롤링 드로다운 |
| Tracking Error Score | 15% | 벤치마크 대비 추적 오차 |
| Concentration Score | 10% | HHI 집중도 |

### 10.3 산출 공식

```python
# 1) Regime Score
if regime == 'Bear':
    regime_score = 80 + (1 - adaptability/100) * 20
elif regime == 'Sideways':
    regime_score = 50
else:  # Bull
    regime_score = 20

# 2) Skill Score (약점 기반)
weak_skills = [s for s in skills.values() if s < 50]
skill_score = min(100, len(weak_skills) * 25 + max(0, 50 - min(skills.values())))

# 3) Volatility Score
rolling_dd = 20일 롤링 드로다운
if rolling_dd > 0.10:
    vol_score = min(100, rolling_dd * 500)
else:
    vol_score = rolling_dd * 200

# 4) Tracking Error Score
te = std(portfolio_returns - benchmark_returns) * sqrt(252)
if te > 0.15:
    te_score = min(100, te * 400)
else:
    te_score = te * 200

# 5) Concentration Score
hhi = sum(weight_i ^ 2)
if hhi > 0.25:
    conc_score = min(100, hhi * 300)
else:
    conc_score = hhi * 150

# 종합 긴급도
urgency = 0.30 * regime + 0.25 * skill + 0.20 * vol + 0.15 * te + 0.10 * conc
```

### 10.4 방향 결정 규칙

| 조건 | 방향 | 의미 |
|------|------|------|
| Bear + 낮은 Adaptability + 높은 드로다운 | 🛡️ 방어적 | 리스크 축소 우선 |
| Bull + 높은 Conviction + 낮은 드로다운 | ⚔️ 공격적 | 수익 극대화 기회 |
| 그 외 | ⚖️ 유지 | 미세 조정만 |

### 10.5 시각화 규칙

| 차트 | 타입 | 용도 |
|------|------|------|
| 긴급도 게이지 | `go.Indicator(mode="gauge")` | 0~100 한눈에 파악 |
| 시그널 타임라인 | `go.Scatter + go.Bar` | 시간별 긴급도 변화 추적 |

### 10.6 해석 가이드

| 긴급도 | 등급 | 권장 행동 |
|--------|------|----------|
| 80~100 | 🔴 긴급 | 즉시 리밸런싱 실행 |
| 60~79 | 🟠 주의 | 1주 내 리밸런싱 검토 |
| 40~59 | 🟡 관찰 | 모니터링 지속 |
| 0~39 | 🟢 양호 | 현재 비중 유지 |

---

## 11. Portfolio DNA 핑거프린트

### 11.1 목적
포트폴리오의 **12차원 특성 벡터**를 DNA처럼 시각화하여, 투자 성향을 한눈에 파악하고
고유한 아키타입(투자자 유형)으로 자동 분류한다.

### 11.2 12차원 DNA 구성

| 카테고리 | 차원 | 소스 | 범위 |
|----------|------|------|------|
| **Skills** | Timing | skills_engine | 0~100 |
| | Diversification | skills_engine | 0~100 |
| | Risk Management | skills_engine | 0~100 |
| | Conviction | skills_engine | 0~100 |
| | Adaptability | skills_engine | 0~100 |
| | Consistency | skills_engine | 0~100 |
| **Style** | Value | style_analysis | 0~100 |
| | Growth | style_analysis | 0~100 |
| **Factor** | Momentum | factor_attribution β | 0~100 |
| | Volatility | garch vol_regime | 0~100 |
| **Structure** | Concentration | 1 - HHI 정규화 | 0~100 |
| | Sector Diversity | min(1, 섹터수/10) × 100 | 0~100 |

### 11.3 아키타입 자동 분류 규칙

```python
# 14+ 아키타입 예시
archetypes = {
    "안정형 가치투자자": strongest_skill in (RiskMgmt, Consistency) and style == 'Value',
    "공격형 모멘텀 트레이더": strongest_skill in (Timing, Conviction) and momentum > 60,
    "분산형 인덱스 추종자": diversification > 80 and concentration > 70,
    "집중형 성장투자자": conviction > 70 and style == 'Growth',
    "방어형 배당투자자": consistency > 70 and volatility < 40,
    "적응형 스윙트레이더": adaptability > 70 and timing > 60,
    "균형잡힌 올라운더": all skills between 50~70,
    ...
}
```

### 11.4 DNA 해시 생성

```python
# 12차원 각각을 0~9로 양자화
quantized = [min(9, int(score / 10)) for score in 12_dimensions]
dna_hash = "DNA-" + "".join(str(q) for q in quantized)
# 예: "DNA-786594738567"
```

### 11.5 시각화 규칙

| 차트 | 타입 | 용도 |
|------|------|------|
| DNA 레이더 (12축) | `go.Scatterpolar` | 12차원 핑거프린트 전체 조감 |
| DNA 비교 레이더 | 복수 `go.Scatterpolar` | 벤치마크 DNA와 오버레이 비교 |
| DNA 상세 테이블 | `go.Table` | 차원별 점수 + 카테고리 분류 |

---

## 12. Backtest Engine (백테스트)

### 12.1 목적
현재 포트폴리오 비중으로 **과거에 투자했다면** 어떤 성과였을지 시뮬레이션한다.
Single Backtest(특정 시점)과 Rolling Backtest(다중 진입점 매트릭스)를 제공.

### 12.2 Single Backtest 규칙

```python
# 입력
prices: DataFrame (T × N, 또는 MultiIndex)
weights: Dict[str, float]
start_date, end_date: str (YYYY-MM-DD)
benchmark: Series (optional)

# Step 1: 기간 슬라이싱 + MultiIndex 처리
prices_slice = prices[start:end]
# yfinance MultiIndex 자동 감지 및 flatten

# Step 2: 포트폴리오 일간 수익률
daily_returns = (prices_slice.pct_change() * weight_series).sum(axis=1)

# Step 3: 누적 곡선
cumulative = (1 + daily_returns).cumprod()

# Step 4: 핵심 지표 산출
total_return = cumulative.iloc[-1] - 1
years = len(daily_returns) / 252
annualized_return = (1 + total_return) ** (1/years) - 1
max_drawdown = min((cumulative - cumulative.cummax()) / cumulative.cummax())
sharpe = mean(excess_returns) / std(excess_returns) * sqrt(252)

# Step 5: 월간 수익률
monthly = daily_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
win_rate = (monthly > 0).mean()
best_month = monthly.max()
worst_month = monthly.min()

# Step 6: 알파 (벤치마크 대비)
alpha = total_return - benchmark_total_return
```

### 12.3 Rolling Backtest 규칙

```python
# 여러 진입 시점에서 반복 실행
window_months: int = 12  # 보유 기간
step_months: int = 3     # 진입 간격

for start in generate_start_dates(step=step_months):
    end = start + window_months
    result = run_backtest(prices, weights, benchmark, start, end)
    results.append(result)

# 결과: 진입 시점별 수익률/Sharpe/MDD 매트릭스
```

### 12.4 시각화 규칙

| 차트 | 타입 | 용도 |
|------|------|------|
| 누적수익률 비교 | `go.Scatter` (이중) | 포트폴리오 vs 벤치마크 |
| 드로다운 영역 | `go.Scatter(fill='tozeroy')` | MDD 구간 강조 |
| 롤링 성과 바 | `go.Bar` | 진입 시점별 수익률 비교 |
| 월별 히트맵 | `go.Heatmap(RdYlGn)` | Year×Month 수익률 패턴 |
| 성과 요약 테이블 | `go.Table` | 핵심 7개 지표 일람 |

### 12.5 해석 가이드

| 조건 | 진단 |
|------|------|
| 승률 > 60% + Sharpe > 1.0 | "안정적으로 시장을 이길 수 있는 전략" |
| 승률 > 60% + Sharpe < 0.5 | "빈도는 높지만 수익 폭이 작음 → 확신 부족" |
| MDD > -30% | "위기 시 큰 낙폭 → 리스크 관리 강화 필요" |
| Alpha > 10% | "벤치마크 대비 뚜렷한 초과수익" |
| 롤링 결과 편차 큼 | "진입 타이밍에 민감 → Timing Skill 중요" |

---

## 13. Performance+ (고급 성과 지표)

### 13.1 목적
기본 Sharpe/MDD를 넘어 **벤치마크 대비 상대 성과**를 정밀 측정한다.
액티브 운용 역량을 정량적으로 평가하는 핵심 지표 세트.

### 13.2 핵심 지표

| 지표 | 공식 | 해석 |
|------|------|------|
| Tracking Error (TE) | `σ(r_p - r_b) × √252` | 벤치마크 대비 추적 오차 |
| Information Ratio (IR) | `(r_p_annual - r_b_annual) / TE` | 추적 오차 1단위당 초과수익 |
| Up Capture Ratio | `annualized(r_p when r_b>0) / annualized(r_b when r_b>0)` | 상승장 포착률 |
| Down Capture Ratio | `annualized(r_p when r_b<0) / annualized(r_b when r_b<0)` | 하락장 노출률 |
| Capture Spread | `Up Capture - Down Capture` | 양수일수록 우수 |
| Beta | `Cov(r_p, r_b) / Var(r_b)` | 시장 민감도 |
| Correlation | `Corr(r_p, r_b)` | 벤치마크 추종도 |

### 13.3 산출 공식

```python
# Tracking Error
excess_returns = portfolio_returns - benchmark_returns
tracking_error = sqrt(mean(excess_returns ** 2)) * sqrt(252)

# Information Ratio
active_return = (1+portfolio_returns).prod()**(252/T) - (1+benchmark_returns).prod()**(252/T)
information_ratio = active_return / tracking_error

# Up Capture
bull_days = benchmark_returns > 0
up_capture = ((1 + portfolio_returns[bull_days]).prod() ** (252/bull_count) - 1) / \
             ((1 + benchmark_returns[bull_days]).prod() ** (252/bull_count) - 1)

# Down Capture (낮을수록 좋음)
bear_days = benchmark_returns < 0
down_capture = ((1 + portfolio_returns[bear_days]).prod() ** (252/bear_count) - 1) / \
               ((1 + benchmark_returns[bear_days]).prod() ** (252/bear_count) - 1)

# Beta
beta = np.cov(portfolio_returns, benchmark_returns)[0][1] / np.var(benchmark_returns)
```

### 13.4 종합 성과 등급

```python
# IR 기반 등급
if ir > 0.5:
    grade = 'S'  # 탁월한 액티브 운용
elif ir > 0.3:
    grade = 'A'  # 우수
elif ir > 0.1:
    grade = 'B'  # 평균 이상
elif ir > 0:
    grade = 'C'  # 미미한 초과수익
else:
    grade = 'D'  # 벤치마크 하회

# Capture 기반 보정
if up_capture > 1.1 and down_capture < 0.9:
    grade = min(grade, 'A')  # 최소 A등급 보장
```

### 13.5 시각화 규칙

| 차트 | 타입 | 용도 |
|------|------|------|
| Capture Ratio 바 | `go.Bar` (그룹) | Up/Down Capture 한눈에 비교 |
| 성과 비교 바 | `go.Bar` (그룹) | 포트폴리오 vs 벤치마크 핵심 지표 |
| 종합 성과 게이지 | `go.Indicator(mode="gauge")` | IR 기반 등급 시각화 |
| 롤링 IR 라인 | `go.Scatter` | 시간별 IR 변화 추적 |
| 요약 테이블 | `go.Table` | 전체 지표 일람 |

### 13.6 해석 가이드

| 조건 | 진단 |
|------|------|
| IR > 0.5 | "탁월한 액티브 운용 능력 — 추적 오차 대비 높은 초과수익" |
| Up Capture > 110% + Down Capture < 90% | "이상적 — 상승장 포착 + 하락장 방어 모두 우수" |
| Up Capture < 90% + Down Capture > 110% | "역전 — 상승장 놓치고 하락장에 더 많이 노출" |
| TE > 15% | "벤치마크와 매우 다른 전략 — 독자적 포지셔닝" |
| TE < 3% | "거의 인덱스 추종 — 패시브에 가까움" |
| Beta > 1.2 | "레버리지형 — 시장 변동의 1.2배 노출" |
| Beta < 0.8 | "방어형 — 시장 하락 시 완충 효과" |

---

## 14. 통계 상수 (확장)

```python
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.035             # 연 3.5%
BEAR_MARKET_THRESHOLD = -0.10      # 벤치마크 드로다운 -10%
TIMING_LOOKBACK_DAYS = 30
CONVICTION_TOP_N = 5
FRONTIER_SAMPLES = 3000            # 효율적 프론티어 기본 샘플 수
CORR_THRESHOLD_DEFAULT = 0.3       # 상관관계 네트워크 기본 임계값
STRESS_GBM_SEED = 42               # 재현성을 위한 시드
GEOPOLITICAL_PATH_DAYS = 30        # 지정학 시나리오 시뮬레이션 기간
MAX_COMPARISON_PORTFOLIOS = 5      # 동시 비교 최대 수
GARCH_MIN_OBSERVATIONS = 50        # GARCH 최소 관측치
GARCH_MAX_ITER = 100               # MLE 최대 반복
BL_DEFAULT_RISK_AVERSION = 2.5     # Black-Litterman 위험회피계수
BL_DEFAULT_TAU = 0.05              # Black-Litterman 스케일링 파라미터
REBALANCE_URGENCY_WEIGHTS = {      # 리밸런싱 트리거 가중치
    "regime": 0.30,
    "skill": 0.25,
    "volatility": 0.20,
    "tracking_error": 0.15,
    "concentration": 0.10,
}
DNA_DIMENSIONS = 12                # Portfolio DNA 차원 수
BACKTEST_DEFAULT_WINDOW = 12       # 롤링 백테스트 기본 윈도우 (개월)
BACKTEST_DEFAULT_STEP = 3          # 롤링 백테스트 스텝 (개월)
PERFORMANCE_IR_THRESHOLDS = {      # IR 등급 임계치
    "S": 0.5, "A": 0.3, "B": 0.1, "C": 0.0,
}
```
