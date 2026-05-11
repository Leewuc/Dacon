# Skills_analysis.md — 투자 데이터 분석 규칙 정의

## 1. 6-Skills 산출 공식

이 문서는 투자 포트폴리오의 6가지 역량(Skills) 점수를 산출하는 정확한 분석 규칙을 정의한다.
모든 점수는 0~100 범위로 정규화하며, 벤치마크 대비 상대 평가를 원칙으로 한다.

---

## 2. Timing Skill (매수/매도 타이밍 역량)

### 2.1 산출 조건
- 매매 이력(trades)이 있을 때만 산출
- 매매 이력 없으면 기본값 50점 (중립)

### 2.2 산출 공식

```python
# 매수 타이밍 점수
buy_score = 1 - (매수가 - 30일내_최저가) / 매수가

# 매도 타이밍 점수
sell_score = 1 - (30일내_최고가 - 매도가) / 매도가

# 종합 Timing Score
timing_score = mean(all_scores) * 100
```

### 2.3 해석 규칙
- 90+ (S): 거의 최적점에서 매수/매도
- 75+ (A): 타이밍이 우수함
- 55+ (B): 평균적인 타이밍
- 35+ (C): 타이밍 개선 필요
- 0~34 (D): 고점 매수/저점 매도 경향

---

## 3. Diversification Skill (분산투자 역량)

### 3.1 산출 공식

```python
# HHI (Herfindahl-Hirschman Index) - 종목 집중도
hhi = sum(weight_i ^ 2)  # 0~1, 낮을수록 분산
hhi_score = (1 - hhi) / (1 - 1/n) * 100  # 정규화

# 섹터 엔트로피 - 섹터 분산도
entropy = -sum(sw_i * log2(sw_i))
entropy_score = (entropy / log2(섹터수)) * 100

# 종합
diversification_score = 0.5 * hhi_score + 0.5 * entropy_score
```

### 3.2 분석 규칙
- 종목 수가 1개면 HHI = 1.0 → 점수 0
- 균등 가중이면 HHI = 1/n → 점수 100
- 섹터 정보 없으면 entropy_score는 기본 50

### 3.3 세부 지표 출력
- HHI 값 (소수점 4자리)
- 보유 종목 수
- HHI 점수, 엔트로피 점수 각각

---

## 4. Risk Management Skill (리스크 관리 역량)

### 4.1 사용 지표
| 지표 | 공식 | 가중치 |
|------|------|--------|
| Sharpe Ratio | `(mean_r - rf/252) / std_r * sqrt(252)` | 40% |
| Max Drawdown | `max(peak - trough) / peak` | 30% |
| Sortino Ratio | `(mean_r - rf/252) / downside_std * sqrt(252)` | 30% |

### 4.2 산출 공식

```python
# 무위험 수익률: 연 3.5% 기본값
rf = 0.035

# Sharpe 상대 점수
if bench_sharpe > 0:
    sharpe_score = min(100, max(0, (my_sharpe / bench_sharpe) * 50))
elif my_sharpe > 0:
    sharpe_score = 70
else:
    sharpe_score = 30

# MDD 상대 점수 (MDD가 벤치마크보다 작으면 좋음)
if my_mdd > 0 and bench_mdd > 0:
    mdd_score = min(100, max(0, (bench_mdd / my_mdd) * 50))
elif my_mdd == 0:
    mdd_score = 100
else:
    mdd_score = 30

# Sortino 절대 점수
sortino_score = min(100, max(0, 50 + sortino * 15))

# 종합
risk_score = 0.4 * sharpe_score + 0.3 * mdd_score + 0.3 * sortino_score
```

### 4.3 세부 지표 출력
- Sharpe Ratio (소수점 3자리)
- Sortino Ratio (소수점 3자리)
- Max Drawdown (퍼센트, 소수점 2자리)
- Calmar Ratio (소수점 3자리)
- 벤치마크 Sharpe, 벤치마크 MDD

---

## 5. Conviction Skill (확신 포지션 운용 역량)

### 5.1 정의
비중 상위 5개 종목(확신 종목)의 성과가 나머지 종목 대비 얼마나 우수한지 측정

### 5.2 산출 공식

```python
# 상위 5 종목 가중 수익률
top5_return = sum(w_i * r_i for top5) / sum(w_i for top5)

# 나머지 종목 가중 수익률
rest_return = sum(w_i * r_i for rest) / sum(w_i for rest)

# Conviction Alpha
conviction_alpha = top5_return - rest_return

# 점수 (alpha가 양수이면 확신이 맞았음)
conviction_score = min(100, max(0, 50 + conviction_alpha * 200))
```

### 5.3 해석 규칙
- 점수 > 70: 집중 투자 종목이 outperform → 확신이 옳았음
- 점수 = 50: 차이 없음
- 점수 < 30: 집중 투자 종목이 underperform → 역선택

### 5.4 세부 지표 출력
- Top 5 종목 리스트
- Top 5 가중 수익률
- 나머지 가중 수익률
- Conviction Alpha
- Top 5 집중도 (비중 합계)

---

## 6. Adaptability Skill (시장 변화 적응력)

### 6.1 정의
시장 하락기(베어마켓)에서 포트폴리오가 벤치마크 대비 얼마나 방어했는지 측정

### 6.2 베어마켓 식별 규칙

```python
# 벤치마크 누적 수익률 기준 드로다운
bench_cumulative = (1 + bench_returns).cumprod()
bench_rolling_max = bench_cumulative.cummax()
bench_drawdown = (bench_cumulative - bench_rolling_max) / bench_rolling_max

# 드로다운 -10% 이하 구간 = 베어마켓
bear_market = bench_drawdown < -0.10
```

### 6.3 산출 공식

```python
# 베어마켓 구간 상대 수익률
bear_relative = mean(my_returns[bear]) - mean(bench_returns[bear])
bear_score = min(100, max(0, 50 + bear_relative * 500))

# 리밸런싱 효과 (리밸런싱 날짜가 있을 때)
rebal_score = min(100, max(0, 50 + rebal_premium * 300))

# 종합 (베어마켓 방어 60%, 리밸런싱 40%)
adaptability_score = 0.6 * bear_score + 0.4 * rebal_score
```

### 6.4 세부 지표 출력
- 베어마켓 구간 일수
- 베어마켓 방어 점수
- 리밸런싱 효과 점수

---

## 7. Consistency Skill (수익 일관성)

### 7.1 산출 공식

```python
# 월간 수익률로 변환
monthly_returns = daily_returns.resample('ME').apply(lambda x: (1+x).prod()-1)

# 1) 양의 수익 월 비율 (Win Rate)
win_rate = (monthly_returns > 0).mean() * 100

# 2) 변동계수 (CV) — 낮을수록 일관
cv = std(monthly) / abs(mean(monthly))
cv_score = max(0, 100 - cv * 20)

# 3) 연속 양의 수익 최대 streak
max_streak = 연속으로 월간 수익 > 0인 최대 개월 수
streak_score = min(100, max_streak * 12)

# 종합
consistency_score = 0.4 * win_rate + 0.3 * cv_score + 0.3 * streak_score
```

### 7.2 세부 지표 출력
- Win Rate (%)
- 변동계수 (CV)
- 최대 연속 양의 수익 streak (개월)
- 분석 대상 총 월수

---

## 8. 종합 프로필 생성 규칙

### 8.1 Overall Score
```python
overall = mean(6개 skill 점수)
```

### 8.2 등급 매핑
| 점수 | 등급 |
|------|------|
| 90+ | S |
| 75~89 | A |
| 55~74 | B |
| 35~54 | C |
| 0~34 | D |

### 8.3 출력 형식
각 Skill마다 반드시 다음을 포함:
1. 점수 (0~100, 소수점 1자리)
2. 등급 (S/A/B/C/D)
3. 한줄 설명 (등급별 자동 생성)
4. 세부 지표 딕셔너리

---

## 9. 벤치마크 규칙

### 9.1 지원 벤치마크
| 코드 | 설명 |
|------|------|
| SPY | S&P 500 ETF (기본값) |
| QQQ | 나스닥 100 ETF |
| IWM | 러셀 2000 소형주 |
| EFA | 선진국 해외 주식 |
| VT | 글로벌 전체 주식 |

### 9.2 인덱스 정렬
포트폴리오와 벤치마크의 날짜 인덱스를 교집합으로 정렬한 후 분석 수행

---

## 10. 통계 상수

```python
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.035  # 연 3.5%
BEAR_MARKET_THRESHOLD = -0.10  # 벤치마크 드로다운 -10%
TIMING_LOOKBACK_DAYS = 30
CONVICTION_TOP_N = 5
```
