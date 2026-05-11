# Skills_insight_report.md — 투자 인사이트 생성 & 리포트 구성 규칙

이 문서는 대시보드가 **자동으로 인사이트를 생성**하는 룰 체계와,
분석 결과를 **리포트로 구성하는 흐름**을 정의한다.

---

## 1. 인사이트 생성 아키텍처

### 1.1 설계 원칙
- **API 키 없이 동작**: 모든 인사이트는 룰 기반 NLG (Natural Language Generation)
- **조건 → 진단 → 제안** 3단계 구조
- **한국어 우선**: 코멘터리는 한국어, 지표명은 영어 유지 (국제 표준)
- **등급 기반 분기**: 모든 코멘터리는 S/A/B/C/D 등급에 따라 다른 텍스트 생성

### 1.2 인사이트 레이어 구조

```
Layer 1: 개별 Skill 인사이트 (6개)
    ↓ 조합
Layer 2: 종합 포트폴리오 인사이트
    ↓ 확장
Layer 3: 시나리오 인사이트 (What-If, Stress Test, Geopolitical)
    ↓ 비교
Layer 4: 벤치마크 대비 인사이트 (Multi-Portfolio)
```

---

## 2. AI 코멘터리 생성 규칙 (Layer 1 & 2)

### 2.1 개별 Skill 코멘터리 생성 흐름

각 Skill에 대해:
```
입력: (score, grade, detail_dict)
    ↓
분기: grade ∈ {S, A} → 칭찬 + 유지 조언
      grade = B      → 중립 + 개선 힌트
      grade ∈ {C, D} → 경고 + 구체적 개선안
    ↓
출력: (진단_텍스트, 개선_제안 or None)
```

### 2.2 Skill별 코멘터리 규칙

**Timing (타이밍):**

| 등급 | 진단 템플릿 | 제안 |
|------|-----------|------|
| S/A | "매수/매도 타이밍 역량이 {등급설명}합니다({score}점)." | None |
| B | "타이밍 역량은 평균 수준입니다({score}점)." | "RSI, MACD 등 기술적 지표 활용 권장" |
| C/D | "고점 매수 또는 저점 매도 경향이 있습니다({score}점)." | "분할 매수/매도 전략 도입 (30%/30%/40%)" |

**Diversification (분산투자):**

| 등급 | 추가 분석 | 제안 |
|------|----------|------|
| S/A | - | None |
| B | 섹터 편중 분석 (top_sector > 35% 시 경고) | "현재 없는 업종 편입 검토" |
| C/D | 섹터 편중 분석 + HHI 경고 | "최소 7~10종목, 4+ 섹터 분산. ETF 활용" |

**Risk Management (리스크 관리):**

| 등급 | 핵심 지표 표시 | 제안 |
|------|------------|------|
| S/A | Sharpe, MDD | None |
| B | MDD | "손절 기준 설정 (-10% trailing stop)" |
| C/D | MDD (30% 초과 시 추가 경고) | "채권/금 10~20% 편입 + 분기 리밸런싱" |

**Conviction (확신 투자):**

| 등급 | 핵심 지표 | 제안 |
|------|----------|------|
| S/A | Top 5 종목 + alpha | None |
| B | - | "확신 종목 비중 점진 상향, 단일 종목 25% 상한" |
| C/D | Conviction Alpha (음수 표시) | "펀더멘털+기술적 신호 이중 확인 프로세스" |

**Adaptability (적응력):**

| 등급 | 핵심 지표 | 제안 |
|------|----------|------|
| S/A | 베어마켓 방어 성공 일수 | None |
| B | - | "VIX 급등 시 현금 비중 10~20% 확보" |
| C/D | 벤치마크 대비 초과 손실 구간 | "인버스 ETF 소량 헤지 또는 역추세 리밸런싱" |

**Consistency (일관성):**

| 등급 | 핵심 지표 | 제안 |
|------|----------|------|
| S/A | 월간 승률, 연속 양의 수익 streak | None |
| B | 월간 승률 | "변동성 큰 종목 축소, 배당주/채권 편입" |
| C/D | 승률 < 50% 경고 | "주식+채권+대안자산 분산, 적립식 투자" |

### 2.3 종합 코멘터리 조합 규칙

```python
# 1. 한줄 요약
summary = f"종합 {grade}등급({score}점). 수익률 {ret}% ({alpha} vs BM)."
if strong_skills: summary += f"{strongest}이 가장 강함."
if weak_skills: summary += f"{weakest}에 개선 여지."

# 2. 상세 진단 = 성과 개요 + 6개 Skill 진단 연결
diagnosis = performance_overview + " ".join(skill_comments)

# 3. 강점 리스트 = score >= 55인 Skills
# 4. 약점 리스트 = score < 55인 Skills
# 5. 개선 제안 = C/D 등급 Skills의 제안 모음
# 6. 리스크 경고 = 조건부 (아래 섹션 참고)
```

### 2.4 리스크 경고 트리거 규칙

| 조건 | 경고 수준 | 메시지 |
|------|----------|--------|
| MDD > 30% AND Diversification < 40 | 🔴 고위험 | "포트폴리오 전면 재검토 강력 권장" |
| MDD > 25% | 🟡 주의 | "손실 허용 범위 재확인, 방어 자산 검토" |
| 6개 Skill 중 3개 이상 D등급 | 🔴 고위험 | "투자 전략 전반의 근본적 재검토 필요" |
| Conviction C등급 + Timing C등급 | 🟡 주의 | "잘못된 종목에 잘못된 타이밍으로 투자 패턴" |

---

## 3. 시나리오 인사이트 규칙 (Layer 3)

### 3.1 What-If 인사이트

```python
# Skill 변화 기반 인사이트
improved = {skill: change for change > 2}
degraded = {skill: change for change < -2}

if improved: "향상되는 Skills: {list}"  # st.success
if degraded: "하락하는 Skills: {list}"  # st.warning
if not improved and not degraded: "유의미한 변화 없음"  # st.info

# Overall 변화 기반
if overall_change > 5: "전략 적용 권장"
if overall_change < -5: "트레이드오프 신중히 검토"
```

### 3.2 Stress Test 인사이트

```python
# 최악/최선 시나리오 식별
worst = max(results, key=|portfolio_loss|)
best = min(results, key=|portfolio_loss|)

# 경고 수준
if |worst.loss| > 40%: "40%+ 손실 예상 → 방어 자산 강력 권장"  # st.error
if |worst.loss| > 25%: "25%+ 손실 예상 → 섹터 분산 검토"       # st.warning
else: "위기 시나리오에서 비교적 견고"                            # st.success
```

### 3.3 Geopolitical 인사이트

```python
# 포트폴리오 영향 기반
if impact < -15%: "심각한 하락 예상 → {worst_sector} 비중 축소/헤지"  # st.error
if impact < -5%:  "부정적 영향 → 방어적 자산 편입"                   # st.warning
if impact > 5%:   "시나리오에서 수혜"                               # st.success
else:            "영향 제한적"                                      # st.info
```

### 3.4 Efficient Frontier 인사이트

```python
gap = max_sharpe.sharpe - current_sharpe

if gap > 0.3: "최적 대비 비효율 → 비중 조정 권장"  # st.warning
if gap > 0.1: "효율적 프론티어에 비교적 근접"       # st.info
else:         "잘 최적화된 포트폴리오"              # st.success
```

### 3.5 Correlation 인사이트

```python
if avg_corr > 0.6: "높은 상관 → 분산 효과 제한적"  # st.warning
if avg_corr > 0.4: "보통 수준"                    # st.info
else:             "좋은 분산 효과"                 # st.success

if n_clusters <= 2 and n_stocks > 4:
    "대부분 동일 클러스터 → 다른 섹터 편입 권장"  # st.warning
```

### 3.6 GARCH 변동성 인사이트

```python
# 변동성 국면 기반
if vol_regime == 'HIGH':
    "현재 변동성 급등 구간 — 포지션 축소 또는 헤지 검토"  # st.error
elif vol_regime == 'MEDIUM':
    "변동성 정상 범위 — 현재 전략 유지 가능"              # st.info
else:  # LOW
    "저변동성 구간 — 기회 포착 또는 변동성 매도 전략 고려"  # st.success

# 지속성 기반
persistence = alpha + beta
if persistence > 0.95:
    "매우 높은 지속성 → 변동성 충격이 오래 지속됨"  # st.warning
if half_life > 50:
    "반감기 {half_life}일 → 변동성 정상화에 {half_life}거래일 소요"  # st.info

# 예측 기반
if forecast_30d > long_run_vol * 1.5:
    "30일 후에도 변동성 높을 전망 → 단기 방어 유지"  # st.warning
```

### 3.7 Black-Litterman 인사이트

```python
# 최적 비중 vs 현재 비중 괴리
max_drift = max(|optimal_weight - current_weight|)
if max_drift > 0.15:
    "최적 비중과 15%p 이상 차이 → 리밸런싱 검토"  # st.warning
elif max_drift > 0.05:
    "적정 범위 내 차이 → 미세 조정 고려"          # st.info
else:
    "현재 비중이 BL 최적에 근접"                   # st.success

# 뷰 반영 효과
if views_applied and max(|posterior - prior|) > 0.05:
    "투자자 뷰가 기대수익률에 유의미한 변화를 가져옴"  # st.info
```

### 3.8 Rebalance Signal 인사이트

```python
# 긴급도 기반 자동 진단
if urgency >= 80:
    "🔴 즉시 리밸런싱 필요 — {top_trigger}이 주요 트리거"  # st.error
elif urgency >= 60:
    "🟠 1주 내 리밸런싱 검토 — {direction} 방향 권장"      # st.warning
elif urgency >= 40:
    "🟡 모니터링 지속 — 현재 비중 유지하되 주시"           # st.info
else:
    "🟢 현재 포트폴리오 양호 — 변경 불필요"               # st.success

# 트리거별 상세
for trigger, score in trigger_scores.items():
    if score > 70:
        "{trigger} 스코어 {score} → 해당 영역 집중 점검"  # st.warning
```

### 3.9 Portfolio DNA 인사이트

```python
# 아키타입 진단
"귀하의 투자 DNA: {archetype} 유형입니다."  # 항상 표시

# 강점/약점 차원
strongest_dim = max(12_dimensions)
weakest_dim = min(12_dimensions)
"가장 강한 DNA: {strongest_dim.name} ({score}점)"
"가장 약한 DNA: {weakest_dim.name} ({score}점) — 보완 필요"

# 균형도
std_of_dims = std(12_scores)
if std_of_dims < 10:
    "매우 균형 잡힌 프로필 — 올라운더형"  # st.success
elif std_of_dims > 25:
    "극단적 편향 — 특화형 투자자, 약점 보완 검토"  # st.warning

# DNA 해시
"포트폴리오 고유 ID: {dna_hash} — 이 핑거프린트로 프로필 추적 가능"
```

### 3.10 Backtest 인사이트

```python
# Single Backtest
if alpha > 0.10:
    "벤치마크 대비 +{alpha}% 초과수익 — 전략 유효성 검증됨"  # st.success
elif alpha > 0:
    "소폭 초과수익 — 전략 효과 있으나 개선 여지 존재"        # st.info
else:
    "벤치마크 하회 — 전략 재검토 필요"                       # st.warning

if win_rate > 0.6 and sharpe > 1.0:
    "높은 승률 + 우수한 Sharpe → 안정적 전략"  # st.success
if max_drawdown < -0.30:
    "MDD {mdd}% — 위기 시 큰 낙폭 주의"  # st.error

# Rolling Backtest
rolling_std = std(rolling_returns)
if rolling_std > 0.15:
    "진입 시점에 따른 성과 편차 큼 → Timing Skill 중요"  # st.warning
else:
    "진입 시점 무관하게 안정적 성과 — 전략 강건성 양호"   # st.success
```

### 3.11 Performance+ 인사이트

```python
# Information Ratio 기반
if ir > 0.5:
    "탁월한 액티브 운용 — 추적 오차 대비 높은 초과수익"     # st.success
elif ir > 0.3:
    "우수한 수준 — 액티브 전략이 유효함"                   # st.success
elif ir > 0:
    "미미한 초과수익 — 비용 고려 시 패시브 전환 검토"      # st.info
else:
    "벤치마크 하회 — 패시브 전략이 더 유리했을 수 있음"    # st.warning

# Capture Ratio 기반
if up_capture > 1.1 and down_capture < 0.9:
    "이상적 패턴 — 상승장 포착 + 하락장 방어 모두 우수"    # st.success
elif up_capture < 0.9 and down_capture > 1.1:
    "역전 패턴 — 상승장 놓치고 하락장에 더 노출"          # st.error
    "포지셔닝 전면 재검토 필요"

# Tracking Error 기반
if te > 0.15:
    "벤치마크와 매우 다른 전략 — 독자적 포지셔닝"         # st.info
elif te < 0.03:
    "사실상 인덱스 추종 — 차별화 전략 부재"              # st.info
```

---

## 4. 리포트 구성 흐름 설계

### 4.1 분석 순서 (Progressive Disclosure)

대시보드의 탭은 **6개 카테고리 × 23탭**으로 구성되며, 논리적 깊이 순서로 배치한다:

```
[카테고리 1: 📊 Core Analysis] → 핵심 진단
├─ Tab 1: Skills Analysis      → 6-Skills 레이더 + AI 진단 + PDF
├─ Tab 2: Performance          → 수익률, 드로다운, 월별 히트맵
├─ Tab 3: Allocation           → 섹터 트리맵, 리스크-리턴
└─ Tab 4: Monte Carlo          → 확률적 미래 예측

[카테고리 2: 🔬 Deep Analysis] → 원인 분석
├─ Tab 5: What-If Scenario     → 비중 변경 시뮬레이션
├─ Tab 6: Factor Attribution   → 수익 원천 분해
├─ Tab 7: Skills Evolution     → 역량 변화 추적
└─ Tab 8: Efficient Frontier   → Markowitz 최적화

[카테고리 3: ⚠️ Risk & Stress] → 리스크 시나리오
├─ Tab 9: Stress Test          → 역사적 위기 리플레이
├─ Tab 10: Geopolitical        → 지정학 충격 시뮬레이션
├─ Tab 11: Correlation Network → 종목 간 관계 분석
└─ Tab 12: Multi-Portfolio     → 벤치마크 대비 평가

[카테고리 4: 📈 Market Insight] → 시장 분석
├─ Tab 13: Risk Contribution   → Euler 리스크 기여도 분해
├─ Tab 14: Style Analysis      → Morningstar 스타일 박스
├─ Tab 15: Market Events       → 39개 글로벌 이벤트 타임라인
├─ Tab 16: Regime Detection    → Bull/Bear/Sideways 국면 탐지
└─ Tab 17: Tail Risk           → Cornish-Fisher VaR + CVaR

[카테고리 5: 🧬 Advanced Models] → 고급 모델
├─ Tab 18: GARCH(1,1)         → 변동성 예측 모델
├─ Tab 19: Black-Litterman    → BL 포트폴리오 최적화
├─ Tab 20: Rebalance Signal   → 5-Trigger 리밸런싱 시그널
└─ Tab 21: Portfolio DNA       → 12차원 핑거프린트

[카테고리 6: 🏆 Performance+] → 성과 검증
├─ Tab 22: Backtest Engine    → Single/Rolling 백테스트
└─ Tab 23: Performance+       → IR, TE, Capture Ratio
```

### 4.2 PDF 리포트 구성 규칙

PDF 리포트는 대시보드의 **핵심만 추출**하여 2~3페이지로 구성한다:

```
페이지 1: Executive Summary
├─ 제목 + 생성 일시
├─ Performance Summary 테이블
│   (수익률, Alpha, Sharpe, MDD, Overall Skill)
├─ Skills Analysis 테이블
│   (6 Skills × 점수 + 등급)
└─ AI Commentary 요약
    (summary + recommendations)

페이지 2: Deep Analysis
├─ Factor Attribution 요약 (4팩터 기여도)
├─ GARCH 변동성 진단 (현재 국면 + 예측)
├─ Backtest 핵심 결과 (승률, Sharpe, Alpha)
└─ Performance+ 요약 (IR, Capture Spread)

[선택] 페이지 3: Risk & Optimization
├─ Stress Test 최악 시나리오
├─ Rebalance Signal 긴급도 + 방향
├─ Black-Litterman 최적 비중 권고
├─ Portfolio DNA 아키타입 + 해시
└─ 주요 개선 제안
```

### 4.3 PDF 생성 규칙
- 라이브러리: `reportlab` (Platypus 엔진)
- 용지: A4
- 한글 폰트: Noto Sans CJK 우선 → Apple SD Gothic → DejaVu Sans 폴백
- 테이블 스타일: 헤더 Indigo 배경 + 흰색 텍스트, 줄무늬 행
- 파일명: `investment_skills_report_YYYYMMDD.pdf`

---

## 5. 인사이트 표시 컴포넌트 규칙

### 5.1 Streamlit 컴포넌트 매핑

| 인사이트 수준 | 컴포넌트 | 사용 조건 |
|-------------|---------|----------|
| 긍정 | `st.success("✅ ...")` | 강점, 목표 달성, 수혜 시나리오 |
| 정보 | `st.info("ℹ️ ...")` | 중립, 보통 수준, 변화 없음 |
| 주의 | `st.warning("⚠️ ...")` | 개선 필요, 주의 사항 |
| 위험 | `st.error("🔴 ...")` | 고위험, 즉각 조치 필요 |

### 5.2 AI 코멘터리 UI 규칙

```
[리스크 경고] ← st.error (조건부, 있을 때만)

[종합 평가 카드] ← gradient 배경 (#1E293B → #312E81)
│  "종합 B등급(68점). 수익률 +12.3%..."

[강점 | 약점] ← 2열 레이아웃
│  ✅ Timing (A, 78점)    │  ⚠️ Diversification (C, 42점)
│  ✅ Conviction (B, 65점) │  ⚠️ Consistency (D, 28점)

[상세 진단] ← st.expander (접이식, 기본 닫힘)
[개선 제안] ← st.expander (접이식, 기본 열림)
│  1. 분할 매수/매도 전략 도입...
│  2. 최소 7~10개 종목, 4개 이상 섹터...

[PDF 다운로드] ← st.download_button
```

### 5.3 시나리오 Insight UI 규칙

```
[KPI 카드 4개] ← st.columns(4) + st.metric
    포트폴리오 영향 | 리스크 점수 | 최대 타격 | 최대 수혜

[차트 영역] ← st.columns([1, 1]) 2열 배치
    좌: 워터폴/레이더/경로    우: 바/히트맵/테이블

[인사이트 블록] ← st.success/warning/error
    조건별 자동 진단 메시지
```

---

## 6. 교차 분석 인사이트 규칙

여러 탭의 결과를 **교차 참조**하여 더 깊은 인사이트를 생성하는 규칙:

### 6.1 Skills + Factor Attribution 교차

| Skills 조건 | Factor 조건 | 교차 인사이트 |
|------------|------------|-------------|
| Risk Mgmt S등급 | β_mkt < 0.8 | "방어적 전략이 리스크 관리 역량을 끌어올림" |
| Timing C등급 | Alpha 음수 | "타이밍 실패가 음의 Alpha로 연결됨" |
| Diversification A등급 | R² < 0.5 | "분산투자로 시장 독립적 수익 구조 구축" |

### 6.2 Skills + Stress Test 교차

| Skills 조건 | Stress 조건 | 교차 인사이트 |
|------------|-----------|-------------|
| Adaptability C등급 | COVID -40% 이상 | "시장 적응력 부족 → 위기 시 큰 손실" |
| Diversification A등급 | 모든 시나리오 -20% 미만 | "분산투자가 위기 방어에 효과적" |

### 6.3 Geopolitical + Correlation 교차

| Geo 조건 | Corr 조건 | 교차 인사이트 |
|---------|----------|-------------|
| 유가 충격 -20% | 에너지 클러스터 비중 30%+ | "유가 민감 종목 집중 → 호르무즈 시나리오 취약" |
| 공급망 충격 -15% | 반도체 클러스터 비중 25%+ | "공급망 의존도 높음 → 대만 시나리오 주의" |

### 6.4 GARCH + Rebalance Signal 교차

| GARCH 조건 | Rebalance 조건 | 교차 인사이트 |
|-----------|---------------|-------------|
| vol_regime = HIGH | urgency > 60 | "변동성 급등 + 리밸런싱 긴급 → 즉시 방어적 조정 필요" |
| vol_regime = LOW | urgency < 30 | "안정 구간 + 리밸런싱 불필요 → 현재 전략 유지" |
| persistence > 0.95 | direction = 방어적 | "변동성 장기 지속 전망 → 방어 포지션 유지 권장" |

### 6.5 Portfolio DNA + Backtest 교차

| DNA 조건 | Backtest 조건 | 교차 인사이트 |
|---------|-------------|-------------|
| 아키타입 = "공격형 모멘텀" | 롤링 편차 큼 | "공격형 전략 + 타이밍 민감 → Timing Skill 강화 필수" |
| 아키타입 = "안정형 가치" | 승률 > 60% | "안정형 DNA에 부합하는 높은 승률 — 전략 적합도 우수" |
| Concentration < 40 | Alpha < 0 | "분산 과다 → 확신 종목 비중 상향으로 Alpha 개선 가능" |

### 6.6 Performance+ + Skills 교차

| Perf+ 조건 | Skills 조건 | 교차 인사이트 |
|-----------|-----------|-------------|
| IR > 0.5 | Overall S등급 | "탁월한 IR + 최상위 Skills → 포트폴리오 매니저 수준 역량" |
| Up Capture > 110% | Timing A등급 | "우수한 시장 포착 = 타이밍 역량이 Capture에 기여" |
| Down Capture > 110% | Adaptability D등급 | "하락장 방어 실패 = 적응력 부족이 원인" |
| TE < 3% | Conviction C등급 | "사실상 패시브 전략 → 확신 투자 비중 높여 차별화 필요" |

### 6.7 Black-Litterman + Efficient Frontier 교차

| BL 조건 | Frontier 조건 | 교차 인사이트 |
|---------|-------------|-------------|
| 최적 비중 괴리 > 15% | 프론티어 gap > 0.3 | "BL과 Markowitz 모두 비중 조정 권장 → 리밸런싱 근거 강함" |
| 최적 비중 ≈ 현재 | 프론티어 근접 | "두 모델 모두 현재 비중 적절 → 안정적 상태" |

---

## 7. 인사이트 품질 규칙

### 7.1 금지 사항
- "투자하세요", "매수하세요" 등 **직접적 투자 권유 금지**
- "반드시 수익이 납니다" 등 **확정적 수익 보장 금지**
- 특정 종목에 대한 **매수/매도 추천 금지**

### 7.2 필수 사항
- 모든 인사이트는 **데이터 기반** (수치 근거 제시)
- 진단 후 반드시 **구체적 개선 방향** 제시 (액션 가능)
- 조건부 표현 사용: "~할 수 있습니다", "~를 검토해보세요", "~를 고려하세요"

### 7.3 톤 & 매너
- 전문적이되 딱딱하지 않게
- 부정적 진단도 **건설적으로** 표현
- 숫자와 함께 직관적 비유 활용 가능
  (예: "MDD 35%는 원금의 1/3이 한때 사라졌다는 의미입니다")

---

## 8. 면책 조항 규칙

모든 인사이트 영역과 PDF 리포트에 반드시 포함:

```
⚠️ 본 분석은 교육/연구 목적이며, 투자 조언을 구성하지 않습니다.
모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
과거 성과가 미래 수익을 보장하지 않습니다.
```
