# Skills.md — 투자 분석 대시보드 메인 규칙

## 1. 서비스 정의

이 문서는 AI 바이브 코딩을 통해 **투자 포트폴리오 분석 대시보드**를 구현하기 위한 핵심 규칙을 정의한다.
대시보드 이름: **InvestScope — 투자 역량 분석 대시보드**

### 1.1 서비스 목적
- 사용자가 보유한 포트폴리오(주식/ETF)를 입력하면, 6가지 투자 역량(Skills) 관점에서 정량 분석 결과를 시각화
- 단순 수익률 나열이 아닌, **왜 이 성과가 나왔는지**를 역량 기반으로 진단

### 1.2 핵심 컨셉: 6-Skills 프레임워크
모든 분석은 아래 6가지 투자 역량 축을 중심으로 수행한다:

| Skill | 정의 | 핵심 질문 |
|-------|------|-----------|
| **Timing** | 매수/매도 타이밍 역량 | "적절한 시점에 진입/퇴출했는가?" |
| **Diversification** | 분산투자 역량 | "리스크가 충분히 분산되어 있는가?" |
| **Risk Management** | 리스크 관리 역량 | "하방 리스크를 잘 통제하고 있는가?" |
| **Conviction** | 확신 포지션 운용 역량 | "집중 투자한 종목이 실제로 성과를 냈는가?" |
| **Adaptability** | 시장 변화 적응력 | "시장 하락기에 얼마나 잘 방어했는가?" |
| **Consistency** | 수익 일관성 | "안정적으로 꾸준히 수익을 내고 있는가?" |

---

## 2. 기술 스택 규칙

### 2.1 필수 기술
- **프레임워크**: Streamlit (Python)
- **시각화**: Plotly (인터랙티브 차트 필수)
- **데이터 처리**: pandas, numpy
- **배포**: Streamlit Community Cloud (무료, 외부 접근 가능)

### 2.2 선택 기술
- 금융 데이터 수집: yfinance (무료, API 키 불필요)
- 통계 분석: scipy (Sharpe, 정규성 검정 등)
- PDF 생성: reportlab (리포트 다운로드)
- ML: scikit-learn (클러스터링, 이상치 탐지)

### 2.3 금지 사항
- 외부 API 키가 필요한 서비스 사용 금지 (심사자가 키 없이 접근해야 함)
- 유료 데이터 소스 사용 금지
- 로그인/인증이 필요한 구조 금지

---

## 3. 데이터 규칙

### 3.1 데이터 소스 우선순위
1. **yfinance** — 글로벌 주식/ETF 실시간 데이터 (1순위)
2. **내장 더미 데이터** — yfinance 실패 시 자동 전환 (2순위, 필수 구현)
3. **CSV 업로드** — 사용자 커스텀 포트폴리오 (3순위)
4. **🇰🇷 한국 주식** — KRX 80+ 종목 한글 매핑, 4개 프리셋 포트폴리오 (국민, 2차전지, 배당, 성장)

### 3.2 더미 데이터 생성 규칙
yfinance 접근 불가 시에도 대시보드가 완전히 동작해야 한다:
- Cholesky decomposition으로 상관관계 있는 멀티에셋 수익률 생성
- Fat tail 특성 포함 (정규분포 + 간헐적 큰 하락)
- 최소 2년(504 거래일) 데이터
- OHLCV 형식 준수

### 3.3 전처리 규칙
- 결측치: forward fill → backward fill 순서
- 수익률 계산: 산술 수익률 `(P_t - P_{t-1}) / P_{t-1}` 사용
- 수정주가(Adjusted Close) 우선 사용
- 비중 합계는 항상 1.0으로 정규화

---

## 4. 대시보드 구성 규칙

### 4.1 레이아웃 구조 (v6.0 — 23탭 · 6카테고리)

```
[사이드바]                    [메인 영역]
├─ 데이터 소스 선택           ├─ 헤더: 서비스명 + 설명
├─ 포트폴리오 입력            ├─ KPI 카드 (6개 요약 지표)
├─ 기간 설정                  ├─ 카테고리 라디오 버튼 (수평 6개)
├─ 벤치마크 선택              ├─ 선택 카테고리 내 탭들 (동적 렌더)
├─ Monte Carlo 설정           └─ 푸터: 면책 조항
└─ [분석 실행] 버튼
```

**6카테고리 네비게이션:**

| 카테고리 | 탭 | 역할 |
|----------|-----|------|
| 📊 기본 분석 | Skills · Performance · Allocation · Simulation | 현황 진단 |
| 🔮 시나리오 | What-If · Stress Test · Geopolitical | 미래/위기 시뮬레이션 |
| 📈 팩터 & 최적화 | Factor · Style · Frontier · Black-Litterman | 수익 원천 분석 + 최적화 |
| ⚡ 리스크 | Risk 기여도 · Tail Risk · GARCH · Regime | 리스크 정밀 분석 |
| 🔗 포트폴리오 진단 | Correlation · Compare · Rebalance · DNA | 구조 진단 + 리밸런싱 |
| 📋 성과 추적 | Evolution · Events · Backtest · Performance+ | 시계열 추적 + 백테스트 |

**전체 23개 탭 목록:**

```
📊 기본 분석
├─ 탭 1: 🎯 Skills 분석 (6-Skills 레이더 + AI 진단 + PDF)
├─ 탭 2: 📊 Performance (누적수익률, 드로다운, 히트맵)
├─ 탭 3: 🗺️ Allocation (섹터 트리맵, 리스크-리턴)
└─ 탭 4: 🔮 Simulation (Monte Carlo 팬차트)

🔮 시나리오
├─ 탭 5: 🔄 What-If 시나리오 (비중 변경 시뮬레이션)
├─ 탭 6: 🔥 Stress Test (역사적 위기 리플레이)
└─ 탭 7: 🌍 Geopolitical (지정학 충격 시뮬레이션)

📈 팩터 & 최적화
├─ 탭 8: 🧬 Factor Attribution (Fama-French 3-Factor)
├─ 탭 9: 🎨 Investment Style (Morningstar Style Box)
├─ 탭 10: 📐 Efficient Frontier (Markowitz 최적화)
└─ 탭 11: 🏦 Black-Litterman (투자자 뷰 최적화)

⚡ 리스크
├─ 탭 12: ⚡ Risk Contribution (Euler 리스크 분해)
├─ 탭 13: 🎲 Tail Risk (Cornish-Fisher VaR)
├─ 탭 14: 📉 GARCH (GARCH(1,1) 변동성 예측)
└─ 탭 15: 🔀 Regime Detection (시장 국면 탐지)

🔗 포트폴리오 진단
├─ 탭 16: 🕸️ Correlation Network (상관관계 네트워크)
├─ 탭 17: ⚖️ Multi-Portfolio Compare
├─ 탭 18: 🔔 Rebalance Signal (리밸런싱 시그널)
└─ 탭 19: 🧬 Portfolio DNA (DNA 핑거프린트)

📋 성과 추적
├─ 탭 20: 📈 Skills Evolution (롤링 윈도우)
├─ 탭 21: 📅 Market Events (시장 이벤트 타임라인)
├─ 탭 22: ⏪ Backtest (과거 시뮬레이션)
└─ 탭 23: 📋 Performance+ (IR, TE, Capture Ratio)
```

### 4.2 필수 KPI 카드 (상단 고정)
| 지표 | 계산식 | 형식 |
|------|--------|------|
| 총 수익률 | `(1+r).prod() - 1` | `+12.3%` |
| Alpha | 포트폴리오 수익률 - 벤치마크 수익률 | `+3.2%` |
| Sharpe Ratio | `(mean_excess / std) * sqrt(252)` | `1.45` |
| Max Drawdown | `max(peak - trough) / peak` | `-15.2%` |
| Overall Skill Score | 6개 Skills 평균 | `72 (A등급)` |
| Best/Worst Skill | 최고/최저 Skill 이름 | 텍스트 |

### 4.3 탭별 필수 차트

**탭 1 — Skills 분석:**
- Skills Radar Chart (6축 레이더, 필수)
- Skills 수평 바 차트 (등급별 색상)
- 각 Skill별 상세 카드 (점수 + 등급 뱃지 + 설명 + 세부 지표)

**탭 2 — 성과 분석:**
- 누적 수익률 (포트폴리오 vs 벤치마크, 라인 차트)
- 드로다운 차트 (MDD 포인트 주석)
- 월별 수익률 히트맵 (Year x Month 매트릭스)

**탭 3 — 자산 배분:**
- 섹터별 트리맵 (비중 + 수익률 컬러)
- 리스크-리턴 산점도 (버블 크기 = 비중)
- 종목별 상세 테이블

**탭 4 — 시뮬레이션:**
- Monte Carlo 팬 차트 (5th~95th 백분위)
- VaR / CVaR 지표 카드
- 시뮬레이션 결과 분포 히스토그램

**탭 5 — What-If 시나리오:**
- 4가지 프리셋 전략 (동일비중, 집중투자, 방어형, 모멘텀) + 수동 조절
- 비교 레이더 차트 (원본 vs 수정), 변화 워터폴 차트, 비중 비교 바 차트
- AI Insight: 향상/하락 Skills 자동 분석

**탭 6 — Factor Attribution:**
- Fama-French 3-Factor 분해 (Market, SMB, HML, Alpha)
- 도넛 차트, 팩터 노출 바, 누적 기여 라인, 알파 유의성 게이지
- 순수 numpy OLS 회귀 (외부 의존성 최소화)

**탭 7 — Skills Evolution:**
- 롤링 윈도우 기반 Skills 시계열 추적 (63~252거래일)
- 6개 Skills + Overall 라인 차트, 등급 밴드(S/A/B/C/D) 오버레이
- 가장 향상/하락한 Skill 자동 식별

**탭 1 추가 — AI 투자 진단:**
- 룰 기반 NLG 코멘터리 (API 키 불필요)
- 종합 평가, 강점/약점 리스트, 개선 제안, 리스크 경고
- PDF 리포트 다운로드 버튼 (reportlab)

**탭 13 — Investment Style:**
- Morningstar 3×3 Style Box (Value/Blend/Growth × Large/Mid/Small)
- 6축 스타일 레이더 (Value, Growth, Momentum, Quality, Dividend, Volatility)
- Style Drift 타임라인: 롤링 윈도우 기반 스타일 변화 추적
- 섹터 기반 + 수익률 패턴 기반 휴리스틱 분류

**탭 14 — Risk Contribution:**
- Euler 분해 (MRC: Marginal Risk Contribution, CRC: Component Risk Contribution)
- 리스크 기여 바 차트 (비중 vs 리스크 비교)
- 리스크 도넛 차트 (종목별 리스크 비중)
- 리스크 이상치 탐지: 비중 대비 과도한 리스크 종목 식별
- 분산효과 비율 (Diversification Ratio)

**탭 15 — Market Events:**
- 39개 글로벌 이벤트(2019~2025) 어노테이션 누적수익률 차트
- 이벤트 전후(5일, 20일) 포트폴리오 영향도 분석 테이블
- 카테고리별(crash/policy/geopolitical/recovery/earnings) 분류
- 이벤트 중요도 기반 자동 필터링

**탭 16 — Regime Detection:**
- 시장 국면 자동 탐지 (Bull/Bear/Sideways)
- 롤링 통계 기반 분류 (평균 수익률 + 변동성, 최소 10일 smoothing)
- 국면별 성과 분석: 연환산 수익률, Sharpe, MDD, 승률, 알파
- Regime Timeline: 누적수익률에 국면 배경색 오버레이
- 전이 확률 행렬 (Markov Transition Matrix)
- 국면 지속 기간 차트

**탭 17 — Tail Risk:**
- Cornish-Fisher VaR: 왜도·첨도 보정으로 Gaussian VaR보다 정확
- CVaR (Expected Shortfall): VaR 이하 평균 손실
- Jarque-Bera 정규성 검정: 수익률 분포 정규분포 적합도
- Q-Q Plot: Fat tail 시각적 확인
- Rolling VaR: 시계열 VaR + 실제 손실 breach 포인트
- 다중 신뢰수준(90/95/99%) 비교 테이블

**탭 18 — GARCH(1,1) 변동성 예측:**
- GARCH(1,1) 모델: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
- MLE (Maximum Likelihood Estimation) 파라미터 추정
- 조건부 변동성 시계열 차트 + 실제 수익률 오버레이
- 30일/60일/90일 미래 변동성 예측 (평균회귀)
- 변동성 터미널 밸류 (장기 균형 변동성)
- 파라미터 진단: α + β 지속성, 반감기 계산

**탭 19 — Black-Litterman 최적화:**
- 시장 균형 수익률 (Market Implied Returns) 역추출
- 투자자 뷰(View) 입력 → 사후 수익률 계산
- Black-Litterman 공식: E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
- 사전/사후 수익률 비교 바 차트
- 최적 비중 vs 현재 비중 비교
- 확신도(confidence) 조절 슬라이더

**탭 20 — Rebalance Signal (리밸런싱 시그널):**
- 5가지 리밸런싱 트리거: Drift, Momentum, Volatility, Correlation, Calendar
- 종합 리밸런싱 긴급도 게이지 (0~100)
- 종목별 목표 비중 vs 현재 비중 드리프트 분석
- 리밸런싱 액션 테이블 (매수/매도/유지)
- 트리거별 상세 점수 + 가중 합산 로직

**탭 21 — Portfolio DNA 핑거프린트:**
- 12차원 DNA 레이더 차트 (6 Skills + Value/Growth/Momentum/Volatility/Concentration/Sector Diversity)
- 포트폴리오 아키타입 자동 분류 (예: "안정형 가치투자자", "공격형 모멘텀 트레이더")
- DNA 차원별 상세 테이블 (스킬/스타일/구조/레짐)
- DNA Insight: 아키타입 요약, 스킬 강약점, 집중도/섹터/스타일/변동성 진단
- DNA 해시 (고유 핑거프린트 ID)

**탭 22 — Backtest Engine:**
- Single Backtest: 특정 시점부터 현재 비중으로 과거 투자 시뮬레이션
- Rolling Backtest: 여러 진입 시점 × 여러 보유기간 조합 매트릭스
- 벤치마크 대비 누적수익률 비교
- 연환산 수익률, Sharpe, MDD, 승률 등 핵심 지표
- MultiIndex DataFrame 자동 처리 (yfinance 호환)

**탭 23 — Performance+ (고급 성과 지표):**
- Information Ratio (IR): 초과수익률 / Tracking Error
- Tracking Error (TE): 벤치마크 대비 추적 오차
- Up Capture Ratio: 상승장 포착 비율
- Down Capture Ratio: 하락장 방어 비율
- Capture Spread: Up - Down (양수일수록 우수)
- 종합 성과 등급 게이지 + 벤치마크 대비 롤링 IR 차트

---

## 5. UI/UX 규칙

### 5.1 디자인 시스템
- **테마**: 다크 모드 기본 (금융 대시보드 관례)
- **배경색**: `#0F172A` (Slate 900)
- **서피스**: `#1E293B` (Slate 800)
- **텍스트**: `#F8FAFC` (Slate 50)
- **주 색상**: `#6366F1` (Indigo 500)
- **양수**: `#10B981` (Emerald 500)
- **음수**: `#EF4444` (Red 500)

### 5.2 차트 스타일 규칙
- 모든 차트는 `plotly_dark` 템플릿 기반
- 배경색은 대시보드와 동일하게 통일
- hover 시 상세 정보 표시 필수
- 축 라벨, 범례 항상 포함
- 숫자 포맷: 퍼센트는 소수점 1자리, 비율은 소수점 2자리

### 5.3 반응형 규칙
- `st.columns()`로 2열 레이아웃 기본
- 모바일 대응 불필요 (데스크톱 심사 기준)
- 차트 높이: 레이더 500px, 라인 400px, 히트맵 300px

### 5.4 등급 시스템
| 점수 범위 | 등급 | 색상 | 의미 |
|-----------|------|------|------|
| 90~100 | S | `#10B981` | 최상위 |
| 75~89 | A | `#6366F1` | 우수 |
| 55~74 | B | `#F59E0B` | 평균 이상 |
| 35~54 | C | `#F97316` | 개선 필요 |
| 0~34 | D | `#EF4444` | 역량 부족 |

---

## 6. 에러 처리 규칙

### 6.1 데이터 수집 실패
```
yfinance 실패 → 자동으로 합성 데이터 전환 → 사용자에게 알림 배너 표시
```

### 6.2 빈 포트폴리오
```
종목 미입력 → 샘플 포트폴리오 자동 로드 → "샘플 데이터입니다" 안내
```

### 6.3 비중 합계 불일치
```
합계 ≠ 1.0 → 자동 정규화 후 진행 → 사이드바에 정규화 적용 알림
```

---

## 7. 파일 구조 규칙

```
project/
├── app.py                     # Streamlit 엔트리포인트 (23탭, 6카테고리 대시보드)
├── requirements.txt           # 의존성 (streamlit, plotly, pandas, numpy, yfinance, scipy, reportlab, arch)
├── src/
│   ├── data_pipeline.py       # 데이터 수집/전처리/합성 데이터 생성
│   ├── skills_engine.py       # 6-Skills 산출 로직
│   ├── visualizations.py      # Plotly 차트 생성 함수
│   ├── whatif_engine.py       # What-If 시나리오 엔진 (4 프리셋 전략)
│   ├── factor_attribution.py  # Fama-French 3-Factor 분석
│   ├── kr_stocks.py           # 한국 주식 KRX 매핑 (80+ 종목)
│   ├── ai_commentary.py       # 룰 기반 NLG 투자 진단
│   ├── efficient_frontier.py  # Markowitz 효율적 프론티어
│   ├── stress_test.py         # 역사적 위기 스트레스 테스트
│   ├── geopolitical_engine.py # 지정학 시나리오 (매크로 충격 전파)
│   ├── correlation_network.py # 상관관계 네트워크 + 클러스터링
│   ├── multi_portfolio.py     # 멀티 포트폴리오 비교
│   ├── risk_contribution.py   # Euler 리스크 기여도 분해 (MRC/CRC)
│   ├── style_analysis.py      # 투자 스타일 분류 (Morningstar Style Box)
│   ├── market_events.py       # 시장 이벤트 타임라인 (39개 글로벌 이벤트)
│   ├── regime_detection.py    # 시장 국면 탐지 (Bull/Bear/Sideways)
│   ├── tail_risk.py           # Cornish-Fisher VaR + 꼬리 리스크 분석
│   ├── garch_model.py         # GARCH(1,1) 변동성 예측 모델
│   ├── black_litterman.py     # Black-Litterman 포트폴리오 최적화
│   ├── rebalance_signal.py    # 5-Trigger 리밸런싱 시그널 엔진
│   ├── portfolio_dna.py       # 12차원 Portfolio DNA 핑거프린트
│   ├── backtest_engine.py     # Single/Rolling 백테스트 엔진
│   └── performance_metrics.py # Performance+ 고급 성과 지표 (IR, TE, Capture)
├── skills/
│   ├── Skills_main.md              # 이 파일 (메인 규칙 — 전체 아키텍처 + 탭 구조)
│   ├── Skills_analysis.md          # 기본 6-Skills 산출 규칙
│   ├── Skills_visualization.md     # 시각화 선택 기준 + 컬러 시스템
│   ├── Skills_advanced_analysis.md # 고급 분석 규칙 (Factor, Frontier, Stress, Geo, Corr, GARCH, BL, Rebalance, DNA, Backtest, Perf+)
│   └── Skills_insight_report.md    # 인사이트 생성 + 리포트 구성 규칙
└── docs/
    ├── planning.pdf           # 기획서 (영문)
    └── planning_kr.pdf        # 기획서 (한국어, DACON 대회 제출용)
```
