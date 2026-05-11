# Skills_visualization.md — 시각화 및 대시보드 구성 규칙

## 1. 시각화 선택 기준

이 문서는 투자 분석 결과를 어떤 차트로, 어떻게 시각화할지의 규칙을 정의한다.

### 1.1 데이터 유형별 차트 선택 매트릭스

| 데이터 유형 | 차트 | 라이브러리 | 이유 |
|-------------|------|------------|------|
| 6-Skills 점수 | **Radar Chart** | `go.Scatterpolar` | 다차원 역량을 한눈에 비교 |
| Skills 개별 점수 | **수평 Bar Chart** | `go.Bar(orientation='h')` | 등급별 색상으로 직관적 비교 |
| 시계열 수익률 | **Line Chart** | `go.Scatter` | 시간 흐름에 따른 추세 파악 |
| 낙폭(Drawdown) | **Area Chart (음수)** | `go.Scatter(fill='tozeroy')` | 손실 구간을 면적으로 강조 |
| 월별 수익률 | **Heatmap** | `go.Heatmap` | Year×Month 패턴 탐색 |
| 섹터/종목 비중 | **Treemap** | `go.Treemap` | 계층적 비중을 면적으로 표현 |
| 리스크 vs 수익 | **Bubble Scatter** | `go.Scatter(mode='markers')` | 3변수(x=변동성, y=수익, size=비중) 동시 표현 |
| 시뮬레이션 경로 | **Fan Chart** | `go.Scatter(fill='toself')` | 확률 밴드로 불확실성 표현 |
| 시뮬레이션 분포 | **Histogram** | `go.Histogram` | 최종 가치 분포 형태 파악 |

### 1.2 고급 분석 차트 선택 매트릭스

| 데이터 유형 | 차트 | 라이브러리 | 이유 |
|-------------|------|------------|------|
| 매크로 충격 전파 | **Waterfall Chart** | `go.Waterfall` | 각 변수의 기여를 누적으로 표현 |
| 팩터 기여도 | **Donut Chart** | `go.Pie(hole=0.5)` | 4팩터 비중을 직관적 표현 |
| 알파 유의성 | **Gauge Chart** | `go.Indicator(mode="gauge")` | t-stat → 신뢰도 시각화 |
| 효율적 프론티어 | **Colored Scatter** | `go.Scatter(colorscale)` | Sharpe 기준 색상 + 최적점 마커 |
| 종목 상관관계 | **Network Graph** | `go.Scatter(lines+markers)` | 노드=종목, 엣지=상관관계 |
| 시나리오×섹터 | **Heatmap (diverging)** | `go.Heatmap` | 2차원 영향도 매트릭스 |
| Skills 시간변화 | **Multi-line + Bands** | `go.Scatter + hrect` | 등급 밴드 오버레이 |
| 포트폴리오 비교 | **Overlay Radar** | 복수 `go.Scatterpolar` | 투명 영역 겹쳐 비교 |
| 위기 경로 시뮬레이션 | **Line + Area** | `go.Scatter(fill)` | 최저점 마커 + 기준선 |
| 회복 타임라인 | **Stacked Bar** | `go.Bar(barmode='stack')` | 위기+회복 기간 표현 |

### 1.3 차트 선택 금지 규칙
- **파이 차트 사용 금지**: 비중 비교에 파이 차트 대신 Treemap 사용 (연구에 따르면 파이 차트는 비중 비교에 부정확)
- **3D 차트 사용 금지**: 가독성 저하, 2D로 충분히 표현 가능
- **표(Table)만으로 분석 결과 제시 금지**: 반드시 시각적 차트와 함께 제시

---

## 2. 컬러 시스템

### 2.1 메인 팔레트
```python
COLORS = {
    "primary": "#6366F1",      # Indigo — 주 색상
    "secondary": "#8B5CF6",    # Violet — 보조
    "accent": "#EC4899",       # Pink — 강조
    "positive": "#10B981",     # Emerald — 수익/양수
    "negative": "#EF4444",     # Red — 손실/음수
    "neutral": "#6B7280",      # Gray — 벤치마크/참조
    "background": "#0F172A",   # Slate dark — 배경
    "surface": "#1E293B",      # Slate — 카드/패널
    "text": "#F8FAFC",         # Slate light — 텍스트
    "grid": "#334155",         # Slate — 그리드선
}
```

### 2.2 등급별 색상
```python
GRADE_COLORS = {
    "S": "#10B981",  # Emerald
    "A": "#6366F1",  # Indigo
    "B": "#F59E0B",  # Amber
    "C": "#F97316",  # Orange
    "D": "#EF4444",  # Red
}
```

### 2.3 Skill별 고유 색상
```python
SKILL_COLORS = {
    "Timing": "#6366F1",
    "Diversification": "#8B5CF6",
    "Risk Management": "#EC4899",
    "Conviction": "#F59E0B",
    "Adaptability": "#10B981",
    "Consistency": "#06B6D4",
}
```

### 2.4 수익률 색상 스케일
- 양수 → 음수: `RdYlGn` colorscale (Plotly 내장)
- 중앙값(0) 기준 대칭: `zmid=0` 설정 필수

---

## 3. 차트별 상세 구현 규칙

### 3.1 Skills Radar Chart
```
필수 요소:
- 6개 축 (Timing, Diversification, Risk Mgmt, Conviction, Adaptability, Consistency)
- 값 범위: 0~100
- 채움 영역: rgba(99, 102, 241, 0.25)
- 라인 두께: 3px
- 마커 크기: 10px
- 각 점에 점수 텍스트 표시
- 배경에 등급 구간 표시 (S=90, A=75, B=55, C=35)
- 높이: 500px
```

### 3.2 Cumulative Returns
```
필수 요소:
- 포트폴리오: 실선, primary 색상, 2.5px
- 벤치마크: 점선, neutral 색상, 1.5px
- Y축: 퍼센트 표시 (ticksuffix="%")
- hover: 날짜 + 수익률
- hovermode: "x unified"
- 높이: 400px
```

### 3.3 Drawdown Chart
```
필수 요소:
- 음수 영역 채우기: rgba(239, 68, 68, 0.3)
- MDD 포인트에 주석(annotation) 표시: "MDD: -XX.X%"
- 화살표 + 배경 박스 주석 스타일
- 높이: 300px
```

### 3.4 Monthly Heatmap
```
필수 요소:
- 행: 연도 (최신 위)
- 열: 월 (Jan~Dec)
- 셀 내부 텍스트: 수익률 소수점 1자리 + %
- 색상: RdYlGn, zmid=0
- X축: 상단 배치
- 높이: 300px
```

### 3.5 Sector Treemap
```
필수 요소:
- 계층: Portfolio → Sector → Ticker
- 면적: 비중(weight) 비례
- 색상: 수익률 기반 RdYlGn
- 라벨: 종목명 + 비중%
- hover: 종목명 + 비중 + 수익률
- 높이: 450px
```

### 3.6 Risk-Return Scatter
```
필수 요소:
- X축: 연환산 변동성 (std * sqrt(252) * 100)
- Y축: 총 수익률 (%)
- 버블 크기: 비중 * 3
- 색상: 수익률 기반 RdYlGn
- 텍스트: 종목 코드 (top center)
- 높이: 400px
```

### 3.7 Monte Carlo Fan Chart
```
필수 요소:
- 5th~95th 백분위 밴드: 연한 primary (0.1 투명도)
- 25th~75th 밴드: 중간 primary (0.2 투명도)
- 중앙값 경로: 실선 primary, 2.5px
- 시작점 수평선: dash, neutral 색상
- 최종값 주석: Median, 5th, 95th 표시
- 높이: 450px
```

### 3.8 Skills Detail Bar
```
필수 요소:
- 수평 바, 등급별 색상 적용
- 점수 텍스트: 바 외부(outside)에 표시
- 등급 구간 세로 점선: 35, 55, 75, 90 위치
- 세로선에 등급 경계 라벨: "D|C", "C|B", "B|A", "A|S"
- X축 범위: 0~105
- 높이: 350px
```

---

## 4. 레이아웃 규칙

### 4.1 공통 레이아웃 설정
모든 Plotly 차트에 적용:
```python
LAYOUT_DEFAULTS = {
    "template": "plotly_dark",
    "paper_bgcolor": "#0F172A",
    "plot_bgcolor": "#0F172A",
    "font": {"family": "Inter, sans-serif", "color": "#F8FAFC"},
    "margin": {"l": 40, "r": 40, "t": 60, "b": 40},
}
```

### 4.2 타이틀 규칙
- 위치: 중앙 (`x=0.5`)
- 크기: 18px (일반), 20px (메인 레이더)
- 폰트: `LAYOUT_DEFAULTS.font`와 동일

### 4.3 그리드 규칙
- 색상: `#334155`
- X/Y 축 모두 표시
- zeroline: 필요시만 (드로다운 차트)

---

## 5. 인터랙션 규칙

### 5.1 Hover
- 모든 차트에 hover 정보 필수
- `hovertemplate` 사용 (기본 hover 금지)
- 날짜 포맷: `%Y-%m-%d`
- 수익률: 소수점 1자리 + %
- 금액: 천단위 콤마 + $

### 5.2 사이드바 인터랙션
- 데이터 소스 전환: `st.radio`
- 기간: `st.date_input` 2개 (시작/종료)
- 벤치마크: `st.selectbox`
- Monte Carlo 파라미터: `st.slider`
- 실행 버튼: `st.button(type="primary")`

### 5.3 로딩 상태
- 데이터 수집 시: `st.spinner("📡 데이터 수집 중...")`
- 분석 시: `st.spinner("🧠 Skills 분석 중...")`
- 분석 완료 후 결과를 `st.session_state`에 캐싱

---

## 6. 카드 컴포넌트 규칙

### 6.1 KPI 카드
- `st.metric()` 사용
- 배경: `#1E293B`
- 패딩: 16px
- 테두리: `1px solid #334155`
- 라운드: 12px

### 6.2 Skill 상세 카드
```html
<div style="background: #1E293B; padding: 16px; border-radius: 12px; border: 1px solid #334155;">
    <h4>{Skill 이름}</h4>
    <span class="badge {grade}">{등급} ({점수})</span>
    <p>{등급별 설명}</p>
</div>
```
- 3열 그리드 배치
- 각 카드에 `st.expander("상세 지표")` 포함

### 6.3 등급 뱃지 CSS
```css
.skill-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
}
.grade-S { background: #10B981; color: white; }
.grade-A { background: #6366F1; color: white; }
.grade-B { background: #F59E0B; color: black; }
.grade-C { background: #F97316; color: white; }
.grade-D { background: #EF4444; color: white; }
```

---

## 7. 면책 조항

대시보드 하단에 반드시 포함:
```
⚠️ 본 대시보드는 교육/분석 목적이며, 투자 조언을 제공하지 않습니다.
투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
```
