# InvestScope — 투자 역량 분석 대시보드

> DACON Monthly Hackathon | 6-Skills Framework 기반 투자 포트폴리오 분석 대시보드

## 프로젝트 개요

InvestScope는 투자 포트폴리오를 **6가지 투자 역량(Skills)** 관점에서 정량 분석하는 대시보드입니다.
23개 분석 탭 | 6개 카테고리 | 5개 Skills.md 규칙 정의서

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 6-Skills Framework

| Skill | 측정 대상 | 핵심 지표 |
|-------|---------|---------|
| Timing | 매수/매도 타이밍 | 30일 롤링 수익률 기반 |
| Diversification | 분산투자 역량 | HHI + Entropy |
| Risk Management | 리스크 관리 | Sharpe, Sortino, MDD |
| Conviction | 확신 포지션 운용 | Top-5 Alpha |
| Adaptability | 시장 변화 적응력 | 하락장 상대 성과 |
| Consistency | 수익 일관성 | 승률, CV, streak |

## 기술 스택

- **프레임워크**: Streamlit (Python)
- **시각화**: Plotly (인터랙티브 차트)
- **데이터**: yfinance (무료, API 키 불필요)
- **분석**: pandas, numpy, scipy
- **PDF**: reportlab (리포트 생성)

## 프로젝트 구조

```
investment-dashboard/
├── app.py                 # Streamlit 메인앱 (23탭, 6카테고리)
├── requirements.txt       # 의존성 목록
├── src/                   # 분석 엔진 모듈 (22개)
│   ├── skills_engine.py   # 6-Skills 산출
│   ├── data_pipeline.py   # 데이터 수집/전처리
│   ├── visualizations.py  # Plotly 차트
│   ├── factor_attribution.py  # Fama-French 3-Factor
│   ├── garch_model.py     # GARCH(1,1) 변동성
│   ├── black_litterman.py # Black-Litterman 모델
│   ├── portfolio_dna.py   # 12차원 DNA 핑거프린트
│   ├── backtest_engine.py # 백테스트 엔진
│   └── ...                # 외 13개 모듈
├── skills/                # Skills.md 규칙 정의서 (5개)
└── .streamlit/config.toml # 테마 설정
```

## 데이터 소스

- **실시간**: yfinance API (미국 주식/ETF, 키 불필요)
- **한국 주식**: KRX 80+ 종목 내장 (4개 섹터)
- **직접 입력**: 종목/비중 수동 입력 또는 CSV 업로드
- **합성 데이터**: 오프라인 데모용 통계적 시뮬레이션

## 참가자

- 이우창 (Woochang Lee)
- 2026년 4월
