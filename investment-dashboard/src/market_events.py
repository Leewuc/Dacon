"""
Market Event Timeline: 주요 시장 이벤트 자동 어노테이션
- 누적수익률 차트에 글로벌/한국 시장 이벤트 오버레이
- 이벤트 전후 수익률 분석
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MarketEvent:
    """시장 이벤트 정의"""
    date: str              # "2020-03-23"
    name: str              # "COVID-19 저점"
    category: str          # "crash", "recovery", "policy", "geopolitical", "earnings"
    description: str       # 상세 설명
    icon: str              # emoji
    impact: str            # "negative", "positive", "neutral"


# Global events database (2019-2026)
MARKET_EVENTS: List[MarketEvent] = [
    # 2019
    MarketEvent("2019-06-04", "미중 무역전쟁 격화", "geopolitical",
                "미국, 대중 관세 25% 부과. 글로벌 공급망 우려", "🇺🇸🇨🇳", "negative"),
    MarketEvent("2019-08-05", "위안화 7 돌파", "policy",
                "인민은행 위안화 절하 허용. 환율전쟁 우려", "💱", "negative"),
    MarketEvent("2019-10-11", "미중 1단계 합의", "geopolitical",
                "미중 무역 1단계 합의 발표. 시장 반등", "🤝", "positive"),

    # 2020
    MarketEvent("2020-01-20", "COVID-19 첫 확진", "geopolitical",
                "WHO, 코로나19 국제적 비상사태 선포", "🦠", "negative"),
    MarketEvent("2020-03-09", "유가 전쟁 + 서킷브레이커", "crash",
                "사우디-러시아 유가 전쟁. S&P500 -7.6% 서킷브레이커", "📉", "negative"),
    MarketEvent("2020-03-23", "COVID 저점", "crash",
                "S&P500 저점. 2,237pt. 고점 대비 -34%", "💥", "negative"),
    MarketEvent("2020-03-27", "CARES Act 서명", "policy",
                "미국 2.2조 달러 경기부양법 서명", "💰", "positive"),
    MarketEvent("2020-08-31", "KOSPI 동학개미 2,300", "recovery",
                "한국 개인투자자 열풍. KOSPI 2,300 돌파", "🇰🇷", "positive"),
    MarketEvent("2020-11-09", "화이자 백신 발표", "recovery",
                "화이자-바이오엔텍 코로나 백신 95% 효과 발표", "💉", "positive"),

    # 2021
    MarketEvent("2021-01-27", "GameStop 숏스퀴즈", "crash",
                "레딧 WallStreetBets의 GME 숏스퀴즈. 밈주 열풍", "🎮", "neutral"),
    MarketEvent("2021-02-26", "미 국채금리 1.5% 급등", "policy",
                "10년물 금리 급등. 성장주 급락", "📈", "negative"),
    MarketEvent("2021-05-19", "비트코인 30% 폭락", "crash",
                "중국 암호화폐 규제. BTC $30K 하회", "₿", "negative"),
    MarketEvent("2021-09-20", "헝다그룹 위기", "crash",
                "중국 부동산 대기업 헝다그룹 디폴트 위기", "🏗️", "negative"),
    MarketEvent("2021-11-26", "오미크론 변이", "geopolitical",
                "남아공 오미크론 변이 발견. 글로벌 매도", "🦠", "negative"),

    # 2022
    MarketEvent("2022-02-24", "러시아 우크라이나 침공", "geopolitical",
                "러시아 우크라이나 전면 침공. 유럽 에너지 위기", "⚔️", "negative"),
    MarketEvent("2022-03-17", "Fed 첫 금리인상", "policy",
                "Fed 0.25%p 인상 시작. 제로금리 시대 종료", "🏦", "negative"),
    MarketEvent("2022-06-13", "나스닥 베어마켓", "crash",
                "나스닥 고점 대비 -33%. 공식 베어마켓", "🐻", "negative"),
    MarketEvent("2022-09-28", "영국 미니 예산 위기", "policy",
                "트러스 총리 감세 정책으로 파운드/길트 폭락", "🇬🇧", "negative"),
    MarketEvent("2022-11-11", "FTX 파산", "crash",
                "암호화폐 거래소 FTX 파산. 암호화폐 시장 충격", "💣", "negative"),

    # 2023
    MarketEvent("2023-01-30", "ChatGPT 열풍", "recovery",
                "AI 테마 본격화. NVIDIA, Microsoft 등 급등", "🤖", "positive"),
    MarketEvent("2023-03-10", "SVB 은행 파산", "crash",
                "실리콘밸리은행 파산. 미국 지역은행 위기", "🏦", "negative"),
    MarketEvent("2023-05-24", "NVIDIA 실적 폭발", "earnings",
                "NVIDIA Q1 실적 발표. AI 칩 수요 폭증. 시가총액 1조 돌파", "🚀", "positive"),
    MarketEvent("2023-07-12", "인플레 완화 확인", "policy",
                "미국 CPI 3.0%로 하락. 금리인상 종료 기대", "📊", "positive"),
    MarketEvent("2023-10-07", "이스라엘-하마스 전쟁", "geopolitical",
                "하마스 기습 공격. 중동 지정학 리스크 급등", "⚔️", "negative"),

    # 2024
    MarketEvent("2024-01-02", "일본 노토 지진", "geopolitical",
                "노토반도 7.6 지진. 일본 시장 하락", "🌊", "negative"),
    MarketEvent("2024-03-05", "BTC 신고가", "recovery",
                "비트코인 $69K 돌파 역대 신고가", "₿", "positive"),
    MarketEvent("2024-07-11", "엔캐리 트레이드 해소", "crash",
                "엔화 급등 + 캐리 트레이드 청산. 일본 -12.4% 급락", "🇯🇵", "negative"),
    MarketEvent("2024-09-18", "Fed 첫 금리인하", "policy",
                "Fed 0.5%p 빅컷 인하. 피벗 시작", "📉", "positive"),
    MarketEvent("2024-11-05", "트럼프 재선", "geopolitical",
                "트럼프 대선 승리. 감세+규제완화 기대", "🇺🇸", "positive"),

    # 2025
    MarketEvent("2025-01-20", "DeepSeek 충격", "crash",
                "중국 AI 스타트업 DeepSeek. 미국 AI 주도권 위협. NVIDIA -17%", "🇨🇳", "negative"),
    MarketEvent("2025-02-04", "트럼프 관세 전쟁 2.0", "geopolitical",
                "미국, 중국·멕시코·캐나다에 대규모 관세 부과", "📦", "negative"),
    MarketEvent("2025-04-02", "상호관세 해방의 날", "geopolitical",
                "트럼프 상호관세 발표. 글로벌 무역 충격", "🌐", "negative"),
    MarketEvent("2025-04-09", "관세 90일 유예", "policy",
                "트럼프 상호관세 90일 유예 발표. S&P500 +9.5% 반등", "🕊️", "positive"),
]


def get_events_in_range(
    start_date: str,
    end_date: str,
) -> List[MarketEvent]:
    """주어진 기간 내의 이벤트 필터링"""
    events = []
    for event in MARKET_EVENTS:
        if start_date <= event.date <= end_date:
            events.append(event)
    return events


def create_annotated_returns_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_name: str = "Portfolio",
    benchmark_name: str = "Benchmark",
    max_annotations: int = 8,
) -> "go.Figure":
    """
    누적수익률 차트에 시장 이벤트 어노테이션 추가

    Rules:
    - 기간 내 이벤트만 표시
    - max_annotations 초과 시 영향이 큰 이벤트 우선
    - crash/policy는 세로 점선으로 표시
    - 양수 이벤트: 초록 마커, 음수: 빨강 마커
    - hover로 상세 설명
    """
    import plotly.graph_objects as go

    # Cumulative returns
    cum_port = (1 + portfolio_returns).cumprod() * 100
    cum_bench = (1 + benchmark_returns).cumprod() * 100

    # Get date range
    start_date = str(portfolio_returns.index[0])[:10]
    end_date = str(portfolio_returns.index[-1])[:10]
    events = get_events_in_range(start_date, end_date)

    # Limit events (prioritize crash and policy)
    priority = {"crash": 0, "policy": 1, "geopolitical": 2, "recovery": 3, "earnings": 4, "neutral": 5}
    events.sort(key=lambda e: priority.get(e.category, 5))
    events = events[:max_annotations]

    fig = go.Figure()

    # Portfolio line
    fig.add_trace(go.Scatter(
        x=cum_port.index, y=cum_port.values,
        name=portfolio_name, mode="lines",
        line=dict(color="#6366F1", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: %{y:.1f}<extra></extra>",
    ))

    # Benchmark line
    fig.add_trace(go.Scatter(
        x=cum_bench.index, y=cum_bench.values,
        name=benchmark_name, mode="lines",
        line=dict(color="#6B7280", width=1.5, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Benchmark: %{y:.1f}<extra></extra>",
    ))

    # Event annotations
    impact_colors = {"negative": "#EF4444", "positive": "#10B981", "neutral": "#F59E0B"}

    for event in events:
        event_date = pd.Timestamp(event.date)

        # Find closest date in index
        if event_date in cum_port.index:
            y_val = cum_port.loc[event_date]
        else:
            # Find nearest date
            closest_idx = cum_port.index.get_indexer([event_date], method='nearest')[0]
            if closest_idx >= 0 and closest_idx < len(cum_port):
                y_val = cum_port.iloc[closest_idx]
                event_date = cum_port.index[closest_idx]
            else:
                continue

        color = impact_colors.get(event.impact, "#F59E0B")

        # Vertical line
        fig.add_vline(
            x=event_date, line_dash="dot",
            line_color=color, line_width=1, opacity=0.5,
        )

        # Marker with hover
        fig.add_trace(go.Scatter(
            x=[event_date], y=[y_val],
            mode="markers+text",
            marker=dict(color=color, size=10, symbol="diamond",
                       line=dict(color="white", width=1)),
            text=[event.icon],
            textposition="top center",
            textfont=dict(size=14),
            hovertemplate=(
                f"<b>{event.icon} {event.name}</b><br>"
                f"{event.date}<br>"
                f"{event.description}<br>"
                f"Portfolio: %{{y:.1f}}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title="Cumulative Returns with Market Events",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=500,
        xaxis_title="Date",
        yaxis_title="Cumulative Value (Start=100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                   xanchor="center", x=0.5),
    )

    return fig


def create_event_impact_table(
    events: List[MarketEvent],
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """
    이벤트 전후 포트폴리오 수익률 분석 테이블
    - 이벤트 전 5일, 후 5일, 후 20일 수익률
    """
    rows = []
    for event in events:
        event_date = pd.Timestamp(event.date)

        # Find the event's position in the index
        if event_date not in portfolio_returns.index:
            closest = portfolio_returns.index.get_indexer([event_date], method='nearest')[0]
            if closest < 0 or closest >= len(portfolio_returns):
                continue
            event_date = portfolio_returns.index[closest]

        idx = portfolio_returns.index.get_loc(event_date)

        # Pre-event (5 days before)
        pre_start = max(0, idx - 5)
        pre_ret = (1 + portfolio_returns.iloc[pre_start:idx]).prod() - 1

        # Post-event (5 days after)
        post_5 = min(len(portfolio_returns), idx + 6)
        post_ret_5 = (1 + portfolio_returns.iloc[idx:post_5]).prod() - 1

        # Post-event (20 days after)
        post_20 = min(len(portfolio_returns), idx + 21)
        post_ret_20 = (1 + portfolio_returns.iloc[idx:post_20]).prod() - 1

        impact_emoji = {"negative": "🔴", "positive": "🟢", "neutral": "🟡"}

        rows.append({
            "이벤트": f"{event.icon} {event.name}",
            "날짜": event.date,
            "영향": impact_emoji.get(event.impact, "⚪"),
            "직전 5일": f"{pre_ret*100:+.1f}%",
            "직후 5일": f"{post_ret_5*100:+.1f}%",
            "직후 20일": f"{post_ret_20*100:+.1f}%",
        })

    return pd.DataFrame(rows)
