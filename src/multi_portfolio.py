"""
Multi-Portfolio Comparison: 여러 포트폴리오를 동시에 비교
- Skills 레이더 오버레이
- 성과 지표 비교 테이블
- 수익률 곡선 비교
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import plotly.graph_objects as go


@dataclass
class PortfolioSummary:
    """포트폴리오 요약 정보"""
    name: str
    weights: Dict[str, float]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    skills: Dict[str, float]  # {skill_name: score}
    overall_skill: float


# ========== Built-in Comparison Portfolios ==========

COMPARISON_PORTFOLIOS = {
    "S&P 500 추종 (US Large Cap)": {
        "AAPL": 0.07, "MSFT": 0.07, "AMZN": 0.04, "NVDA": 0.04,
        "GOOGL": 0.04, "META": 0.03, "BRK-B": 0.02, "UNH": 0.02,
        "JNJ": 0.02, "JPM": 0.02, "V": 0.02, "PG": 0.02,
        "XOM": 0.02, "HD": 0.02, "MA": 0.02, "CVX": 0.02,
        "ABBV": 0.02, "MRK": 0.02, "KO": 0.02, "PEP": 0.02,
        "SPY": 0.45,
    },
    "글로벌 분산 (All Weather 스타일)": {
        "VTI": 0.30, "VXUS": 0.15, "BND": 0.20,
        "TLT": 0.15, "GLD": 0.075, "GSG": 0.075, "VNQ": 0.05,
    },
    "한국 대표 (KOSPI Top)": {
        "005930.KS": 0.25, "000660.KS": 0.15, "005380.KS": 0.10,
        "035420.KS": 0.10, "035720.KS": 0.08, "373220.KS": 0.08,
        "207940.KS": 0.07, "105560.KS": 0.07, "005490.KS": 0.05,
        "017670.KS": 0.05,
    },
    "성장주 집중": {
        "NVDA": 0.15, "TSLA": 0.12, "AMD": 0.10, "AMZN": 0.10,
        "META": 0.10, "GOOGL": 0.08, "CRM": 0.08, "NFLX": 0.07,
        "SHOP": 0.05, "SQ": 0.05, "SNOW": 0.05, "PLTR": 0.05,
    },
    "배당 안정형": {
        "VYM": 0.20, "SCHD": 0.20, "O": 0.08, "JNJ": 0.08,
        "PG": 0.08, "KO": 0.08, "PEP": 0.07, "MMM": 0.05,
        "T": 0.05, "VZ": 0.05, "XOM": 0.03, "CVX": 0.03,
    },
}


def compute_portfolio_summary(
    name: str,
    weights: Dict[str, float],
    returns: pd.Series,
    benchmark_returns: pd.Series,
    skills_dict: Dict[str, float],
) -> PortfolioSummary:
    """포트폴리오 요약 정보 계산"""
    total_return = float((1 + returns).prod() - 1)
    vol = float(returns.std() * np.sqrt(252))
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = float(dd.min())

    overall = sum(skills_dict.values()) / len(skills_dict) if skills_dict else 0

    return PortfolioSummary(
        name=name,
        weights=weights,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        volatility=vol,
        skills=skills_dict,
        overall_skill=overall,
    )


# ========== Plotly Visualizations ==========

def create_multi_radar(portfolios: List[PortfolioSummary]) -> go.Figure:
    """
    여러 포트폴리오의 Skills 레이더 오버레이
    - 각 포트폴리오는 다른 색상의 반투명 영역
    - 최대 5개까지 비교
    - 다크 테마
    - Colors: ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
    """
    skill_names = ["Timing", "Diversification", "Risk Management",
                   "Conviction", "Adaptability", "Consistency"]
    colors = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    fig = go.Figure()

    for i, pf in enumerate(portfolios[:5]):
        values = [pf.skills.get(s, 50) for s in skill_names]
        values.append(values[0])  # close the polygon

        color = colors[i % len(colors)]

        # Convert hex color to RGB for rgba
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=skill_names + [skill_names[0]],
            name=f"{pf.name} ({pf.overall_skill:.0f})",
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Multi-Portfolio Skills Comparison",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=500,
        polar=dict(
            bgcolor="#1E293B",
            radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.2,
            xanchor="center", x=0.5,
        ),
        font=dict(color="#E2E8F0"),
    )

    return fig


def create_performance_comparison_bar(portfolios: List[PortfolioSummary]) -> go.Figure:
    """
    성과 지표 비교 그룹 바 차트
    - 지표: Total Return, Sharpe, MDD, Volatility, Overall Skill
    - 각 포트폴리오별 그룹
    - 다크 테마
    """
    colors = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    metrics = ["수익률(%)", "Sharpe", "MDD(%)", "변동성(%)", "Overall Skill"]

    fig = go.Figure()

    for i, pf in enumerate(portfolios[:5]):
        values = [
            pf.total_return * 100,
            pf.sharpe_ratio,
            pf.max_drawdown * 100,
            pf.volatility * 100,
            pf.overall_skill,
        ]
        fig.add_trace(go.Bar(
            name=pf.name,
            x=metrics,
            y=values,
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title="Performance Metrics Comparison",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=400,
        barmode="group",
        xaxis=dict(title="지표", tickfont=dict(size=11)),
        yaxis=dict(title="값", tickfont=dict(size=11)),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
            font=dict(size=10)
        ),
        font=dict(color="#E2E8F0"),
        margin=dict(b=120),
    )

    return fig


def create_comparison_table(portfolios: List[PortfolioSummary]) -> pd.DataFrame:
    """
    비교 테이블 DataFrame
    Columns: Portfolio, Return, Sharpe, MDD, Vol, Overall Skill, Best Skill, Worst Skill
    """
    rows = []
    for pf in portfolios:
        best = max(pf.skills, key=pf.skills.get) if pf.skills else "-"
        worst = min(pf.skills, key=pf.skills.get) if pf.skills else "-"
        rows.append({
            "포트폴리오": pf.name,
            "수익률": f"{pf.total_return*100:+.1f}%",
            "Sharpe": f"{pf.sharpe_ratio:.2f}",
            "MDD": f"{pf.max_drawdown*100:.1f}%",
            "변동성": f"{pf.volatility*100:.1f}%",
            "Overall Skill": f"{pf.overall_skill:.0f}",
            "강점": best,
            "약점": worst,
        })
    return pd.DataFrame(rows)


def create_cumulative_returns_line(
    portfolios: List[PortfolioSummary],
    returns_dict: Dict[str, pd.Series],
) -> go.Figure:
    """
    누적 수익률 비교 곡선 (시계열)
    - 각 포트폴리오의 누적 수익률을 라인으로 표시
    - 다크 테마
    """
    colors = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    fig = go.Figure()

    for i, pf in enumerate(portfolios[:5]):
        if pf.name in returns_dict:
            returns = returns_dict[pf.name]
            cum_returns = (1 + returns).cumprod() - 1

            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns.values * 100,
                name=pf.name,
                line=dict(color=color, width=2),
                mode="lines",
            ))

    fig.update_layout(
        title="Cumulative Returns Comparison",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=400,
        xaxis=dict(title="Date", tickfont=dict(size=10)),
        yaxis=dict(title="누적 수익률 (%)", tickfont=dict(size=10)),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
            font=dict(size=10)
        ),
        font=dict(color="#E2E8F0"),
        margin=dict(b=120),
        hovermode="x unified",
    )

    return fig


def create_risk_return_scatter(portfolios: List[PortfolioSummary]) -> go.Figure:
    """
    위험-수익 산점도
    - X축: 변동성 (위험), Y축: 수익률
    - 각 포트폴리오를 버블로 표시 (Sharpe 비율로 크기 결정)
    - 다크 테마
    """
    colors = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    fig = go.Figure()

    for i, pf in enumerate(portfolios[:5]):
        color = colors[i % len(colors)]

        # Sharpe ratio를 크기로 사용 (음수인 경우 처리)
        size = max(10, min(50, 10 + pf.sharpe_ratio * 5))

        fig.add_trace(go.Scatter(
            x=[pf.volatility * 100],
            y=[pf.total_return * 100],
            mode="markers+text",
            name=pf.name,
            marker=dict(
                size=size,
                color=color,
                line=dict(width=2, color="rgba(226, 232, 240, 0.5)"),
                opacity=0.8,
            ),
            text=[pf.name],
            textposition="top center",
            textfont=dict(color="#E2E8F0", size=10),
            hovertemplate=(
                f"<b>{pf.name}</b><br>"
                f"위험(변동성): %{{x:.1f}}%<br>"
                f"수익률: %{{y:+.1f}}%<br>"
                f"Sharpe: {pf.sharpe_ratio:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Risk-Return Profile",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=400,
        xaxis=dict(title="변동성 - 위험 (%)", tickfont=dict(size=11)),
        yaxis=dict(title="누적 수익률 (%)", tickfont=dict(size=11)),
        legend=dict(
            orientation="v", yanchor="top", y=0.95,
            xanchor="left", x=0.02,
            font=dict(size=10),
            bgcolor="rgba(15, 23, 42, 0.8)",
        ),
        font=dict(color="#E2E8F0"),
        showlegend=False,
        hovermode="closest",
    )

    return fig


def get_portfolio_weights_comparison(portfolios: List[PortfolioSummary]) -> pd.DataFrame:
    """
    포트폴리오 가중치 비교 테이블
    - 각 포트폴리오의 상위 5개 자산 및 가중치 표시
    """
    rows = []
    for pf in portfolios:
        sorted_weights = sorted(pf.weights.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_weights[:5]

        for rank, (ticker, weight) in enumerate(top_5, 1):
            if rank == 1:
                rows.append({
                    "포트폴리오": pf.name,
                    "순위": rank,
                    "자산": ticker,
                    "가중치": f"{weight*100:.1f}%",
                })
            else:
                rows.append({
                    "포트폴리오": "",
                    "순위": rank,
                    "자산": ticker,
                    "가중치": f"{weight*100:.1f}%",
                })

    return pd.DataFrame(rows)
