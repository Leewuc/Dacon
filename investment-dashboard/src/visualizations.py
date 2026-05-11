"""
Visualizations: Plotly 기반 인터랙티브 차트 모듈

차트 목록:
    1. Skills Radar Chart (6축 레이더)
    2. Cumulative Returns (누적 수익률)
    3. Drawdown Chart (드로다운)
    4. Sector Allocation (섹터 비중 트리맵)
    5. Monthly Heatmap (월별 수익률 히트맵)
    6. Risk-Return Scatter (리스크-수익 산점도)
    7. Monte Carlo Simulation (시뮬레이션 팬 차트)
    8. Skills Detail Bar (스킬 상세 바 차트)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

# 컬러 팔레트 (일관된 디자인)
COLORS = {
    "primary": "#6366F1",      # Indigo
    "secondary": "#8B5CF6",    # Violet
    "accent": "#EC4899",       # Pink
    "positive": "#10B981",     # Emerald
    "negative": "#EF4444",     # Red
    "neutral": "#6B7280",      # Gray
    "background": "#0F172A",   # Slate dark
    "surface": "#1E293B",      # Slate
    "text": "#F8FAFC",         # Slate light
    "grid": "#334155",         # Slate grid
}

SKILL_COLORS = {
    "Timing": "#6366F1",
    "Diversification": "#8B5CF6",
    "Risk Management": "#EC4899",
    "Conviction": "#F59E0B",
    "Adaptability": "#10B981",
    "Consistency": "#06B6D4",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["background"],
    plot_bgcolor=COLORS["background"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(l=40, r=40, t=60, b=40),
)


def create_skills_radar(skills_dict: Dict[str, float], title: str = "Investment Skills Radar") -> go.Figure:
    """
    6축 레이더 차트 - 투자 Skills 시각화

    Parameters:
        skills_dict: {skill_name: score (0-100)}
    """
    categories = list(skills_dict.keys())
    values = list(skills_dict.values())

    # 레이더 닫기
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    # 배경 영역 (등급 구간)
    for threshold, color, name in [
        (90, "rgba(16, 185, 129, 0.08)", "S등급"),
        (75, "rgba(99, 102, 241, 0.08)", "A등급"),
        (55, "rgba(245, 158, 11, 0.08)", "B등급"),
        (35, "rgba(239, 68, 68, 0.08)", "C등급"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=[threshold] * (len(categories) + 1),
            theta=categories_closed,
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 메인 데이터
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.25)",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=10, color=COLORS["primary"]),
        name="My Skills",
        text=[f"{v:.0f}" for v in values_closed],
        textposition="top center",
        mode="lines+markers+text",
        textfont=dict(size=14, color=COLORS["text"]),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, x=0.5, font=dict(size=20)),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                gridcolor=COLORS["grid"],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                gridcolor=COLORS["grid"],
                tickfont=dict(size=13, color=COLORS["text"]),
            ),
            bgcolor=COLORS["background"],
        ),
        height=500,
    )

    return fig


def create_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_name: str = "Portfolio",
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    """누적 수익률 차트 (포트폴리오 vs 벤치마크)"""
    port_cum = (1 + portfolio_returns).cumprod() - 1
    bench_cum = (1 + benchmark_returns).cumprod() - 1

    fig = go.Figure()

    # 포트폴리오
    fig.add_trace(go.Scatter(
        x=port_cum.index,
        y=port_cum.values * 100,
        name=portfolio_name,
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tonexty" if False else None,
        hovertemplate="%{x|%Y-%m-%d}<br>수익률: %{y:.1f}%<extra></extra>",
    ))

    # 벤치마크
    fig.add_trace(go.Scatter(
        x=bench_cum.index,
        y=bench_cum.values * 100,
        name=benchmark_name,
        line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>수익률: %{y:.1f}%<extra></extra>",
    ))

    # 초과 수익 영역 표시
    alpha = port_cum - bench_cum
    positive_alpha = alpha.copy()
    negative_alpha = alpha.copy()
    positive_alpha[positive_alpha < 0] = 0
    negative_alpha[negative_alpha > 0] = 0

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Cumulative Returns", x=0.5, font=dict(size=18)),
        xaxis=dict(gridcolor=COLORS["grid"], title=""),
        yaxis=dict(gridcolor=COLORS["grid"], title="Return (%)", ticksuffix="%"),
        legend=dict(x=0.02, y=0.98),
        height=400,
        hovermode="x unified",
    )

    return fig


def create_drawdown_chart(returns: pd.Series) -> go.Figure:
    """드로다운(낙폭) 차트"""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.3)",
        line=dict(color=COLORS["negative"], width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1f}%<extra></extra>",
        name="Drawdown",
    ))

    # MDD 표시
    mdd_idx = drawdown.idxmin()
    mdd_val = drawdown.min()

    fig.add_annotation(
        x=mdd_idx,
        y=mdd_val,
        text=f"MDD: {mdd_val:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["negative"],
        font=dict(color=COLORS["negative"], size=12),
        bgcolor=COLORS["surface"],
        borderpad=4,
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Drawdown Analysis", x=0.5, font=dict(size=18)),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], title="Drawdown (%)", ticksuffix="%"),
        height=300,
        showlegend=False,
    )

    return fig


def create_sector_treemap(
    weights: Dict[str, float],
    sector_map: Dict[str, str],
    returns_by_ticker: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """섹터별 비중 트리맵"""
    labels = []
    parents = []
    values = []
    colors = []

    # 섹터 집계
    sector_weights: Dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = sector_map.get(ticker, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0) + weight

    # Root
    labels.append("Portfolio")
    parents.append("")
    values.append(0)
    colors.append(0)

    # 섹터 레벨
    for sector, sw in sector_weights.items():
        labels.append(sector)
        parents.append("Portfolio")
        values.append(round(sw * 100, 1))
        colors.append(0)

    # 종목 레벨
    for ticker, weight in weights.items():
        sector = sector_map.get(ticker, "Unknown")
        ret = returns_by_ticker.get(ticker, 0) if returns_by_ticker else 0
        labels.append(f"{ticker}<br>{weight*100:.1f}%")
        parents.append(sector)
        values.append(round(weight * 100, 1))
        colors.append(ret * 100)

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale="RdYlGn",
            cmid=0,
            showscale=True,
            colorbar=dict(title="Return %"),
        ),
        textinfo="label+value",
        texttemplate="%{label}<br>%{value:.1f}%",
        hovertemplate="<b>%{label}</b><br>비중: %{value:.1f}%<br>수익률: %{color:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Sector Allocation", x=0.5, font=dict(size=18)),
        height=450,
    )

    return fig


def create_monthly_heatmap(returns: pd.Series) -> go.Figure:
    """월별 수익률 히트맵"""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    # Year x Month 피벗
    heatmap_data = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })

    pivot = heatmap_data.pivot_table(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate="%{text:.1f}%",
        textfont=dict(size=11),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return %"),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Monthly Returns Heatmap", x=0.5, font=dict(size=18)),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        height=300,
    )

    return fig


def create_risk_return_scatter(
    returns_by_ticker: Dict[str, float],
    prices: pd.DataFrame,
    weights: Dict[str, float],
) -> go.Figure:
    """종목별 리스크-수익 산점도 (버블 크기 = 비중)"""
    tickers = list(weights.keys())
    data = []

    for ticker in tickers:
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                close = prices["Close"][ticker]
            else:
                close = prices["Close"]
            daily_ret = close.pct_change().dropna()
            vol = daily_ret.std() * np.sqrt(252) * 100  # 연환산 변동성
            ret = returns_by_ticker.get(ticker, 0) * 100
            w = weights[ticker] * 100
            data.append({"ticker": ticker, "return": ret, "volatility": vol, "weight": w})
        except Exception:
            continue

    df = pd.DataFrame(data)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="데이터 부족", showarrow=False)
        return fig

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["volatility"],
        y=df["return"],
        mode="markers+text",
        marker=dict(
            size=df["weight"] * 3,
            color=df["return"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Return %"),
            line=dict(width=1, color=COLORS["text"]),
        ),
        text=df["ticker"],
        textposition="top center",
        textfont=dict(size=10),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "수익률: %{y:.1f}%<br>"
            "변동성: %{x:.1f}%<br>"
            "비중: %{marker.size:.1f}%"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Risk-Return Analysis", x=0.5, font=dict(size=18)),
        xaxis=dict(title="Annualized Volatility (%)", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Total Return (%)", gridcolor=COLORS["grid"]),
        height=400,
        showlegend=False,
    )

    return fig


def create_monte_carlo_chart(
    returns: pd.Series,
    n_simulations: int = 500,
    n_days: int = 252,
    initial_value: float = 10000,
) -> go.Figure:
    """
    Monte Carlo 시뮬레이션 팬 차트
    현재 포트폴리오 통계를 기반으로 향후 1년 경로 시뮬레이션
    """
    mu = returns.mean()
    sigma = returns.std()

    # 시뮬레이션
    np.random.seed(42)
    simulations = np.zeros((n_simulations, n_days))

    for i in range(n_simulations):
        daily_returns = np.random.normal(mu, sigma, n_days)
        simulations[i] = initial_value * np.cumprod(1 + daily_returns)

    # 백분위수
    percentiles = {
        "5th": np.percentile(simulations, 5, axis=0),
        "25th": np.percentile(simulations, 25, axis=0),
        "50th (Median)": np.percentile(simulations, 50, axis=0),
        "75th": np.percentile(simulations, 75, axis=0),
        "95th": np.percentile(simulations, 95, axis=0),
    }

    days = list(range(n_days))

    fig = go.Figure()

    # 5-95% 밴드
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(percentiles["95th"]) + list(percentiles["5th"][::-1]),
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.1)",
        line=dict(width=0),
        name="5th-95th Percentile",
        hoverinfo="skip",
    ))

    # 25-75% 밴드
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(percentiles["75th"]) + list(percentiles["25th"][::-1]),
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.2)",
        line=dict(width=0),
        name="25th-75th Percentile",
        hoverinfo="skip",
    ))

    # 중앙값
    fig.add_trace(go.Scatter(
        x=days,
        y=percentiles["50th (Median)"],
        line=dict(color=COLORS["primary"], width=2.5),
        name="Median Path",
        hovertemplate="Day %{x}<br>Value: $%{y:,.0f}<extra></extra>",
    ))

    # 시작점
    fig.add_hline(
        y=initial_value,
        line_dash="dash",
        line_color=COLORS["neutral"],
        annotation_text=f"Start: ${initial_value:,.0f}",
    )

    # 최종 통계 annotation
    final_median = percentiles["50th (Median)"][-1]
    final_5th = percentiles["5th"][-1]
    final_95th = percentiles["95th"][-1]

    fig.add_annotation(
        x=n_days - 1,
        y=final_median,
        text=(
            f"Median: ${final_median:,.0f}<br>"
            f"5th: ${final_5th:,.0f}<br>"
            f"95th: ${final_95th:,.0f}"
        ),
        showarrow=True,
        arrowhead=2,
        bgcolor=COLORS["surface"],
        borderpad=6,
        font=dict(size=11),
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Monte Carlo Simulation ({n_simulations} paths, {n_days} days)",
            x=0.5,
            font=dict(size=18),
        ),
        xaxis=dict(title="Trading Days", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Portfolio Value ($)", gridcolor=COLORS["grid"]),
        height=450,
        legend=dict(x=0.02, y=0.98),
    )

    return fig


def create_skills_detail_bars(skills_dict: Dict[str, float]) -> go.Figure:
    """Skills 상세 수평 바 차트 (등급 컬러)"""
    skills = list(skills_dict.keys())
    scores = list(skills_dict.values())

    colors = []
    for s in scores:
        if s >= 90:
            colors.append("#10B981")  # S - Emerald
        elif s >= 75:
            colors.append("#6366F1")  # A - Indigo
        elif s >= 55:
            colors.append("#F59E0B")  # B - Amber
        elif s >= 35:
            colors.append("#F97316")  # C - Orange
        else:
            colors.append("#EF4444")  # D - Red

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=skills,
        x=scores,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
        textfont=dict(size=14, color=COLORS["text"]),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
    ))

    # 등급 구간 세로선
    for threshold, label in [(35, "D|C"), (55, "C|B"), (75, "B|A"), (90, "A|S")]:
        fig.add_vline(
            x=threshold,
            line_dash="dot",
            line_color=COLORS["grid"],
            annotation_text=label,
            annotation_position="top",
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Skills Breakdown", x=0.5, font=dict(size=18)),
        xaxis=dict(range=[0, 105], title="Score", gridcolor=COLORS["grid"]),
        yaxis=dict(autorange="reversed"),
        height=350,
        showlegend=False,
    )

    return fig
