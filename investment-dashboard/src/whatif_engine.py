"""
What-If Scenario Engine: 포트폴리오 비중 변경 시 Skills 변화를 실시간 시뮬레이션

핵심 기능:
    1. 비중 조절 → Skills 재산출 → Before/After 비교
    2. 종목 추가/제거 시뮬레이션
    3. 프리셋 시나리오 (동일비중, 리스크 패리티 등)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from skills_engine import generate_skills_profile, SkillsProfile
from visualizations import COLORS


@dataclass
class ScenarioResult:
    """시나리오 비교 결과"""
    original_profile: SkillsProfile
    modified_profile: SkillsProfile
    original_weights: Dict[str, float]
    modified_weights: Dict[str, float]
    original_return: float
    modified_return: float

    def skill_changes(self) -> Dict[str, float]:
        """각 Skill의 변화량 (modified - original)"""
        orig = self.original_profile.to_dict()
        mod = self.modified_profile.to_dict()
        return {k: mod[k] - orig[k] for k in orig}

    def overall_change(self) -> float:
        return self.modified_profile.overall_score() - self.original_profile.overall_score()


# =============================================================================
# 프리셋 시나리오 생성
# =============================================================================

def generate_equal_weight(tickers: List[str]) -> Dict[str, float]:
    """동일 비중 포트폴리오"""
    n = len(tickers)
    return {t: 1.0 / n for t in tickers}


def generate_concentrated(
    weights: Dict[str, float],
    returns_by_ticker: Dict[str, float],
    top_n: int = 3,
) -> Dict[str, float]:
    """수익률 상위 N개 종목에 집중하는 시나리오"""
    sorted_tickers = sorted(
        returns_by_ticker.items(), key=lambda x: x[1], reverse=True
    )
    top_tickers = [t for t, _ in sorted_tickers[:top_n] if t in weights]
    rest_tickers = [t for t in weights if t not in top_tickers]

    new_weights = {}
    top_weight = 0.7 / len(top_tickers) if top_tickers else 0
    rest_weight = 0.3 / len(rest_tickers) if rest_tickers else 0

    for t in top_tickers:
        new_weights[t] = top_weight
    for t in rest_tickers:
        new_weights[t] = rest_weight

    return new_weights


def generate_defensive(
    weights: Dict[str, float],
    prices: pd.DataFrame,
) -> Dict[str, float]:
    """변동성 역비례 비중 (Low-Vol 전략)"""
    tickers = list(weights.keys())
    vols = {}

    for ticker in tickers:
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                close = prices["Close"][ticker]
            else:
                close = prices["Close"]
            daily_ret = close.pct_change().dropna()
            vols[ticker] = daily_ret.std()
        except Exception:
            vols[ticker] = 0.02  # 기본값

    # 변동성 역수 비중
    inv_vols = {t: 1.0 / v if v > 0 else 0 for t, v in vols.items()}
    total = sum(inv_vols.values())

    if total == 0:
        return generate_equal_weight(tickers)

    return {t: v / total for t, v in inv_vols.items()}


def generate_momentum(
    weights: Dict[str, float],
    returns_by_ticker: Dict[str, float],
) -> Dict[str, float]:
    """모멘텀 전략: 수익률 비례 비중 (양수만)"""
    positive_rets = {t: max(r, 0.001) for t, r in returns_by_ticker.items() if t in weights}

    if not positive_rets:
        return generate_equal_weight(list(weights.keys()))

    total = sum(positive_rets.values())
    momentum_weights = {t: r / total for t, r in positive_rets.items()}

    # 음수 수익률 종목은 최소 비중
    for t in weights:
        if t not in momentum_weights:
            momentum_weights[t] = 0.01

    # 재정규화
    total = sum(momentum_weights.values())
    return {t: w / total for t, w in momentum_weights.items()}


# =============================================================================
# 시나리오 실행
# =============================================================================

def run_scenario(
    modified_weights: Dict[str, float],
    original_weights: Dict[str, float],
    original_profile: SkillsProfile,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    prices: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
) -> ScenarioResult:
    """
    수정된 비중으로 Skills를 재산출하고 원본과 비교

    Parameters:
        modified_weights: 변경된 비중
        original_weights: 원래 비중
        original_profile: 원래 Skills 프로필
        portfolio_returns: 원래 포트폴리오 수익률
        benchmark_returns: 벤치마크 수익률
        prices: 가격 데이터
        sector_map: 섹터 매핑
    """
    # 비중 정규화
    total = sum(modified_weights.values())
    if total > 0:
        modified_weights = {t: w / total for t, w in modified_weights.items()}

    # 수정 포트폴리오 수익률 재계산
    tickers = list(modified_weights.keys())
    w = np.array([modified_weights.get(t, 0) for t in tickers])

    try:
        if isinstance(prices.columns, pd.MultiIndex):
            close_prices = prices["Close"][tickers]
        else:
            close_prices = prices[["Close"]].copy()
            close_prices.columns = tickers

        daily_returns = close_prices.pct_change().dropna()
        modified_port_returns = daily_returns.dot(w)

        # 종목별 누적 수익률
        cumulative = (1 + daily_returns).prod() - 1
        modified_returns_by_ticker = {t: float(cumulative[t]) for t in tickers}
    except Exception:
        # 폴백: 원래 수익률 사용
        modified_port_returns = portfolio_returns
        modified_returns_by_ticker = {t: 0.0 for t in tickers}

    # 인덱스 정렬
    common_idx = modified_port_returns.index.intersection(benchmark_returns.index)
    mod_returns = modified_port_returns.reindex(common_idx)
    bench_returns = benchmark_returns.reindex(common_idx)

    # Skills 재산출
    modified_profile = generate_skills_profile(
        portfolio_returns=mod_returns,
        benchmark_returns=bench_returns,
        weights=modified_weights,
        returns_by_ticker=modified_returns_by_ticker,
        sector_map=sector_map,
    )

    # 총 수익률
    orig_total_return = float((1 + portfolio_returns).prod() - 1)
    mod_total_return = float((1 + mod_returns).prod() - 1)

    return ScenarioResult(
        original_profile=original_profile,
        modified_profile=modified_profile,
        original_weights=original_weights,
        modified_weights=modified_weights,
        original_return=orig_total_return,
        modified_return=mod_total_return,
    )


# =============================================================================
# 시각화
# =============================================================================

def create_comparison_radar(
    original_skills: Dict[str, float],
    modified_skills: Dict[str, float],
    original_label: str = "Current",
    modified_label: str = "What-If",
) -> go.Figure:
    """Before/After Skills Radar 오버레이"""
    categories = list(original_skills.keys())
    orig_values = list(original_skills.values())
    mod_values = list(modified_skills.values())

    # 닫기
    categories_closed = categories + [categories[0]]
    orig_closed = orig_values + [orig_values[0]]
    mod_closed = mod_values + [mod_values[0]]

    fig = go.Figure()

    # 등급 구간 배경
    for threshold, color in [
        (90, "rgba(16, 185, 129, 0.06)"),
        (75, "rgba(99, 102, 241, 0.06)"),
        (55, "rgba(245, 158, 11, 0.06)"),
        (35, "rgba(239, 68, 68, 0.06)"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=[threshold] * len(categories_closed),
            theta=categories_closed,
            fill="toself", fillcolor=color,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

    # Original (회색 점선)
    fig.add_trace(go.Scatterpolar(
        r=orig_closed, theta=categories_closed,
        fill="toself",
        fillcolor="rgba(107, 114, 128, 0.15)",
        line=dict(color=COLORS["neutral"], width=2, dash="dot"),
        marker=dict(size=7, color=COLORS["neutral"]),
        name=original_label,
        text=[f"{v:.0f}" for v in orig_closed],
        mode="lines+markers",
    ))

    # Modified (밝은 인디고 실선)
    fig.add_trace(go.Scatterpolar(
        r=mod_closed, theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.25)",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=10, color=COLORS["primary"]),
        name=modified_label,
        text=[f"{v:.0f}" for v in mod_closed],
        textposition="top center",
        mode="lines+markers+text",
        textfont=dict(size=13, color=COLORS["text"]),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        title=dict(text="Skills Comparison: Current vs What-If", x=0.5, font=dict(size=18)),
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                gridcolor=COLORS["grid"],
            ),
            angularaxis=dict(gridcolor=COLORS["grid"], tickfont=dict(size=12)),
            bgcolor=COLORS["background"],
        ),
        legend=dict(x=0.35, y=-0.15, orientation="h"),
        height=500,
        margin=dict(l=40, r=40, t=60, b=60),
    )

    return fig


def create_change_waterfall(skill_changes: Dict[str, float]) -> go.Figure:
    """Skills 변화량 워터폴 차트"""
    skills = list(skill_changes.keys())
    changes = list(skill_changes.values())

    colors_list = [
        COLORS["positive"] if c >= 0 else COLORS["negative"]
        for c in changes
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=skills,
        x=changes,
        orientation="h",
        marker=dict(color=colors_list, line=dict(width=0)),
        text=[f"{c:+.1f}" for c in changes],
        textposition="outside",
        textfont=dict(size=13, color=COLORS["text"]),
        hovertemplate="<b>%{y}</b><br>Change: %{x:+.1f}<extra></extra>",
    ))

    # 0선
    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        title=dict(text="Skill Score Changes", x=0.5, font=dict(size=18)),
        xaxis=dict(
            title="Score Change",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor=COLORS["neutral"],
        ),
        yaxis=dict(autorange="reversed"),
        height=300,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def create_weight_comparison_bar(
    original: Dict[str, float],
    modified: Dict[str, float],
) -> go.Figure:
    """비중 변화 grouped bar chart"""
    all_tickers = sorted(set(list(original.keys()) + list(modified.keys())))

    orig_vals = [original.get(t, 0) * 100 for t in all_tickers]
    mod_vals = [modified.get(t, 0) * 100 for t in all_tickers]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Current",
        y=all_tickers, x=orig_vals,
        orientation="h",
        marker=dict(color=COLORS["neutral"], opacity=0.7),
        texttemplate="%{x:.1f}%",
        textposition="outside",
        textfont=dict(size=9),
    ))

    fig.add_trace(go.Bar(
        name="What-If",
        y=all_tickers, x=mod_vals,
        orientation="h",
        marker=dict(color=COLORS["primary"]),
        texttemplate="%{x:.1f}%",
        textposition="outside",
        textfont=dict(size=9),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        title=dict(text="Weight Allocation: Current vs What-If", x=0.5, font=dict(size=16)),
        barmode="group",
        xaxis=dict(title="Weight (%)", gridcolor=COLORS["grid"]),
        yaxis=dict(autorange="reversed"),
        legend=dict(x=0.7, y=1.0),
        height=max(300, len(all_tickers) * 35),
        margin=dict(l=40, r=60, t=60, b=40),
    )

    return fig
