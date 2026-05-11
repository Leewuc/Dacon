"""
Investment Style Analysis: 포트폴리오 투자 스타일 자동 분류
- Returns-Based Style Analysis (RBSA)
- Morningstar Style Box 시각화
- Style Drift 분석 (시간에 따른 스타일 변화)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import plotly.graph_objects as go


# Style categories
STYLE_LABELS = {
    "value": "가치주",
    "growth": "성장주",
    "blend": "혼합형",
    "large": "대형주",
    "mid": "중형주",
    "small": "소형주",
    "momentum": "모멘텀",
    "quality": "퀄리티",
    "dividend": "배당",
    "defensive": "방어적",
    "aggressive": "공격적",
}


@dataclass
class StyleResult:
    """투자 스타일 분석 결과"""
    # Primary classification
    size_style: str           # "large", "mid", "small"
    value_growth: str         # "value", "growth", "blend"
    style_box_position: Tuple[float, float]  # (x: value-growth, y: size) for style box

    # Style scores (0~100)
    value_score: float
    growth_score: float
    momentum_score: float
    quality_score: float
    dividend_score: float
    volatility_score: float   # low = defensive, high = aggressive

    # Style description
    primary_style: str        # e.g., "Large-Cap Growth"
    secondary_traits: List[str]  # e.g., ["High Momentum", "Quality"]
    style_description: str    # Korean description

    # For style drift
    rolling_styles: Optional[pd.DataFrame] = field(default=None)


def _get_sector_classification(sector_name: str) -> Tuple[str, str, str]:
    """
    Classify sector into growth/value, large/small, dividend status.
    Returns: (vg_type, size_type, dividend_status)

    Handles both English and Korean sector names.
    """
    if not sector_name:
        return "blend", "mid", "non-dividend"

    sector = sector_name.strip().lower()

    # Growth sectors
    growth_sectors = {
        "technology", "it", "it/플랫폼", "게임", "바이오", "생명과학",
        "2차전지", "배터리", "반도체", "핀테크", "전자부품", "소프트웨어",
        "인터넷", "온라인", "전자상거래"
    }

    # Value sectors
    value_sectors = {
        "금융", "financial", "은행", "보험", "유틸리티", "통신", "텔레콤",
        "에너지", "에너지/원자력", "석유화학", "화학", "식품", "식품료",
        "건설", "부동산", "철강", "자동차", "자동차부품"
    }

    # Small-cap heavy sectors
    small_sectors = {
        "게임", "바이오", "생명과학", "핀테크", "화장품", "여행", "여행/레저",
        "레저", "엔터테인먼트", "의료기기", "의약품"
    }

    # Large-cap heavy sectors
    large_sectors = {
        "반도체", "금융", "financial", "은행", "에너지", "에너지/원자력",
        "IT", "IT/플랫폼", "자동차", "통신", "텔레콤", "식품", "식품료",
        "철강", "화학"
    }

    # Dividend-paying sectors
    dividend_sectors = {
        "금융", "financial", "은행", "보험", "통신", "텔레콤", "유틸리티",
        "에너지", "에너지/원자력", "식품", "식품료", "택배", "물류"
    }

    # Determine value-growth
    if any(g in sector for g in growth_sectors):
        vg = "growth"
    elif any(v in sector for v in value_sectors):
        vg = "value"
    else:
        vg = "blend"

    # Determine size
    if any(s in sector for s in small_sectors):
        size = "small"
    elif any(l in sector for l in large_sectors):
        size = "large"
    else:
        size = "mid"

    # Determine dividend
    div = "dividend" if any(d in sector for d in dividend_sectors) else "non-dividend"

    return vg, size, div


def analyze_portfolio_style(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: Dict[str, float],
    prices: Optional[pd.DataFrame] = None,
    sector_map: Optional[Dict[str, str]] = None,
    returns_by_ticker: Optional[Dict[str, float]] = None,
) -> StyleResult:
    """
    Analyze the investment style of the portfolio.

    Method: Heuristic style scoring based on portfolio characteristics
    (since we don't have fundamental data, we use return patterns and sector composition)

    Scoring:
    1. Value vs Growth:
       - Low volatility + high dividend sectors → Value
       - High beta + tech/growth sectors → Growth

    2. Size:
       - Based on typical sector composition
       - Korean small caps vs large caps

    3. Momentum: Recent returns vs long-term
    4. Quality: Sharpe ratio, consistency
    5. Dividend: Presence of dividend-paying sectors
    6. Volatility: Portfolio volatility relative to benchmark

    Args:
        returns: Daily portfolio returns (pd.Series)
        benchmark_returns: Daily benchmark returns (pd.Series)
        weights: Dict[ticker] -> weight
        prices: Optional DataFrame of prices for reference
        sector_map: Dict[ticker] -> sector name
        returns_by_ticker: Optional dict of individual stock returns

    Returns:
        StyleResult with complete style analysis
    """

    # Handle edge cases
    if weights is None or len(weights) == 0:
        # Empty portfolio - neutral style
        return _create_neutral_style()

    if returns is None or len(returns) == 0:
        return _create_neutral_style()

    # Initialize sector maps if not provided
    if sector_map is None:
        sector_map = {}

    # Ensure we have numeric data
    returns = pd.Series(returns, dtype=float)
    benchmark_returns = pd.Series(benchmark_returns, dtype=float)

    # Remove any NaN values
    valid_idx = returns.notna() & (returns != 0)
    returns = returns[valid_idx]
    benchmark_returns = benchmark_returns[valid_idx]

    if len(returns) < 2:
        return _create_neutral_style()

    # ========== Sector-based classification ==========
    growth_weight = 0.0
    value_weight = 0.0
    small_weight = 0.0
    large_weight = 0.0
    dividend_weight = 0.0

    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1.0

    for ticker, weight in weights.items():
        sector = sector_map.get(ticker, "")
        vg, size, div = _get_sector_classification(sector)

        normalized_weight = weight / total_weight if total_weight > 0 else 0

        if vg == "growth":
            growth_weight += normalized_weight
        elif vg == "value":
            value_weight += normalized_weight
        else:  # blend
            growth_weight += normalized_weight * 0.5
            value_weight += normalized_weight * 0.5

        if size == "small":
            small_weight += normalized_weight
        elif size == "large":
            large_weight += normalized_weight

        if div == "dividend":
            dividend_weight += normalized_weight

    # Value-Growth score (0=deep value, 100=pure growth)
    total_style_weight = growth_weight + value_weight
    if total_style_weight > 0:
        growth_score = (growth_weight / total_style_weight) * 100
        value_score = (value_weight / total_style_weight) * 100
    else:
        growth_score = 50.0
        value_score = 50.0

    # Adjust with return characteristics
    try:
        port_vol = returns.std() * np.sqrt(252)
        bench_vol = benchmark_returns.std() * np.sqrt(252)

        if bench_vol > 0:
            cov = returns.cov(benchmark_returns)
            var = benchmark_returns.var()
            beta = cov / var if var > 0 else 1.0
        else:
            beta = 1.0

        port_vol = max(port_vol, 0.001)  # Avoid division by zero
    except (ValueError, ZeroDivisionError):
        beta = 1.0
        port_vol = 0.15
        bench_vol = 0.15

    # High beta → more growth-like
    if beta > 1.2:
        growth_score = min(100, growth_score + 15)
        value_score = max(0, value_score - 15)
    elif beta < 0.8:
        value_score = min(100, value_score + 15)
        growth_score = max(0, growth_score - 15)

    # Normalize
    growth_score = np.clip(growth_score, 0, 100)
    value_score = np.clip(value_score, 0, 100)

    # Value-Growth classification
    if growth_score > 65:
        value_growth = "growth"
    elif value_score > 65:
        value_growth = "value"
    else:
        value_growth = "blend"

    # ========== Size classification ==========
    # Based on holdings count and sector composition
    n_holdings = len(weights)

    if large_weight > 0.5:
        size_style = "large"
    elif small_weight > 0.3:
        size_style = "small"
    elif n_holdings <= 8:
        # Concentrated portfolio → likely large-cap
        size_style = "large"
    else:
        size_style = "mid"

    # ========== Momentum score ==========
    # Recent 3-month vs 12-month performance
    if len(returns) > 63:
        recent_returns = returns.iloc[-63:].values
        recent_ret = (1 + recent_returns).prod() - 1
        total_returns = returns.values
        total_ret = (1 + total_returns).prod() - 1

        # Annualized recent vs period return
        if len(returns) > 63:
            expected_total = (1 + recent_ret / 1) * (1 + total_ret / (len(returns) / 63))
            momentum_raw = (recent_ret - total_ret / (len(returns) / 63)) * 500
            momentum_score = np.clip(50 + momentum_raw, 0, 100)
        else:
            momentum_score = 50.0
    else:
        momentum_score = 50.0

    # ========== Quality score ==========
    # Based on Sharpe ratio and return consistency
    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret > 0:
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        quality_score = np.clip(50 + sharpe * 20, 0, 100)
    else:
        quality_score = 50.0

    # ========== Dividend score ==========
    dividend_score = np.clip(dividend_weight * 150, 0, 100)

    # ========== Volatility score ==========
    # Low = defensive (0), High = aggressive (100)
    if bench_vol > 0:
        vol_ratio = port_vol / bench_vol
    else:
        vol_ratio = 1.0

    volatility_score = np.clip(vol_ratio * 50, 0, 100)

    # ========== Style box position ==========
    # x: 0=value, 1=blend, 2=growth
    x_pos = (growth_score / 100.0) * 2.0

    # y: 0=small, 1=mid, 2=large
    y_map = {"small": 0.3, "mid": 1.0, "large": 1.7}
    y_pos = y_map.get(size_style, 1.0)

    # ========== Primary style label ==========
    size_kr = {"large": "대형주", "mid": "중형주", "small": "소형주"}
    vg_kr = {"value": "가치", "growth": "성장", "blend": "혼합형"}

    primary_style = f"{size_kr[size_style]} {vg_kr[value_growth]}"

    # ========== Secondary traits ==========
    secondary = []
    if momentum_score > 65:
        secondary.append("모멘텀 강세")
    if quality_score > 70:
        secondary.append("고퀄리티")
    if dividend_score > 60:
        secondary.append("배당 성향")
    if volatility_score > 70:
        secondary.append("공격적")
    elif volatility_score < 30:
        secondary.append("방어적")

    # ========== Description ==========
    desc_parts = [f"포트폴리오는 **{primary_style}** 스타일로 분류됩니다."]

    if value_growth == "growth":
        desc_parts.append("기술/성장 섹터 비중이 높아 시장 상승기에 강한 모습을 보이지만, 하락기 변동성이 클 수 있습니다.")
    elif value_growth == "value":
        desc_parts.append("가치/방어 섹터 중심으로 안정적인 수익을 추구합니다. 배당 수입과 하방 방어에 유리합니다.")
    else:
        desc_parts.append("성장과 가치가 균형 잡힌 포트폴리오입니다.")

    if secondary:
        desc_parts.append(f"추가 특성: {', '.join(secondary)}")

    style_description = " ".join(desc_parts)

    return StyleResult(
        size_style=size_style,
        value_growth=value_growth,
        style_box_position=(x_pos, y_pos),
        value_score=value_score,
        growth_score=growth_score,
        momentum_score=momentum_score,
        quality_score=quality_score,
        dividend_score=dividend_score,
        volatility_score=volatility_score,
        primary_style=primary_style,
        secondary_traits=secondary,
        style_description=style_description,
        rolling_styles=None,
    )


def _create_neutral_style() -> StyleResult:
    """Create a neutral/balanced style result for edge cases."""
    return StyleResult(
        size_style="mid",
        value_growth="blend",
        style_box_position=(1.0, 1.0),
        value_score=50.0,
        growth_score=50.0,
        momentum_score=50.0,
        quality_score=50.0,
        dividend_score=50.0,
        volatility_score=50.0,
        primary_style="중형주 혼합형",
        secondary_traits=[],
        style_description="포트폴리오 데이터가 부족합니다. 중립적 분류로 표시됩니다.",
        rolling_styles=None,
    )


def analyze_style_drift(
    returns_history: pd.DataFrame,
    weights_history: Dict[str, pd.Series],
    benchmark_returns: pd.Series,
    sector_map: Dict[str, str],
    window: int = 63,
) -> pd.DataFrame:
    """
    Analyze how portfolio style changes over time.

    Args:
        returns_history: DataFrame with column for each date's portfolio returns
        weights_history: Dict[ticker] -> pd.Series of weights over time
        benchmark_returns: Series of benchmark returns
        sector_map: Dict[ticker] -> sector
        window: Rolling window size (days)

    Returns:
        DataFrame with columns: date, size_style, value_growth, growth_score, value_score, momentum_score
    """

    if returns_history is None or len(returns_history) == 0:
        return pd.DataFrame()

    # Convert to series if needed
    if isinstance(returns_history, pd.DataFrame):
        if len(returns_history.columns) == 1:
            returns_history = returns_history.iloc[:, 0]
        else:
            # Multiple columns - shouldn't happen but take first
            returns_history = returns_history.iloc[:, 0]

    returns_history = pd.Series(returns_history, dtype=float)

    results = []
    dates = returns_history.index

    for i in range(window, len(returns_history)):
        window_returns = returns_history.iloc[i-window:i]
        window_bench = benchmark_returns.iloc[i-window:i] if isinstance(benchmark_returns, pd.Series) else benchmark_returns

        # Reconstruct weights for this period
        current_weights = {}
        for ticker, weight_series in weights_history.items():
            if isinstance(weight_series, pd.Series) and len(weight_series) > i:
                current_weights[ticker] = weight_series.iloc[i]
            elif isinstance(weight_series, (int, float)):
                current_weights[ticker] = weight_series

        if len(current_weights) == 0:
            continue

        # Analyze style for this window
        style = analyze_portfolio_style(
            returns=window_returns,
            benchmark_returns=window_bench,
            weights=current_weights,
            sector_map=sector_map,
        )

        results.append({
            'date': dates[i] if hasattr(dates[i], 'to_pydatetime') else dates[i],
            'size_style': style.size_style,
            'value_growth': style.value_growth,
            'value_score': style.value_score,
            'growth_score': style.growth_score,
            'momentum_score': style.momentum_score,
            'quality_score': style.quality_score,
            'dividend_score': style.dividend_score,
            'volatility_score': style.volatility_score,
        })

    if len(results) == 0:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ========== Visualizations ==========

def create_style_box(result: StyleResult) -> go.Figure:
    """
    Morningstar-style 3x3 Style Box

    - X axis: Value | Blend | Growth
    - Y axis: Large | Mid | Small
    - Dot showing current portfolio position
    - Background grid with 9 cells
    - Dark theme

    Args:
        result: StyleResult from analyze_portfolio_style

    Returns:
        plotly Figure object
    """

    fig = go.Figure()

    # Draw 3x3 grid cells
    cell_labels = [
        ["대형 가치", "대형 혼합", "대형 성장"],
        ["중형 가치", "중형 혼합", "중형 성장"],
        ["소형 가치", "소형 혼합", "소형 성장"],
    ]

    for row in range(3):
        for col in range(3):
            x0, x1 = col * (2/3), (col + 1) * (2/3)
            y0, y1 = (2 - row) * (2/3), (3 - row) * (2/3)

            # Check if this is the current portfolio cell
            px, py = result.style_box_position
            x_norm = px / 2.0
            y_norm = py / 2.0

            active = (x0 <= x_norm < x1) and (y0 <= y_norm < y1)

            if active:
                fillcolor = "rgba(99, 102, 241, 0.4)"
                edgecolor = "#6366F1"
                edgewidth = 2
            else:
                fillcolor = "rgba(30, 41, 59, 0.6)"
                edgecolor = "#475569"
                edgewidth = 1

            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=edgecolor, width=edgewidth),
                fillcolor=fillcolor,
                layer="below",
            )

            label_text = cell_labels[row][col]
            label_color = "#E2E8F0" if active else "#94A3B8"

            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=label_text,
                showarrow=False,
                font=dict(size=11, color=label_color, family="Arial"),
            )

    # Portfolio position marker
    px, py = result.style_box_position
    x_norm = px / 2.0
    y_norm = py / 2.0

    fig.add_trace(go.Scatter(
        x=[x_norm],
        y=[y_norm],
        mode="markers+text",
        marker=dict(
            color="#6366F1",
            size=25,
            symbol="diamond",
            line=dict(color="white", width=2.5),
        ),
        text=[result.primary_style],
        textposition="top center",
        textfont=dict(color="#E2E8F0", size=12, family="Arial"),
        hovertemplate="<b>%{text}</b><extra></extra>",
        name="현재 포트폴리오",
        showlegend=False,
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="투자 스타일 박스 (Morningstar Style Box)",
            font=dict(size=16, color="#E2E8F0"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=500,
        width=500,
        hovermode="closest",
        xaxis=dict(
            range=[-0.15, 2.15],
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            ticktext=["가치", "혼합", "성장"],
            tickvals=[1/3, 1, 5/3],
            tickfont=dict(size=12, color="#94A3B8"),
            title=dict(text="Value ← → Growth", font=dict(size=12, color="#94A3B8")),
        ),
        yaxis=dict(
            range=[-0.15, 2.15],
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            ticktext=["소형", "중형", "대형"],
            tickvals=[1/3, 1, 5/3],
            tickfont=dict(size=12, color="#94A3B8"),
            title=dict(text="Small ← → Large", font=dict(size=12, color="#94A3B8")),
        ),
        margin=dict(l=80, r=40, t=80, b=80),
    )

    return fig


def create_style_radar(result: StyleResult) -> go.Figure:
    """
    Style characteristics radar chart.

    6 axes: Value, Growth, Momentum, Quality, Dividend, Volatility
    Shows the portfolio's complete style profile.

    Args:
        result: StyleResult from analyze_portfolio_style

    Returns:
        plotly Figure object
    """

    categories = ["가치", "성장", "모멘텀", "퀄리티", "배당", "공격성"]
    values = [
        result.value_score,
        result.growth_score,
        result.momentum_score,
        result.quality_score,
        result.dividend_score,
        result.volatility_score,
    ]

    # Close the polygon
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.3)",
        line=dict(color="#6366F1", width=2.5),
        marker=dict(size=8, color="#6366F1"),
        name="스타일 프로필",
        hovertemplate="<b>%{theta}</b><br>점수: %{r:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="투자 스타일 프로필",
            font=dict(size=16, color="#E2E8F0"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=500,
        width=500,
        polar=dict(
            bgcolor="#1E293B",
            radialaxis=dict(
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=10, color="#94A3B8"),
                gridcolor="#334155",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#94A3B8"),
            ),
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=80, b=60),
    )

    return fig


def create_style_timeline(
    drift_df: pd.DataFrame,
) -> go.Figure:
    """
    Timeline chart showing style evolution over time.

    Shows how value_growth score and momentum change over rolling windows.

    Args:
        drift_df: DataFrame from analyze_style_drift with date, growth_score, momentum_score columns

    Returns:
        plotly Figure object
    """

    if drift_df is None or len(drift_df) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="데이터가 부족합니다",
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    # Growth score line
    if 'growth_score' in drift_df.columns:
        fig.add_trace(go.Scatter(
            x=drift_df['date'],
            y=drift_df['growth_score'],
            mode='lines',
            name='성장점수',
            line=dict(color='#6366F1', width=2.5),
            hovertemplate="<b>성장점수</b><br>날짜: %{x|%Y-%m-%d}<br>점수: %{y:.1f}<extra></extra>",
        ))

    # Momentum score line
    if 'momentum_score' in drift_df.columns:
        fig.add_trace(go.Scatter(
            x=drift_df['date'],
            y=drift_df['momentum_score'],
            mode='lines',
            name='모멘텀',
            line=dict(color='#10B981', width=2.5),
            hovertemplate="<b>모멘텀</b><br>날짜: %{x|%Y-%m-%d}<br>점수: %{y:.1f}<extra></extra>",
        ))

    # Quality score line
    if 'quality_score' in drift_df.columns:
        fig.add_trace(go.Scatter(
            x=drift_df['date'],
            y=drift_df['quality_score'],
            mode='lines',
            name='퀄리티',
            line=dict(color='#F59E0B', width=2.5),
            hovertemplate="<b>퀄리티</b><br>날짜: %{x|%Y-%m-%d}<br>점수: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="투자 스타일 변화 추이",
            font=dict(size=16, color="#E2E8F0"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=400,
        hovermode='x unified',
        xaxis=dict(
            title=dict(text="날짜", font=dict(size=12, color="#94A3B8")),
            showgrid=True,
            gridcolor="#1E293B",
            gridwidth=0.5,
        ),
        yaxis=dict(
            title=dict(text="점수 (0-100)", font=dict(size=12, color="#94A3B8")),
            range=[0, 100],
            showgrid=True,
            gridcolor="#1E293B",
            gridwidth=0.5,
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#475569",
            borderwidth=1,
            font=dict(color="#E2E8F0"),
        ),
        margin=dict(l=70, r=40, t=80, b=60),
    )

    return fig


def create_style_comparison(
    results: Dict[str, StyleResult],
) -> go.Figure:
    """
    Compare style profiles across multiple portfolios.

    Creates a grouped bar chart comparing key style metrics.

    Args:
        results: Dict[portfolio_name] -> StyleResult

    Returns:
        plotly Figure object
    """

    if not results:
        fig = go.Figure()
        fig.add_annotation(text="데이터가 없습니다", showarrow=False)
        return fig

    portfolio_names = list(results.keys())

    metrics = {
        "가치": [results[name].value_score for name in portfolio_names],
        "성장": [results[name].growth_score for name in portfolio_names],
        "모멘텀": [results[name].momentum_score for name in portfolio_names],
        "퀄리티": [results[name].quality_score for name in portfolio_names],
        "배당": [results[name].dividend_score for name in portfolio_names],
    }

    fig = go.Figure()

    colors = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#EC4899"]

    for i, (metric, values) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(
            name=metric,
            x=portfolio_names,
            y=values,
            marker=dict(color=colors[i]),
            text=[f"{v:.0f}" for v in values],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="포트폴리오 스타일 비교",
            font=dict(size=16, color="#E2E8F0"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=400,
        barmode="group",
        xaxis=dict(
            title="포트폴리오",
            showgrid=False,
        ),
        yaxis=dict(
            title="스타일 점수 (0-100)",
            range=[0, 100],
            showgrid=True,
            gridcolor="#1E293B",
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#475569",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=80, b=60),
    )

    return fig


def format_style_result(result: StyleResult) -> Dict[str, str]:
    """
    Format StyleResult into human-readable dictionary for display.

    Args:
        result: StyleResult object

    Returns:
        Dict with formatted strings
    """

    return {
        "주요 스타일": result.primary_style,
        "규모": f"{result.size_style.upper()}",
        "성장성향": "성장주" if result.value_growth == "growth" else ("가치주" if result.value_growth == "value" else "혼합형"),
        "가치점수": f"{result.value_score:.1f}",
        "성장점수": f"{result.growth_score:.1f}",
        "모멘텀": f"{result.momentum_score:.1f}",
        "퀄리티": f"{result.quality_score:.1f}",
        "배당성향": f"{result.dividend_score:.1f}",
        "변동성": f"{result.volatility_score:.1f}",
        "추가특성": ", ".join(result.secondary_traits) if result.secondary_traits else "없음",
        "설명": result.style_description,
    }
