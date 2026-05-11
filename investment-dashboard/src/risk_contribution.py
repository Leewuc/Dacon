"""
Risk Contribution Decomposition
- Marginal Risk Contribution (MRC)
- Component Risk Contribution (CRC)
- Risk Budgeting analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RiskContributionResult:
    """리스크 기여도 분석 결과"""
    tickers: List[str]
    weights: np.ndarray
    marginal_risk: np.ndarray        # MRC: ∂σ/∂w_i
    component_risk: np.ndarray       # CRC: w_i × MRC_i
    pct_contribution: np.ndarray     # CRC_i / σ_p × 100
    portfolio_vol: float             # annualized vol
    diversification_ratio: float     # sum(w_i × σ_i) / σ_p — higher = less diversified
    concentration_risk: float        # max(pct_contribution) — how concentrated risk is


def compute_risk_contribution(
    prices: pd.DataFrame,
    weights: Dict[str, float],
) -> RiskContributionResult:
    """
    Compute risk contribution for each holding.

    Math:
    - Portfolio variance: σ²_p = w^T Σ w
    - Portfolio vol: σ_p = sqrt(w^T Σ w)
    - Marginal Risk Contribution: MRC_i = (Σ w)_i / σ_p
    - Component Risk Contribution: CRC_i = w_i × MRC_i
    - Sum of all CRC = σ_p (Euler decomposition)
    - Percentage: pct_i = CRC_i / σ_p × 100

    Diversification Ratio = Σ(w_i × σ_i) / σ_p
    If ratio = 1.0, no diversification benefit
    If ratio > 1.0, diversification is reducing risk

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data with tickers as columns
    weights : Dict[str, float]
        Position weights by ticker

    Returns
    -------
    RiskContributionResult
        Risk contribution decomposition
    """
    # MultiIndex columns 처리 (OHLCV → Close만 추출)
    if isinstance(prices.columns, pd.MultiIndex):
        level0_vals = prices.columns.get_level_values(0).unique().tolist()
        level1_vals = prices.columns.get_level_values(1).unique().tolist()
        if "Close" in level0_vals:
            close_prices = prices["Close"]
        elif "Close" in level1_vals:
            close_prices = prices.xs("Close", axis=1, level=1)
        else:
            close_prices = prices.iloc[:, :len(weights)]
        # flatten remaining MultiIndex
        if isinstance(close_prices.columns, pd.MultiIndex):
            close_prices.columns = close_prices.columns.get_level_values(-1)
    else:
        close_prices = prices

    # Filter to tickers in price data
    tickers = [t for t in weights.keys() if t in close_prices.columns]

    # Handle edge cases
    if len(tickers) == 0:
        raise ValueError("No tickers found in price data")

    if len(tickers) == 1:
        # Single asset: all risk is from that asset
        ticker = tickers[0]
        returns = close_prices[ticker].pct_change().dropna()
        ann_vol = returns.std() * np.sqrt(252)

        return RiskContributionResult(
            tickers=[ticker],
            weights=np.array([1.0]),
            marginal_risk=np.array([ann_vol]),
            component_risk=np.array([ann_vol]),
            pct_contribution=np.array([100.0]),
            portfolio_vol=ann_vol,
            diversification_ratio=1.0,
            concentration_risk=100.0,
        )

    # Daily returns
    returns = close_prices[tickers].pct_change().dropna()

    if len(returns) < 2:
        raise ValueError("Insufficient price history (minimum 2 observations)")

    # Annualized covariance matrix (252 trading days)
    cov = returns.cov().values * 252

    # Weight vector
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()  # normalize to sum to 1

    # Portfolio variance and volatility
    port_var = w @ cov @ w
    port_vol = np.sqrt(max(port_var, 0))  # ensure non-negative

    if port_vol < 1e-10:
        # Degenerate case: no volatility
        return RiskContributionResult(
            tickers=tickers,
            weights=w,
            marginal_risk=np.zeros(len(tickers)),
            component_risk=np.zeros(len(tickers)),
            pct_contribution=np.full(len(tickers), 100.0 / len(tickers)),
            portfolio_vol=0.0,
            diversification_ratio=1.0,
            concentration_risk=100.0 / len(tickers),
        )

    # Marginal risk contribution: ∂σ/∂w_i = (Σ w)_i / σ_p
    mrc = (cov @ w) / port_vol

    # Component risk contribution: CRC_i = w_i × MRC_i
    crc = w * mrc

    # Percentage contribution: pct_i = CRC_i / σ_p × 100
    # Note: sum(crc) = port_vol, so sum(pct) should be ~100
    pct = (crc / port_vol) * 100

    # Individual volatilities for diversification ratio
    individual_vols = np.sqrt(np.diag(cov))
    div_ratio = np.sum(w * individual_vols) / port_vol if port_vol > 0 else 1.0

    # Concentration: maximum risk contribution
    concentration = float(np.max(np.abs(pct)))

    return RiskContributionResult(
        tickers=tickers,
        weights=w,
        marginal_risk=mrc,
        component_risk=crc,
        pct_contribution=pct,
        portfolio_vol=port_vol,
        diversification_ratio=div_ratio,
        concentration_risk=concentration,
    )


# ========== Visualizations ==========

def create_risk_contribution_bar(result: RiskContributionResult) -> "go.Figure":
    """
    Risk vs Weight comparison bar chart
    - Grouped bar: Weight % vs Risk Contribution %
    - Sorted by risk contribution (descending)
    - Highlight overweight-risk positions (risk% > weight% significantly)
    - Dark theme

    Parameters
    ----------
    result : RiskContributionResult
        Risk contribution analysis result

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    import plotly.graph_objects as go

    # Sort by risk contribution (descending)
    sorted_idx = np.argsort(-result.pct_contribution)
    tickers = [result.tickers[i] for i in sorted_idx]
    weight_pcts = result.weights[sorted_idx] * 100
    risk_pcts = result.pct_contribution[sorted_idx]

    fig = go.Figure()

    # Weight bars
    fig.add_trace(go.Bar(
        y=tickers,
        x=weight_pcts,
        orientation='h',
        name='비중 (%)',
        marker=dict(color='#6366F1'),
        text=[f'{v:.1f}%' for v in weight_pcts],
        textposition='outside',
        textfont=dict(size=10, color='#A5B4FC'),
    ))

    # Risk contribution bars
    fig.add_trace(go.Bar(
        y=tickers,
        x=risk_pcts,
        orientation='h',
        name='리스크 기여 (%)',
        marker=dict(color='#EF4444'),
        text=[f'{v:.1f}%' for v in risk_pcts],
        textposition='outside',
        textfont=dict(size=10, color='#FCA5A5'),
    ))

    fig.update_layout(
        title='비중 vs 리스크 기여도',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        height=max(350, len(tickers) * 40 + 100),
        barmode='group',
        xaxis_title='%',
        yaxis_title='',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
        ),
        margin=dict(l=100, b=100),
        hovermode='closest',
    )

    return fig


def create_risk_donut(result: RiskContributionResult) -> "go.Figure":
    """
    Risk contribution donut chart
    - Size = risk contribution percentage
    - Colors by risk level (high risk = redder)
    - Center text: portfolio vol

    Parameters
    ----------
    result : RiskContributionResult
        Risk contribution analysis result

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    import plotly.graph_objects as go

    # Sort by risk contribution (descending)
    sorted_idx = np.argsort(-result.pct_contribution)
    labels = [result.tickers[i] for i in sorted_idx]
    values = [max(0.1, result.pct_contribution[i]) for i in sorted_idx]

    # Color gradient: low risk = green, high risk = red
    max_pct = max(values) if values else 1.0
    colors = []
    for v in values:
        ratio = v / max_pct if max_pct > 0 else 0
        if ratio > 0.7:
            colors.append('#EF4444')
        elif ratio > 0.4:
            colors.append('#F59E0B')
        else:
            colors.append('#10B981')

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(
            colors=colors,
            line=dict(color='#0F172A', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=11, color='#E2E8F0'),
        hovertemplate='<b>%{label}</b><br>리스크 기여: %{value:.1f}%<extra></extra>',
    ))

    fig.update_layout(
        title='리스크 기여도 분해',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        height=400,
        annotations=[dict(
            text=f'σ={result.portfolio_vol*100:.1f}%',
            x=0.5,
            y=0.5,
            font=dict(size=16, color='#E2E8F0'),
            showarrow=False,
        )],
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def create_risk_budget_table(result: RiskContributionResult) -> pd.DataFrame:
    """
    Create a detailed risk budget table

    Parameters
    ----------
    result : RiskContributionResult
        Risk contribution analysis result

    Returns
    -------
    pd.DataFrame
        Risk budget details with status indicators
    """
    rows = []
    sorted_idx = np.argsort(-result.pct_contribution)

    for i in sorted_idx:
        ticker = result.tickers[i]
        weight = result.weights[i] * 100
        risk_pct = result.pct_contribution[i]
        mrc = result.marginal_risk[i] * 100
        crc = result.component_risk[i] * 100

        # Risk/Weight ratio: how much risk per unit of weight
        rw_ratio = risk_pct / weight if weight > 0.01 else 0

        # Status based on risk/weight ratio
        if rw_ratio > 1.5:
            status = "⚠️ 과다"
        elif rw_ratio > 0.7:
            status = "✅ 적정"
        else:
            status = "🟢 과소"

        rows.append({
            "종목": ticker,
            "비중(%)": f"{weight:.1f}",
            "리스크 기여(%)": f"{risk_pct:.1f}",
            "MRC": f"{mrc:.2f}",
            "CRC": f"{crc:.2f}",
            "리스크/비중": f"{rw_ratio:.2f}x",
            "상태": status,
        })

    # Add summary row
    rows.append({
        "종목": "합계",
        "비중(%)": f"{result.weights.sum()*100:.1f}",
        "리스크 기여(%)": f"{result.pct_contribution.sum():.1f}",
        "MRC": "—",
        "CRC": f"{result.component_risk.sum()*100:.2f}",
        "리스크/비중": "—",
        "상태": "—",
    })

    return pd.DataFrame(rows)


def get_risk_summary(result: RiskContributionResult) -> Dict:
    """
    Get summary statistics from risk contribution analysis

    Parameters
    ----------
    result : RiskContributionResult
        Risk contribution analysis result

    Returns
    -------
    Dict
        Summary statistics including diversification and concentration metrics
    """
    return {
        "portfolio_volatility": round(result.portfolio_vol * 100, 2),
        "concentration_risk": round(result.concentration_risk, 1),
        "diversification_ratio": round(result.diversification_ratio, 2),
        "num_holdings": len(result.tickers),
        "herfindahl_index": float(np.sum(result.pct_contribution**2) / 10000),
        "max_risk_contributor": result.tickers[np.argmax(result.pct_contribution)],
        "max_risk_contribution": round(float(np.max(result.pct_contribution)), 1),
    }


def identify_risk_outliers(
    result: RiskContributionResult,
    threshold: float = 1.5
) -> Dict[str, List[str]]:
    """
    Identify holdings with outsized risk contributions relative to their weight

    Parameters
    ----------
    result : RiskContributionResult
        Risk contribution analysis result
    threshold : float
        Risk/weight ratio threshold for outliers (default 1.5)

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with "overweight_risk" (high risk relative to weight) and
        "underweight_risk" (low risk relative to weight) holdings
    """
    overweight = []
    underweight = []

    for i, ticker in enumerate(result.tickers):
        weight_pct = result.weights[i] * 100
        risk_pct = result.pct_contribution[i]

        if weight_pct > 0.01:
            ratio = risk_pct / weight_pct
            if ratio > threshold:
                overweight.append(ticker)
            elif ratio < (1.0 / threshold):
                underweight.append(ticker)

    return {
        "overweight_risk": overweight,
        "underweight_risk": underweight,
    }
