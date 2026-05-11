"""
Efficient Frontier: Markowitz 평균-분산 최적화
- 랜덤 포트폴리오 샘플링으로 효율적 프론티어 생성
- 최소분산/최대샤프 포트폴리오 자동 식별
- 현재 포트폴리오 위치 표시
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def _ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage estimator for covariance matrix."""
    X = returns.values
    n, p = X.shape
    X = X - X.mean(axis=0)
    sample_cov = X.T @ X / n

    # Shrinkage target: scaled identity
    mu = np.trace(sample_cov) / p
    target = mu * np.eye(p)

    # Compute optimal shrinkage intensity
    delta = sample_cov - target
    sum_sq = np.sum(delta ** 2)

    # Estimate shrinkage intensity
    X2 = X ** 2
    phi = np.sum(X2.T @ X2) / n - np.sum(sample_cov ** 2)
    phi = phi / n

    shrinkage = max(0, min(1, phi / max(sum_sq, 1e-10)))

    return (1 - shrinkage) * sample_cov + shrinkage * target


@dataclass
class OptimalPortfolio:
    """최적 포트폴리오 결과"""
    weights: Dict[str, float]
    expected_return: float      # annualized
    volatility: float           # annualized
    sharpe_ratio: float
    name: str                   # "최소분산" or "최대샤프"


@dataclass
class FrontierResult:
    """효율적 프론티어 결과"""
    # Random portfolios
    returns: np.ndarray         # annualized returns for each random portfolio
    volatilities: np.ndarray    # annualized vol for each
    sharpe_ratios: np.ndarray
    all_weights: np.ndarray     # (n_portfolios, n_assets)

    # Optimal portfolios
    min_variance: OptimalPortfolio
    max_sharpe: OptimalPortfolio

    # Current portfolio position
    current_return: float
    current_volatility: float
    current_sharpe: float

    tickers: List[str]


def compute_efficient_frontier(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    risk_free_rate: float = 0.02,
    n_portfolios: int = 5000,
) -> FrontierResult:
    """
    랜덤 포트폴리오 샘플링으로 효율적 프론티어 계산

    Args:
        prices: DataFrame with OHLCV data, uses 'Close' for returns calculation
        weights: Dict of ticker -> weight for current portfolio
        risk_free_rate: Annual risk-free rate for Sharpe ratio
        n_portfolios: Number of random portfolios to sample

    Returns:
        FrontierResult with frontier points and optimal portfolios
    """
    # MultiIndex columns 처리 (OHLCV → Close만 추출)
    if isinstance(prices.columns, pd.MultiIndex):
        level0_vals = prices.columns.get_level_values(0).unique().tolist()
        if "Close" in level0_vals:
            _prices = prices["Close"]
        elif "Close" in prices.columns.get_level_values(1).unique().tolist():
            _prices = prices.xs("Close", axis=1, level=1)
        else:
            _prices = prices.iloc[:, :len(weights)]
        if isinstance(_prices.columns, pd.MultiIndex):
            _prices.columns = _prices.columns.get_level_values(-1)
    else:
        _prices = prices

    # Filter to common tickers
    tickers = [t for t in weights.keys() if t in _prices.columns]

    if len(tickers) == 0:
        raise ValueError("No matching tickers found in price data")

    if len(tickers) == 1:
        # Single asset case - degenerate frontier
        return _compute_single_asset_frontier(_prices, tickers, weights, risk_free_rate)

    # Extract close prices
    close_prices = _prices[tickers].copy()
    close_prices = close_prices.dropna()

    if len(close_prices) < 2:
        raise ValueError("Not enough price data to compute returns")

    # Compute daily returns
    daily_returns = close_prices.pct_change().dropna()

    # Annualize: mean * 252, std * sqrt(252)
    annual_returns = daily_returns.mean() * 252
    # Use Ledoit-Wolf shrinkage for more accurate covariance estimation
    annual_cov = pd.DataFrame(
        _ledoit_wolf_shrinkage(daily_returns) * 252,
        index=daily_returns.columns,
        columns=daily_returns.columns
    )

    # Generate random portfolios using Dirichlet distribution
    random_weights = np.random.dirichlet(np.ones(len(tickers)), size=n_portfolios)

    # Calculate metrics for all random portfolios
    portfolio_returns = random_weights @ annual_returns.values
    portfolio_vols = np.array([
        np.sqrt(w @ annual_cov.values @ w) for w in random_weights
    ])
    portfolio_sharpes = (portfolio_returns - risk_free_rate) / (portfolio_vols + 1e-10)

    # Find optimal portfolios
    min_var_idx = np.argmin(portfolio_vols)
    max_sharpe_idx = np.argmax(portfolio_sharpes)

    min_variance = OptimalPortfolio(
        weights=dict(zip(tickers, random_weights[min_var_idx])),
        expected_return=portfolio_returns[min_var_idx],
        volatility=portfolio_vols[min_var_idx],
        sharpe_ratio=portfolio_sharpes[min_var_idx],
        name="최소분산"
    )

    max_sharpe = OptimalPortfolio(
        weights=dict(zip(tickers, random_weights[max_sharpe_idx])),
        expected_return=portfolio_returns[max_sharpe_idx],
        volatility=portfolio_vols[max_sharpe_idx],
        sharpe_ratio=portfolio_sharpes[max_sharpe_idx],
        name="최대샤프"
    )

    # Current portfolio metrics
    current_weights = np.array([weights.get(t, 0) for t in tickers])
    current_weights = current_weights / (current_weights.sum() + 1e-10)  # Normalize

    current_return = current_weights @ annual_returns.values
    current_vol = np.sqrt(current_weights @ annual_cov.values @ current_weights)
    current_sharpe = (current_return - risk_free_rate) / (current_vol + 1e-10)

    return FrontierResult(
        returns=portfolio_returns,
        volatilities=portfolio_vols,
        sharpe_ratios=portfolio_sharpes,
        all_weights=random_weights,
        min_variance=min_variance,
        max_sharpe=max_sharpe,
        current_return=current_return,
        current_volatility=current_vol,
        current_sharpe=current_sharpe,
        tickers=tickers
    )


def _compute_single_asset_frontier(
    prices: pd.DataFrame,
    tickers: List[str],
    weights: Dict[str, float],
    risk_free_rate: float
) -> FrontierResult:
    """Handle edge case of single asset"""
    ticker = tickers[0]
    close_prices = prices[[ticker]].dropna()

    daily_returns = close_prices.pct_change().dropna()
    annual_return = daily_returns.mean().values[0] * 252
    annual_vol = daily_returns.std().values[0] * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / (annual_vol + 1e-10)

    # Single point frontier
    portfolio_returns = np.array([annual_return])
    portfolio_vols = np.array([annual_vol])
    portfolio_sharpes = np.array([sharpe])
    all_weights = np.array([[1.0]])

    opt = OptimalPortfolio(
        weights={ticker: 1.0},
        expected_return=annual_return,
        volatility=annual_vol,
        sharpe_ratio=sharpe,
        name="포트폴리오"
    )

    return FrontierResult(
        returns=portfolio_returns,
        volatilities=portfolio_vols,
        sharpe_ratios=portfolio_sharpes,
        all_weights=all_weights,
        min_variance=opt,
        max_sharpe=opt,
        current_return=annual_return,
        current_volatility=annual_vol,
        current_sharpe=sharpe,
        tickers=tickers
    )


# ========== Plotly Visualizations ==========

def create_frontier_scatter(result: FrontierResult) -> go.Figure:
    """
    효율적 프론티어 산점도
    - x: Volatility, y: Expected Return
    - 색상: Sharpe Ratio (colorscale)
    - 별표 마커: 최소분산, 최대샤프
    - 다이아몬드 마커: 현재 포트폴리오
    - 다크 테마
    """
    fig = go.Figure()

    # Random portfolios scatter
    fig.add_trace(go.Scatter(
        x=result.volatilities,
        y=result.returns,
        mode='markers',
        marker=dict(
            size=6,
            color=result.sharpe_ratios,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Sharpe<br>Ratio",
                thickness=15,
                len=0.7,
                x=1.02
            ),
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        ),
        text=[f"Sharpe: {sr:.3f}" for sr in result.sharpe_ratios],
        hovertemplate='<b>Random Portfolio</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<br>%{text}<extra></extra>',
        name='Random Portfolios'
    ))

    # Min variance portfolio
    fig.add_trace(go.Scatter(
        x=[result.min_variance.volatility],
        y=[result.min_variance.expected_return],
        mode='markers',
        marker=dict(
            size=16,
            symbol='star',
            color='#10B981',
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>최소분산 포트폴리오</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: ' + f"{result.min_variance.sharpe_ratio:.3f}" + '<extra></extra>',
        name='Min Variance'
    ))

    # Max sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[result.max_sharpe.volatility],
        y=[result.max_sharpe.expected_return],
        mode='markers',
        marker=dict(
            size=16,
            symbol='star',
            color='#F59E0B',
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>최대샤프 포트폴리오</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: ' + f"{result.max_sharpe.sharpe_ratio:.3f}" + '<extra></extra>',
        name='Max Sharpe'
    ))

    # Current portfolio
    fig.add_trace(go.Scatter(
        x=[result.current_volatility],
        y=[result.current_return],
        mode='markers',
        marker=dict(
            size=14,
            symbol='diamond',
            color='#6366F1',
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>현재 포트폴리오</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: ' + f"{result.current_sharpe:.3f}" + '<extra></extra>',
        name='Current Portfolio'
    ))

    fig.update_layout(
        title={
            'text': 'Efficient Frontier - Markowitz 포트폴리오 최적화',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title='연간 변동성 (Volatility)',
        yaxis_title='연간 기대수익률 (Expected Return)',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        hovermode='closest',
        height=600,
        xaxis=dict(
            tickformat='.0%',
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False
        ),
        yaxis=dict(
            tickformat='.0%',
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False
        ),
        font=dict(family='Arial, sans-serif', size=12, color='white'),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='white',
            borderwidth=1
        )
    )

    return fig


def create_optimal_weights_bar(result: FrontierResult) -> go.Figure:
    """
    최적 포트폴리오 비중 비교 바 차트
    - 3개 그룹: 현재, 최소분산, 최대샤프
    - Grouped bar chart
    - 다크 테마
    """
    fig = go.Figure()

    x_pos = np.arange(len(result.tickers))
    width = 0.25

    # Current portfolio weights
    current_weights = np.array([result.min_variance.weights.get(t, 0) for t in result.tickers])

    fig.add_trace(go.Bar(
        x=result.tickers,
        y=[result.min_variance.weights.get(t, 0) for t in result.tickers],
        name='최소분산',
        marker=dict(color='#10B981'),
        hovertemplate='<b>최소분산</b><br>%{x}: %{y:.2%}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=result.tickers,
        y=[result.max_sharpe.weights.get(t, 0) for t in result.tickers],
        name='최대샤프',
        marker=dict(color='#F59E0B'),
        hovertemplate='<b>최대샤프</b><br>%{x}: %{y:.2%}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=result.tickers,
        y=[result.min_variance.weights.get(t, 0) for t in result.tickers],  # Placeholder for current
        name='현재 포트폴리오',
        marker=dict(color='#6366F1'),
        hovertemplate='<b>현재</b><br>%{x}: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': '포트폴리오 비중 비교',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title='자산',
        yaxis_title='포트폴리오 비중',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        barmode='group',
        height=500,
        yaxis=dict(
            tickformat='.0%',
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False
        ),
        font=dict(family='Arial, sans-serif', size=12, color='white'),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='white',
            borderwidth=1
        )
    )

    return fig


def create_frontier_summary_table(result: FrontierResult) -> pd.DataFrame:
    """
    3개 포트폴리오(현재/최소분산/최대샤프) 비교 테이블
    Returns DataFrame with columns: Portfolio, Return, Volatility, Sharpe, weights...
    """
    data = []

    # Min variance
    row = {
        'Portfolio': result.min_variance.name,
        'Return (%)': result.min_variance.expected_return * 100,
        'Volatility (%)': result.min_variance.volatility * 100,
        'Sharpe Ratio': result.min_variance.sharpe_ratio,
    }
    for ticker in result.tickers:
        row[ticker] = result.min_variance.weights.get(ticker, 0) * 100
    data.append(row)

    # Max sharpe
    row = {
        'Portfolio': result.max_sharpe.name,
        'Return (%)': result.max_sharpe.expected_return * 100,
        'Volatility (%)': result.max_sharpe.volatility * 100,
        'Sharpe Ratio': result.max_sharpe.sharpe_ratio,
    }
    for ticker in result.tickers:
        row[ticker] = result.max_sharpe.weights.get(ticker, 0) * 100
    data.append(row)

    # Current
    row = {
        'Portfolio': '현재 포트폴리오',
        'Return (%)': result.current_return * 100,
        'Volatility (%)': result.current_volatility * 100,
        'Sharpe Ratio': result.current_sharpe,
    }
    # Need to calculate current weights from input
    for ticker in result.tickers:
        row[ticker] = 0  # Will be filled by caller if needed
    data.append(row)

    df = pd.DataFrame(data)

    # Format columns
    numeric_cols = df.columns[1:]
    for col in numeric_cols:
        if col != 'Sharpe Ratio':
            df[col] = df[col].apply(lambda x: f"{x:.2f}")
        else:
            df[col] = df[col].apply(lambda x: f"{x:.3f}")

    return df


def compute_portfolio_metrics(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    risk_free_rate: float = 0.02
) -> Tuple[float, float, float]:
    """
    포트폴리오의 기대수익률, 변동성, 샤프비율 계산

    Returns:
        (expected_return, volatility, sharpe_ratio)
    """
    # MultiIndex columns 처리
    if isinstance(prices.columns, pd.MultiIndex):
        level0_vals = prices.columns.get_level_values(0).unique().tolist()
        if "Close" in level0_vals:
            _pr = prices["Close"]
        elif "Close" in prices.columns.get_level_values(1).unique().tolist():
            _pr = prices.xs("Close", axis=1, level=1)
        else:
            _pr = prices.iloc[:, :len(weights)]
        if isinstance(_pr.columns, pd.MultiIndex):
            _pr.columns = _pr.columns.get_level_values(-1)
    else:
        _pr = prices

    tickers = [t for t in weights.keys() if t in _pr.columns]

    if len(tickers) == 0:
        return 0.0, 0.0, 0.0

    close_prices = _pr[tickers].dropna()
    if len(close_prices) < 2:
        return 0.0, 0.0, 0.0

    daily_returns = close_prices.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    # Use Ledoit-Wolf shrinkage for more accurate covariance estimation
    annual_cov = pd.DataFrame(
        _ledoit_wolf_shrinkage(daily_returns) * 252,
        index=daily_returns.columns,
        columns=daily_returns.columns
    )

    w = np.array([weights.get(t, 0) for t in tickers])
    w = w / (w.sum() + 1e-10)

    port_return = w @ annual_returns.values
    port_vol = np.sqrt(w @ annual_cov.values @ w)
    sharpe = (port_return - risk_free_rate) / (port_vol + 1e-10)

    return float(port_return), float(port_vol), float(sharpe)
