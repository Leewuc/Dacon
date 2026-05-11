"""
Black-Litterman Model for Portfolio Optimization
- Combines market equilibrium implied returns with investor views
- Produces posterior distribution of returns
- Optimal portfolio weights from posterior distribution
- Comparison: market implied vs. posterior returns
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BlackLittermanResult:
    """Black-Litterman 모델 결과"""
    tickers: List[str]
    prior_returns: np.ndarray          # Market-implied equilibrium returns (π)
    posterior_returns: np.ndarray      # Updated returns after views (μ)
    optimal_weights: np.ndarray        # Optimal portfolio weights
    market_weights: np.ndarray         # Initial market cap weights
    views_matrix: Optional[np.ndarray] # P matrix (K x N)
    views_returns: Optional[np.ndarray]# Q vector (K,)
    posterior_cov: np.ndarray          # Posterior covariance matrix
    views_impact: np.ndarray           # Change in weights due to views


def compute_market_implied_returns(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 2.5
) -> np.ndarray:
    """
    Compute market-implied equilibrium returns (π)

    Formula: π = δ × Σ × w
    where:
    - δ = risk aversion coefficient
    - Σ = covariance matrix
    - w = market weights

    Args:
        weights: Market cap weights (N,)
        cov_matrix: Covariance matrix (N x N)
        risk_aversion: Risk aversion coefficient (default 2.5)

    Returns:
        Equilibrium returns vector (N,)
    """
    if len(weights) == 0:
        return np.array([])

    # π = δΣw
    pi = risk_aversion * cov_matrix @ weights

    return pi


def black_litterman_posterior(
    pi: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: Optional[np.ndarray] = None,
    tau: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Black-Litterman posterior returns and covariance

    Formula:
    - Posterior mean: μ = π + Σ P^T (P Σ P^T + Ω)^(-1) (Q - P π)
    - Posterior cov: Σ_BL = Σ + τ Σ = (1 + τ) Σ (approximation)

    Args:
        pi: Prior returns (equilibrium) - (N,)
        sigma: Prior covariance matrix - (N x N)
        P: Views matrix, each row is a view - (K x N)
        Q: Views return targets - (K,)
        omega: Uncertainty of each view, diagonal of Ω - (K,)
               If None, auto-computed as diag(τ P Σ P^T)
        tau: Scaling factor for view uncertainty (default 0.05)

    Returns:
        (posterior_returns, posterior_cov)
    """
    N = len(pi)
    K = len(Q)

    if K == 0:
        # No views - return prior
        return pi.copy(), sigma.copy()

    # Auto-compute omega if not provided
    if omega is None:
        # ω_k = τ × (P_k Σ P_k^T)
        PSP = np.diag([P[k] @ sigma @ P[k] for k in range(K)])
        omega = tau * np.diag(np.diag(PSP))
    else:
        # Convert to diagonal matrix if vector
        if omega.ndim == 1:
            omega = np.diag(omega)

    # Compute posterior mean
    # M = (P Σ P^T + Ω)^(-1)
    PSP = P @ sigma @ P.T
    M_inv = PSP + omega

    # Add small regularization for numerical stability
    M_inv = M_inv + 1e-10 * np.eye(K)

    try:
        M = np.linalg.inv(M_inv)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse if singular
        M = np.linalg.pinv(M_inv)

    # Posterior mean: μ = π + Σ P^T (P Σ P^T + Ω)^(-1) (Q - P π)
    view_residual = Q - P @ pi
    posterior_returns = pi + sigma @ P.T @ M @ view_residual

    # Posterior covariance: Σ_BL ≈ (1 + τ) Σ
    # (More accurate formula involves matrix algebra, use approximation)
    posterior_cov = (1 + tau) * sigma

    return posterior_returns, posterior_cov


def optimize_bl_portfolio(
    posterior_returns: np.ndarray,
    posterior_cov: np.ndarray,
    risk_free_rate: float = 0.02,
    method: str = "analytical"
) -> np.ndarray:
    """
    Optimize portfolio using posterior returns and covariance

    Solves: max w^T μ - (λ/2) w^T Σ w
    Analytical solution: w = (1/λ) Σ^(-1) (μ - r_f)
    where λ = risk aversion (computed as 1/2 for max Sharpe)

    Args:
        posterior_returns: Posterior expected returns (N,)
        posterior_cov: Posterior covariance matrix (N x N)
        risk_free_rate: Risk-free rate for Sharpe ratio
        method: "analytical" (default) or "grid"

    Returns:
        Optimal weights (N,)
    """
    N = len(posterior_returns)

    if N == 0:
        return np.array([])

    if N == 1:
        return np.array([1.0])

    # Regularize covariance for numerical stability
    cov_reg = posterior_cov + 1e-10 * np.eye(N)

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)

    # Analytical solution: w ∝ Σ^(-1) (μ - r_f)
    excess_returns = posterior_returns - risk_free_rate
    weights = cov_inv @ excess_returns

    # Normalize to sum to 1
    weights_sum = np.sum(weights)
    if abs(weights_sum) < 1e-10:
        # Degenerate case: equal weights
        weights = np.ones(N) / N
    else:
        weights = weights / weights_sum

    # Clip negative weights to 0 (no shorting)
    weights = np.maximum(weights, 0)
    weights = weights / (np.sum(weights) + 1e-10)

    return weights


def black_litterman_analysis(
    prices: pd.DataFrame,
    market_weights: Dict[str, float],
    views: Optional[List[Dict]] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    risk_free_rate: float = 0.02
) -> BlackLittermanResult:
    """
    Complete Black-Litterman analysis

    Args:
        prices: Historical price data with tickers as columns
        market_weights: Market cap weights dict {ticker: weight}
        views: List of views, each dict with:
            - 'P': List of weights (view expression)
            - 'Q': Expected return of view
            Example: {'P': [0.5, -0.5, 0], 'Q': 0.05} means 50% asset A vs 50% asset B has +5% expected outperformance
        risk_aversion: Risk aversion coefficient
        tau: Uncertainty scaling for views
        risk_free_rate: Risk-free rate

    Returns:
        BlackLittermanResult
    """
    # MultiIndex columns 처리 (OHLCV → Close만 추출)
    if isinstance(prices.columns, pd.MultiIndex):
        level0_vals = prices.columns.get_level_values(0).unique().tolist()
        if "Close" in level0_vals:
            _prices = prices["Close"]
        elif "Close" in prices.columns.get_level_values(1).unique().tolist():
            _prices = prices.xs("Close", axis=1, level=1)
        else:
            _prices = prices.iloc[:, :len(market_weights)]
        if isinstance(_prices.columns, pd.MultiIndex):
            _prices.columns = _prices.columns.get_level_values(-1)
    else:
        _prices = prices

    tickers = list(market_weights.keys())
    tickers = [t for t in tickers if t in _prices.columns]

    if len(tickers) == 0:
        raise ValueError("No matching tickers found in price data")

    # Extract close prices and compute returns
    close_prices = _prices[tickers].dropna()
    if len(close_prices) < 2:
        raise ValueError("Not enough price data")

    daily_returns = close_prices.pct_change().dropna()

    # Annualize
    annual_returns = daily_returns.mean() * 252
    annual_cov = daily_returns.cov() * 252

    # Market weights normalized
    w_market = np.array([market_weights.get(t, 0) for t in tickers])
    w_market = w_market / (np.sum(w_market) + 1e-10)

    # Step 1: Compute prior (market-implied) returns
    pi = compute_market_implied_returns(w_market, annual_cov.values, risk_aversion)

    # Step 2: Process views into P and Q matrices
    P_matrix = None
    Q_vector = None

    if views and len(views) > 0:
        P_list = []
        Q_list = []

        for view in views:
            if 'P' in view and 'Q' in view:
                P_list.append(view['P'])
                Q_list.append(view['Q'])

        if len(P_list) > 0:
            P_matrix = np.array(P_list)
            Q_vector = np.array(Q_list)

    # Step 3: Compute posterior if views exist
    if P_matrix is not None and Q_vector is not None:
        posterior_returns, posterior_cov = black_litterman_posterior(
            pi, annual_cov.values, P_matrix, Q_vector, tau=tau
        )
    else:
        posterior_returns = pi.copy()
        posterior_cov = annual_cov.values.copy()

    # Step 4: Optimize portfolio with posterior
    optimal_weights = optimize_bl_portfolio(posterior_returns, posterior_cov, risk_free_rate)

    # Compute impact of views (change in weights)
    w_prior = optimize_bl_portfolio(pi, annual_cov.values, risk_free_rate)
    views_impact = optimal_weights - w_prior

    return BlackLittermanResult(
        tickers=tickers,
        prior_returns=pi,
        posterior_returns=posterior_returns,
        optimal_weights=optimal_weights,
        market_weights=w_market,
        views_matrix=P_matrix,
        views_returns=Q_vector,
        posterior_cov=posterior_cov,
        views_impact=views_impact
    )


# ========== Plotly Visualizations ==========

def create_bl_comparison_chart(result: BlackLittermanResult) -> go.Figure:
    """
    Bar chart comparing prior (market-implied) vs posterior (BL) returns

    Args:
        result: BlackLittermanResult

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=result.tickers,
        y=result.prior_returns * 100,
        name='Market-Implied Returns (Prior)',
        marker=dict(color='#3B82F6'),
        hovertemplate='<b>%{x}</b><br>Prior: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=result.tickers,
        y=result.posterior_returns * 100,
        name='BL Posterior Returns',
        marker=dict(color='#10B981'),
        hovertemplate='<b>%{x}</b><br>Posterior: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Black-Litterman: Prior vs Posterior Returns',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title='Asset',
        yaxis_title='Expected Return (%)',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        barmode='group',
        height=500,
        hovermode='x unified',
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.2)'
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


def create_bl_weights_chart(
    result: BlackLittermanResult,
    tickers: Optional[List[str]] = None
) -> go.Figure:
    """
    Grouped bar chart: Market weights vs BL optimal weights vs Prior optimal weights

    Args:
        result: BlackLittermanResult
        tickers: Optional override of ticker order

    Returns:
        Plotly figure
    """
    if tickers is None:
        tickers = result.tickers

    # Compute prior optimal weights for comparison
    w_prior = optimize_bl_portfolio(result.prior_returns, result.posterior_cov, 0.02)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tickers,
        y=result.market_weights * 100,
        name='Market Weights',
        marker=dict(color='#6366F1'),
        hovertemplate='<b>%{x}</b><br>Market: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=tickers,
        y=w_prior * 100,
        name='Prior Optimal (Equilibrium)',
        marker=dict(color='#3B82F6'),
        hovertemplate='<b>%{x}</b><br>Prior Optimal: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=tickers,
        y=result.optimal_weights * 100,
        name='BL Posterior Optimal',
        marker=dict(color='#10B981'),
        hovertemplate='<b>%{x}</b><br>BL Optimal: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Black-Litterman: Weight Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title='Asset',
        yaxis_title='Portfolio Weight (%)',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        barmode='group',
        height=500,
        hovermode='x unified',
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.2)'
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


def create_bl_impact_chart(result: BlackLittermanResult) -> go.Figure:
    """
    Waterfall chart showing impact of views on portfolio weights

    Args:
        result: BlackLittermanResult

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    colors = ['#10B981' if x > 0 else '#EF4444' for x in result.views_impact * 100]

    fig.add_trace(go.Bar(
        x=result.tickers,
        y=result.views_impact * 100,
        marker=dict(color=colors),
        hovertemplate='<b>%{x}</b><br>Weight Change: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Impact of Views on Portfolio Weights',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title='Asset',
        yaxis_title='Weight Change (%)',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        height=400,
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.5)'
        ),
        font=dict(family='Arial, sans-serif', size=12, color='white'),
        showlegend=False
    )

    return fig


def create_bl_summary_table(result: BlackLittermanResult) -> pd.DataFrame:
    """
    Summary table of Black-Litterman results

    Args:
        result: BlackLittermanResult

    Returns:
        DataFrame with columns: Ticker, Market Weight, Prior Return, Posterior Return, BL Weight
    """
    data = []

    for i, ticker in enumerate(result.tickers):
        data.append({
            'Ticker': ticker,
            'Market Weight (%)': result.market_weights[i] * 100,
            'Prior Return (%)': result.prior_returns[i] * 100,
            'Posterior Return (%)': result.posterior_returns[i] * 100,
            'BL Weight (%)': result.optimal_weights[i] * 100,
            'Weight Change (%)': result.views_impact[i] * 100
        })

    df = pd.DataFrame(data)

    # Format
    for col in ['Market Weight (%)', 'Prior Return (%)', 'Posterior Return (%)', 'BL Weight (%)', 'Weight Change (%)']:
        df[col] = df[col].apply(lambda x: f"{x:.2f}")

    return df
