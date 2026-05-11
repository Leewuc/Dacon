"""
Performance Metrics: Information Ratio, Tracking Error, Capture Ratios
- Tracking Error: annualized volatility of active returns
- Information Ratio: active return / tracking error
- Up-Capture Ratio: portfolio return in up markets / benchmark return
- Down-Capture Ratio: portfolio return in down markets / benchmark return
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """포트폴리오 성과지표"""
    # Total return metrics
    portfolio_return: float            # Annualized return
    benchmark_return: float            # Annualized return
    active_return: float               # (excess return)

    # Risk metrics
    portfolio_volatility: float        # Annualized volatility
    benchmark_volatility: float        # Annualized volatility
    tracking_error: float              # Annualized active risk

    # Risk-adjusted metrics
    information_ratio: float           # Active return / tracking error
    sharpe_ratio: float                # (portfolio return - rf) / volatility
    benchmark_sharpe: float            # (benchmark return - rf) / volatility

    # Capture ratios
    up_capture_ratio: float            # Portfolio return in bull markets / benchmark
    down_capture_ratio: float          # Portfolio return in bear markets / benchmark

    # Beta and correlation
    beta: float                        # Portfolio beta to benchmark
    correlation: float                 # Correlation with benchmark


def calc_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate annualized tracking error

    TE = sqrt(mean((r_p - r_b)^2)) × sqrt(252)

    Args:
        portfolio_returns: Daily portfolio returns
        benchmark_returns: Daily benchmark returns

    Returns:
        Annualized tracking error
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Align indices
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) == 0:
        return 0.0

    r_p = portfolio_returns.loc[common_idx].values
    r_b = benchmark_returns.loc[common_idx].values

    # Active returns
    active = r_p - r_b

    # TE = sqrt(E[active^2])
    te_squared = np.mean(active ** 2)

    if te_squared < 0:
        return 0.0

    te = np.sqrt(te_squared)

    # Annualize
    return float(te * np.sqrt(252))


def calc_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Information Ratio

    IR = (r_p - r_b) / TE
    where TE = tracking error

    Args:
        portfolio_returns: Daily portfolio returns
        benchmark_returns: Daily benchmark returns
        risk_free_rate: Daily risk-free rate (default 0)

    Returns:
        Information Ratio
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Align indices
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2:
        return 0.0

    r_p = portfolio_returns.loc[common_idx].values
    r_b = benchmark_returns.loc[common_idx].values

    # Annualized returns
    annual_r_p = (1 + r_p.mean()) ** 252 - 1
    annual_r_b = (1 + r_b.mean()) ** 252 - 1

    # Active return
    active_return = annual_r_p - annual_r_b

    # Tracking error
    te = calc_tracking_error(portfolio_returns, benchmark_returns)

    if te < 1e-10:
        return 0.0

    return float(active_return / te)


def calc_up_capture(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate Up-Capture Ratio

    Ratio of portfolio returns during up-market periods (benchmark > 0) to benchmark returns
    UpCapture = Annualized return during bull / Annualized benchmark return during bull

    Args:
        portfolio_returns: Daily portfolio returns
        benchmark_returns: Daily benchmark returns

    Returns:
        Up-capture ratio (>1 means outperformance in bull markets)
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Align indices
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2:
        return 0.0

    r_p = portfolio_returns.loc[common_idx].values
    r_b = benchmark_returns.loc[common_idx].values

    # Filter for up periods (benchmark > 0)
    up_mask = r_b > 0

    if np.sum(up_mask) < 1:
        return 0.0

    r_p_up = r_p[up_mask]
    r_b_up = r_b[up_mask]

    # Annualized returns during up periods
    annual_r_p_up = (1 + r_p_up.mean()) ** 252 - 1
    annual_r_b_up = (1 + r_b_up.mean()) ** 252 - 1

    if annual_r_b_up < 1e-10:
        return 0.0

    return float(annual_r_p_up / annual_r_b_up)


def calc_down_capture(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate Down-Capture Ratio

    Ratio of portfolio returns during down-market periods (benchmark < 0) to benchmark returns
    DownCapture = Annualized return during bear / Annualized benchmark return during bear

    Args:
        portfolio_returns: Daily portfolio returns
        benchmark_returns: Daily benchmark returns

    Returns:
        Down-capture ratio (<1 means less downside in bear markets)
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Align indices
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2:
        return 0.0

    r_p = portfolio_returns.loc[common_idx].values
    r_b = benchmark_returns.loc[common_idx].values

    # Filter for down periods (benchmark < 0)
    down_mask = r_b < 0

    if np.sum(down_mask) < 1:
        return 0.0

    r_p_down = r_p[down_mask]
    r_b_down = r_b[down_mask]

    # Annualized returns during down periods
    annual_r_p_down = (1 + r_p_down.mean()) ** 252 - 1
    annual_r_b_down = (1 + r_b_down.mean()) ** 252 - 1

    if annual_r_b_down > -1e-10:  # Near zero or positive (unusual)
        return 0.0

    return float(annual_r_p_down / annual_r_b_down)


def calc_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate portfolio beta to benchmark

    β = Cov(r_p, r_b) / Var(r_b)

    Args:
        portfolio_returns: Daily portfolio returns
        benchmark_returns: Daily benchmark returns

    Returns:
        Beta
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Align indices
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2:
        return 0.0

    r_p = portfolio_returns.loc[common_idx].values
    r_b = benchmark_returns.loc[common_idx].values

    # Covariance
    cov = np.cov(r_p, r_b)[0, 1]

    # Variance of benchmark
    var_b = np.var(r_b, ddof=1)

    if var_b < 1e-10:
        return 0.0

    return float(cov / var_b)


def calc_correlation(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate correlation between portfolio and benchmark

    Args:
        portfolio_returns: Daily portfolio returns
        benchmark_returns: Daily benchmark returns

    Returns:
        Correlation coefficient [-1, 1]
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Align indices
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2:
        return 0.0

    r_p = portfolio_returns.loc[common_idx].values
    r_b = benchmark_returns.loc[common_idx].values

    corr = np.corrcoef(r_p, r_b)[0, 1]

    # Handle NaN
    if np.isnan(corr):
        return 0.0

    return float(corr)


def calc_all_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> PerformanceMetrics:
    """
    Calculate all performance metrics

    Args:
        portfolio_returns: Daily portfolio returns
        benchmark_returns: Daily benchmark returns
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        PerformanceMetrics dataclass
    """
    # Annualized returns and volatility
    annual_r_p = (1 + portfolio_returns.mean()) ** 252 - 1
    annual_r_b = (1 + benchmark_returns.mean()) ** 252 - 1

    annual_vol_p = portfolio_returns.std() * np.sqrt(252)
    annual_vol_b = benchmark_returns.std() * np.sqrt(252)

    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    # Sharpe ratios
    sharpe_p = (annual_r_p - risk_free_rate) / (annual_vol_p + 1e-10)
    sharpe_b = (annual_r_b - risk_free_rate) / (annual_vol_b + 1e-10)

    # Active metrics
    active_return = annual_r_p - annual_r_b
    te = calc_tracking_error(portfolio_returns, benchmark_returns)
    ir = calc_information_ratio(portfolio_returns, benchmark_returns, daily_rf)

    # Capture ratios
    up_cap = calc_up_capture(portfolio_returns, benchmark_returns)
    down_cap = calc_down_capture(portfolio_returns, benchmark_returns)

    # Beta and correlation
    beta = calc_beta(portfolio_returns, benchmark_returns)
    corr = calc_correlation(portfolio_returns, benchmark_returns)

    return PerformanceMetrics(
        portfolio_return=float(annual_r_p),
        benchmark_return=float(annual_r_b),
        active_return=float(active_return),
        portfolio_volatility=float(annual_vol_p),
        benchmark_volatility=float(annual_vol_b),
        tracking_error=float(te),
        information_ratio=float(ir),
        sharpe_ratio=float(sharpe_p),
        benchmark_sharpe=float(sharpe_b),
        up_capture_ratio=float(up_cap),
        down_capture_ratio=float(down_cap),
        beta=float(beta),
        correlation=float(corr)
    )


# ========== Plotly Visualizations ==========

def create_capture_ratio_chart(metrics: PerformanceMetrics) -> go.Figure:
    """
    Create grouped bar chart showing up-capture and down-capture ratios

    Args:
        metrics: PerformanceMetrics

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    scenarios = ['Up Markets', 'Down Markets']
    capture_ratios = [metrics.up_capture_ratio, metrics.down_capture_ratio]

    colors = ['#10B981' if x > 1 else '#EF4444' for x in capture_ratios]

    fig.add_trace(go.Bar(
        x=scenarios,
        y=capture_ratios,
        marker=dict(color=colors),
        text=[f'{x:.2f}' for x in capture_ratios],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Capture Ratio: %{y:.2f}<extra></extra>'
    ))

    # Add reference line at 1.0
    fig.add_hline(
        y=1.0,
        line_dash='dash',
        line_color='rgba(255,255,255,0.5)',
        annotation_text='Benchmark (1.0)',
        annotation_position='right',
        annotation_font_color='rgba(255,255,255,0.7)'
    )

    fig.update_layout(
        title={
            'text': 'Up/Down Capture Ratios',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        yaxis_title='Capture Ratio',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        height=400,
        showlegend=False,
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False
        ),
        font=dict(family='Arial, sans-serif', size=12, color='white')
    )

    return fig


def create_performance_comparison_chart(metrics: PerformanceMetrics) -> go.Figure:
    """
    Create comparison chart: Portfolio vs Benchmark metrics

    Args:
        metrics: PerformanceMetrics

    Returns:
        Plotly figure
    """
    metrics_to_show = ['Return (%)', 'Volatility (%)', 'Sharpe Ratio']
    portfolio_vals = [
        metrics.portfolio_return * 100,
        metrics.portfolio_volatility * 100,
        metrics.sharpe_ratio
    ]
    benchmark_vals = [
        metrics.benchmark_return * 100,
        metrics.benchmark_volatility * 100,
        metrics.benchmark_sharpe
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics_to_show,
        y=portfolio_vals,
        name='Portfolio',
        marker=dict(color='#10B981'),
        hovertemplate='<b>Portfolio</b><br>%{x}: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=metrics_to_show,
        y=benchmark_vals,
        name='Benchmark',
        marker=dict(color='#3B82F6'),
        hovertemplate='<b>Benchmark</b><br>%{x}: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Portfolio vs Benchmark Performance',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        yaxis_title='Value',
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


def create_metrics_summary_table(metrics: PerformanceMetrics) -> pd.DataFrame:
    """
    Create summary table of key performance metrics

    Args:
        metrics: PerformanceMetrics

    Returns:
        DataFrame
    """
    data = [
        ('Portfolio Return', f"{metrics.portfolio_return:.2%}"),
        ('Benchmark Return', f"{metrics.benchmark_return:.2%}"),
        ('Active Return', f"{metrics.active_return:.2%}"),
        ('', ''),
        ('Portfolio Volatility', f"{metrics.portfolio_volatility:.2%}"),
        ('Benchmark Volatility', f"{metrics.benchmark_volatility:.2%}"),
        ('Tracking Error', f"{metrics.tracking_error:.2%}"),
        ('', ''),
        ('Sharpe Ratio', f"{metrics.sharpe_ratio:.3f}"),
        ('Benchmark Sharpe', f"{metrics.benchmark_sharpe:.3f}"),
        ('Information Ratio', f"{metrics.information_ratio:.3f}"),
        ('', ''),
        ('Beta', f"{metrics.beta:.3f}"),
        ('Correlation', f"{metrics.correlation:.3f}"),
        ('Up Capture', f"{metrics.up_capture_ratio:.2f}"),
        ('Down Capture', f"{metrics.down_capture_ratio:.2f}"),
    ]

    df = pd.DataFrame(data, columns=['Metric', 'Value'])
    return df
