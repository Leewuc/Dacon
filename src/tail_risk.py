"""
Advanced tail risk analysis module for investment dashboards.

Provides Cornish-Fisher VaR, CVaR/Expected Shortfall, and comprehensive tail risk metrics.
Pure Python implementation using numpy, scipy, and pandas only.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, skew, kurtosis, jarque_bera, gaussian_kde


COLORS = {
    "background": "#0F172A",
    "surface": "#1E293B",
    "text": "#F8FAFC",
    "grid": "#334155",
    "primary": "#6366F1",
    "negative": "#EF4444",
    "positive": "#10B981",
    "accent": "#EC4899",
}


@dataclass
class TailRiskResult:
    """Container for tail risk analysis results."""
    gaussian_var: float
    cornish_fisher_var: float
    historical_var: float
    cvar: float
    skewness: float
    kurtosis_excess: float
    tail_ratio: float
    max_loss: float
    jarque_bera_pvalue: float
    left_tail_heaviness: float
    cf_improvement: float


def cornish_fisher_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute VaR using Cornish-Fisher expansion accounting for skewness and kurtosis.

    The Cornish-Fisher adjustment modifies the normal quantile to account for
    higher moments in the return distribution.

    CF adjustment formula:
    z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36

    where:
    - z = normal quantile at the given confidence level
    - S = skewness of returns
    - K = excess kurtosis of returns

    Args:
        returns: Series of returns (typically daily or periodic)
        confidence: Confidence level (default 0.95 for 95% VaR)

    Returns:
        Dictionary containing:
        - gaussian_var: Standard normal VaR
        - cf_var: Cornish-Fisher adjusted VaR
        - historical_var: Empirical quantile VaR
        - skewness: Skewness of returns
        - kurtosis: Excess kurtosis of returns
        - cf_improvement: Percentage improvement over Gaussian VaR
    """
    returns_clean = returns.dropna()

    # Compute moments
    S = skew(returns_clean)  # Skewness
    K = kurtosis(returns_clean)  # Excess kurtosis (scipy default)

    # Normal quantile at (1 - confidence)
    alpha = 1 - confidence
    z = norm.ppf(alpha)

    # Cornish-Fisher adjustment
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3*z) * K / 24 - (2*z**3 - 5*z) * S**2 / 36

    # Calculate standard deviation
    std = returns_clean.std()

    # VaR values (expressed as negative losses)
    gaussian_var = z * std
    cf_var = z_cf * std

    # Historical/empirical VaR
    historical_var = returns_clean.quantile(alpha)

    # CF improvement as percentage
    cf_improvement = ((cf_var - gaussian_var) / abs(gaussian_var)) * 100

    return {
        "gaussian_var": gaussian_var,
        "cf_var": cf_var,
        "historical_var": historical_var,
        "skewness": S,
        "kurtosis": K,
        "cf_improvement": cf_improvement,
    }


def multi_confidence_var(
    returns: pd.Series,
    confidences: List[float] = None,
) -> pd.DataFrame:
    """
    Compare Gaussian, Cornish-Fisher, and Historical VaR at multiple confidence levels.

    Args:
        returns: Series of returns
        confidences: List of confidence levels (default: [0.90, 0.95, 0.99])

    Returns:
        DataFrame with columns: confidence, gaussian, cornish_fisher, historical
    """
    if confidences is None:
        confidences = [0.90, 0.95, 0.99]

    results = []

    for conf in confidences:
        var_dict = cornish_fisher_var(returns, conf)
        results.append({
            "confidence": f"{conf*100:.0f}%",
            "gaussian": var_dict["gaussian_var"],
            "cornish_fisher": var_dict["cf_var"],
            "historical": var_dict["historical_var"],
        })

    return pd.DataFrame(results)


def tail_risk_analysis(
    returns: pd.Series,
    confidence: float = 0.95,
) -> TailRiskResult:
    """
    Comprehensive tail risk analysis with multiple metrics and statistics.

    Computes:
    - VaR using three methods (Gaussian, Cornish-Fisher, Historical)
    - CVaR / Expected Shortfall (average loss beyond VaR)
    - Tail ratio: abs(95th percentile) / abs(5th percentile)
    - Maximum loss
    - Jarque-Bera normality test (p-value)
    - Left tail heaviness indicator

    Args:
        returns: Series of returns
        confidence: Confidence level (default 0.95)

    Returns:
        TailRiskResult dataclass with all metrics
    """
    returns_clean = returns.dropna()

    # Compute VaR using all methods
    var_dict = cornish_fisher_var(returns_clean, confidence)

    # CVaR / Expected Shortfall
    alpha = 1 - confidence
    var_threshold = returns_clean.quantile(alpha)
    cvar = returns_clean[returns_clean <= var_threshold].mean()

    # Tail ratio: ratio of absolute values of upper vs lower tails
    p95 = returns_clean.quantile(0.95)
    p5 = returns_clean.quantile(0.05)
    tail_ratio = abs(p95) / abs(p5) if p5 != 0 else np.inf

    # Maximum loss
    max_loss = returns_clean.min()

    # Jarque-Bera test for normality
    jb_stat, jb_pvalue = jarque_bera(returns_clean)

    # Left tail heaviness: ratio of left tail to right tail
    # Compare distance of 5th percentile vs 95th percentile from median
    median = returns_clean.median()
    left_distance = abs(p5 - median)
    right_distance = abs(p95 - median)
    left_tail_heaviness = left_distance / right_distance if right_distance != 0 else np.inf

    return TailRiskResult(
        gaussian_var=var_dict["gaussian_var"],
        cornish_fisher_var=var_dict["cf_var"],
        historical_var=var_dict["historical_var"],
        cvar=cvar,
        skewness=var_dict["skewness"],
        kurtosis_excess=var_dict["kurtosis"],
        tail_ratio=tail_ratio,
        max_loss=max_loss,
        jarque_bera_pvalue=jb_pvalue,
        left_tail_heaviness=left_tail_heaviness,
        cf_improvement=var_dict["cf_improvement"],
    )


def create_var_comparison_chart(
    returns: pd.Series,
    confidences: List[float] = None,
) -> go.Figure:
    """
    Bar chart comparing Gaussian vs Cornish-Fisher vs Historical VaR.

    Creates grouped bars at each confidence level showing the three VaR methods.
    Uses dark theme with highlighted accent color for CF VaR.

    Args:
        returns: Series of returns
        confidences: List of confidence levels (default: [0.90, 0.95, 0.99])

    Returns:
        Plotly Figure object
    """
    if confidences is None:
        confidences = [0.90, 0.95, 0.99]

    df = multi_confidence_var(returns, confidences)

    fig = go.Figure()

    # Add bars for each method
    fig.add_trace(go.Bar(
        x=df["confidence"],
        y=df["gaussian"],
        name="Gaussian",
        marker=dict(color=COLORS["grid"]),
        hovertemplate="<b>Gaussian VaR</b><br>%{x}<br>%{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=df["confidence"],
        y=df["cornish_fisher"],
        name="Cornish-Fisher",
        marker=dict(color=COLORS["accent"]),
        hovertemplate="<b>Cornish-Fisher VaR</b><br>%{x}<br>%{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=df["confidence"],
        y=df["historical"],
        name="Historical",
        marker=dict(color=COLORS["negative"]),
        hovertemplate="<b>Historical VaR</b><br>%{x}<br>%{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="VaR Comparison: Gaussian vs Cornish-Fisher vs Historical",
        xaxis_title="Confidence Level",
        yaxis_title="VaR (Daily Return Impact)",
        barmode="group",
        height=400,
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", size=12, color=COLORS["text"]),
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            color=COLORS["text"],
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=COLORS["grid"],
            borderwidth=1,
        ),
    )

    return fig


def create_return_distribution(
    returns: pd.Series,
) -> go.Figure:
    """
    Histogram of returns with overlays for normal distribution and VaR thresholds.

    Shows:
    - Histogram of actual returns
    - Normal distribution curve (red dashed)
    - Kernel density estimate (solid line)
    - VaR lines at 95% and 99% confidence
    - Shaded tail region below VaR
    - Annotated skewness and kurtosis

    Args:
        returns: Series of returns

    Returns:
        Plotly Figure object
    """
    returns_clean = returns.dropna()

    # Get statistics
    result = tail_risk_analysis(returns_clean, 0.95)

    # Create histogram
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns_clean,
        nbinsx=50,
        name="Returns",
        marker=dict(color=COLORS["primary"], opacity=0.7),
        hovertemplate="<b>Return Range</b><br>%{x:.4f}<br>Count: %{y}<extra></extra>",
    ))

    # Add normal distribution overlay
    x_range = np.linspace(returns_clean.min(), returns_clean.max(), 200)
    normal_dist = norm.pdf(x_range, returns_clean.mean(), returns_clean.std())

    # Scale normal distribution to match histogram
    hist_max = np.histogram(returns_clean, bins=50)[0].max()
    bin_width = (returns_clean.max() - returns_clean.min()) / 50
    normal_scaled = normal_dist * hist_max * bin_width

    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_scaled,
        name="Normal Distribution",
        mode="lines",
        line=dict(color=COLORS["negative"], dash="dash", width=2),
        hovertemplate="<b>Normal PDF</b><br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
    ))

    # Add kernel density estimate
    try:
        kde = gaussian_kde(returns_clean)
        kde_values = kde(x_range) * hist_max * bin_width
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_values,
            name="KDE",
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            hovertemplate="<b>Kernel Density</b><br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
        ))
    except Exception:
        pass  # Skip KDE if insufficient data

    # Add VaR lines
    var_95 = result.historical_var
    var_99 = cornish_fisher_var(returns_clean, 0.99)["historical_var"]

    fig.add_vline(
        x=var_95,
        line_dash="solid",
        line_color=COLORS["negative"],
        annotation_text="95% VaR",
        annotation_position="top",
        annotation=dict(textangle=-90),
    )

    fig.add_vline(
        x=var_99,
        line_dash="dot",
        line_color=COLORS["accent"],
        annotation_text="99% VaR",
        annotation_position="top",
        annotation=dict(textangle=-90),
    )

    # Annotation with statistics
    stats_text = (
        f"<b>Distribution Statistics</b><br>"
        f"Skewness: {result.skewness:.3f}<br>"
        f"Kurtosis: {result.kurtosis_excess:.3f}<br>"
        f"JB p-value: {result.jarque_bera_pvalue:.4f}"
    )

    fig.add_annotation(
        text=stats_text,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.97,
        showarrow=False,
        bgcolor="rgba(30, 41, 59, 0.8)",
        bordercolor=COLORS["grid"],
        borderwidth=1,
        borderpad=10,
        font=dict(size=11, color=COLORS["text"]),
        align="left",
        xanchor="right",
        yanchor="top",
    )

    fig.update_layout(
        title="Return Distribution with Normal Overlay and VaR Thresholds",
        xaxis_title="Daily Returns",
        yaxis_title="Frequency",
        height=400,
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", size=12, color=COLORS["text"]),
        hovermode="x unified",
        showlegend=True,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=COLORS["grid"],
            borderwidth=1,
        ),
    )

    return fig


def create_tail_qq_plot(
    returns: pd.Series,
) -> go.Figure:
    """
    Q-Q plot comparing actual returns vs theoretical normal distribution.

    Points deviating from the 45-degree line indicate departures from normality.
    Fat tails appear as divergence at the extremes.

    Args:
        returns: Series of returns

    Returns:
        Plotly Figure object
    """
    returns_clean = returns.dropna().sort_values().reset_index(drop=True)

    n = len(returns_clean)
    # Theoretical quantiles from standard normal
    theoretical_quantiles = norm.ppf(np.arange(1, n + 1) / (n + 1))

    # Sample quantiles (standardized)
    sample_quantiles = (returns_clean - returns_clean.mean()) / returns_clean.std()

    # Create figure
    fig = go.Figure()

    # Add scatter plot of actual quantiles
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles.values,
        mode="markers",
        name="Actual vs Normal",
        marker=dict(
            size=6,
            color=COLORS["primary"],
            opacity=0.7,
        ),
        hovertemplate="<b>Q-Q Plot</b><br>Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>",
    ))

    # Add 45-degree reference line
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Normality",
        line=dict(color=COLORS["grid"], dash="dash", width=2),
        hoverinfo="skip",
    ))

    fig.update_layout(
        title="Q-Q Plot: Sample vs Theoretical Normal Distribution",
        xaxis_title="Theoretical Normal Quantiles",
        yaxis_title="Sample Quantiles (Standardized)",
        height=400,
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", size=12, color=COLORS["text"]),
        hovermode="closest",
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=COLORS["grid"],
            borderwidth=1,
        ),
    )

    return fig


def create_rolling_var_chart(
    returns: pd.Series,
    window: int = 63,
    confidence: float = 0.95,
) -> go.Figure:
    """
    Time series of rolling VaR (Cornish-Fisher) vs actual realized losses.

    Shows:
    - Rolling CF VaR line (red)
    - Rolling Gaussian VaR line (gray dashed)
    - Scatter of actual losses below VaR (breach points highlighted)

    Args:
        returns: Series of returns with datetime index
        window: Rolling window size in periods (default 63 for 3 months of trading days)
        confidence: Confidence level (default 0.95)

    Returns:
        Plotly Figure object
    """
    returns_clean = returns.dropna()

    # Calculate rolling VaR
    rolling_cf_var = []
    rolling_gaussian_var = []
    rolling_dates = []

    for i in range(window, len(returns_clean)):
        window_returns = returns_clean.iloc[i-window:i]
        var_dict = cornish_fisher_var(window_returns, confidence)
        rolling_cf_var.append(var_dict["cf_var"])
        rolling_gaussian_var.append(var_dict["gaussian_var"])
        rolling_dates.append(returns_clean.index[i])

    # Create figure
    fig = go.Figure()

    # Add CF VaR line
    fig.add_trace(go.Scatter(
        x=rolling_dates,
        y=rolling_cf_var,
        name="Cornish-Fisher VaR",
        mode="lines",
        line=dict(color=COLORS["negative"], width=2),
        hovertemplate="<b>CF VaR</b><br>Date: %{x|%Y-%m-%d}<br>VaR: %{y:.4f}<extra></extra>",
    ))

    # Add Gaussian VaR line
    fig.add_trace(go.Scatter(
        x=rolling_dates,
        y=rolling_gaussian_var,
        name="Gaussian VaR",
        mode="lines",
        line=dict(color=COLORS["grid"], dash="dash", width=2),
        hovertemplate="<b>Gaussian VaR</b><br>Date: %{x|%Y-%m-%d}<br>VaR: %{y:.4f}<extra></extra>",
    ))

    # Add actual returns and highlight breaches
    alpha = 1 - confidence
    breach_indices = []
    breach_dates = []
    breach_returns = []

    for i in range(window, len(returns_clean)):
        actual_return = returns_clean.iloc[i]
        var_threshold = rolling_cf_var[i - window]

        if actual_return < var_threshold:
            breach_indices.append(i)
            breach_dates.append(returns_clean.index[i])
            breach_returns.append(actual_return)

    if breach_dates:
        fig.add_trace(go.Scatter(
            x=breach_dates,
            y=breach_returns,
            name="VaR Breaches",
            mode="markers",
            marker=dict(
                size=8,
                color=COLORS["accent"],
                symbol="x",
            ),
            hovertemplate="<b>VaR Breach</b><br>Date: %{x|%Y-%m-%d}<br>Return: %{y:.4f}<extra></extra>",
        ))

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color=COLORS["text"],
        opacity=0.3,
        annotation_text="Zero Return",
        annotation_position="right",
    )

    fig.update_layout(
        title=f"Rolling {confidence*100:.0f}% VaR ({window}-Period Window) vs Realized Returns",
        xaxis_title="Date",
        yaxis_title="Daily Return / VaR",
        height=400,
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", size=12, color=COLORS["text"]),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=COLORS["grid"],
            borderwidth=1,
        ),
    )

    return fig
