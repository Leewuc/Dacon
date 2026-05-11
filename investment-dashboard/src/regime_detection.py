"""
Market regime detection module.

Detects market regimes (Bull/Bear/Sideways) using pure rule-based + statistical approach.
No external ML libraries (hmmlearn, sklearn) to avoid Streamlit Cloud deployment issues.

Regimes are classified using rolling statistics:
- Bull: Rolling mean return > 0 AND > 1σ above zero
- Bear: Rolling mean return < 0 AND < -1σ below zero
- Sideways: All other periods
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta


COLORS = {
    "background": "#0F172A",
    "surface": "#1E293B",
    "text": "#F8FAFC",
    "grid": "#334155",
    "primary": "#6366F1",
    "positive": "#10B981",
    "negative": "#EF4444",
    "neutral": "#F59E0B",
}

REGIME_COLORS = {
    "Bull": "rgba(16, 185, 129, 0.2)",      # green with transparency
    "Bear": "rgba(239, 68, 68, 0.2)",       # red with transparency
    "Sideways": "rgba(245, 158, 11, 0.2)",  # amber with transparency
}

REGIME_LINE_COLORS = {
    "Bull": "#10B981",      # green
    "Bear": "#EF4444",      # red
    "Sideways": "#F59E0B",  # amber
}


@dataclass
class RegimeResult:
    """Data class for regime detection results."""
    date: pd.DatetimeIndex
    regime: np.ndarray
    rolling_return: np.ndarray
    rolling_volatility: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easier manipulation."""
        return pd.DataFrame({
            "date": self.date,
            "regime": self.regime,
            "rolling_return": self.rolling_return,
            "rolling_volatility": self.rolling_volatility,
        })


def detect_regimes(
    returns: pd.Series,
    window: int = 63,
    min_duration: int = 10,
) -> pd.DataFrame:
    """
    Detect market regimes using rolling statistics.

    Algorithm:
    1. Calculate rolling mean return (default 63-day window)
    2. Calculate rolling volatility (standard deviation)
    3. Calculate rolling standard error (1σ threshold)
    4. Classify regimes:
       - Bull: rolling_mean > 0 AND rolling_mean > 1σ above zero
       - Bear: rolling_mean < 0 AND rolling_mean < -1σ below zero
       - Sideways: everything else
    5. Smooth with minimum regime duration to avoid rapid switching

    Args:
        returns: Series of daily returns
        window: Rolling window size in days (default 63)
        min_duration: Minimum consecutive days to maintain a regime (default 10)

    Returns:
        DataFrame with columns:
        - date: DatetimeIndex
        - regime: Regime label ('Bull', 'Bear', 'Sideways')
        - rolling_return: Rolling mean return
        - rolling_volatility: Rolling volatility (std dev)
    """
    # Ensure we're working with a clean series
    returns = returns.dropna()

    # Calculate rolling statistics
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    # Use rolling standard error as threshold (std / sqrt(n))
    rolling_se = rolling_std / np.sqrt(window)

    # Initialize regime array
    regimes = np.empty(len(returns), dtype=object)
    regimes[:] = "Sideways"

    # Classify based on rolling statistics
    # Bull: positive return AND statistically significant above zero
    bull_mask = (rolling_mean > 0) & (rolling_mean > rolling_se)
    regimes[bull_mask] = "Bull"

    # Bear: negative return AND statistically significant below zero
    bear_mask = (rolling_mean < 0) & (rolling_mean < -rolling_se)
    regimes[bear_mask] = "Bear"

    # Smooth regimes: enforce minimum duration
    regimes = _smooth_regimes(regimes, min_duration)

    # Create result DataFrame
    result = pd.DataFrame({
        "date": returns.index,
        "regime": regimes,
        "rolling_return": rolling_mean.values,
        "rolling_volatility": rolling_std.values,
    })

    return result


def _smooth_regimes(regimes: np.ndarray, min_duration: int = 10) -> np.ndarray:
    """
    Smooth regime changes to avoid rapid switching.

    Replace isolated regimes that last < min_duration days with the surrounding regime.

    Args:
        regimes: Array of regime labels
        min_duration: Minimum consecutive days to maintain a regime

    Returns:
        Smoothed regime array
    """
    smoothed = regimes.copy()
    i = 0

    while i < len(smoothed):
        # Find regime change point
        if i == 0 or smoothed[i] != smoothed[i - 1]:
            # Count consecutive days in this regime
            j = i
            while j < len(smoothed) and smoothed[j] == smoothed[i]:
                j += 1

            duration = j - i

            # If regime duration < min_duration, replace with surrounding regime
            if duration < min_duration:
                if i > 0:
                    # Use previous regime
                    smoothed[i:j] = smoothed[i - 1]
                elif j < len(smoothed):
                    # Use next regime
                    smoothed[i:j] = smoothed[j]

            i = j
        else:
            i += 1

    return smoothed


def analyze_regime_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
    regimes: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> Dict:
    """
    Compute regime-specific performance metrics.

    Args:
        portfolio_returns: Series of portfolio daily returns
        benchmark_returns: Series of benchmark daily returns (optional)
        regimes: DataFrame from detect_regimes()
        risk_free_rate: Annual risk-free rate for Sharpe calculation (default 2%)

    Returns:
        Dict with regime statistics:
        {
            'Bull': {...},
            'Bear': {...},
            'Sideways': {...},
            'overall': {...}
        }
    """
    # Align all series by date
    df = pd.DataFrame({
        "portfolio_returns": portfolio_returns,
        "regime": regimes.set_index(regimes["date"])["regime"],
    })

    if benchmark_returns is not None:
        df["benchmark_returns"] = benchmark_returns

    df = df.dropna()

    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    stats = {}

    # Calculate stats for each regime
    for regime in ["Bull", "Bear", "Sideways"]:
        regime_data = df[df["regime"] == regime]

        if len(regime_data) == 0:
            stats[regime] = {
                "count_days": 0,
                "annualized_return": 0,
                "annualized_volatility": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "avg_duration_days": 0,
                "alpha": 0 if benchmark_returns is not None else None,
            }
            continue

        port_rets = regime_data["portfolio_returns"].values

        # Basic stats
        count_days = len(regime_data)
        annualized_return = (1 + port_rets.mean()) ** 252 - 1
        annualized_volatility = port_rets.std() * np.sqrt(252)

        # Sharpe ratio
        excess_return = port_rets.mean() - daily_rf
        sharpe_ratio = (excess_return / port_rets.std() * np.sqrt(252)) if port_rets.std() > 0 else 0

        # Max drawdown
        cumulative = (1 + port_rets).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # Win rate (% days with positive returns)
        win_rate = (port_rets > 0).sum() / len(port_rets) if len(port_rets) > 0 else 0

        # Average duration of regime episodes
        avg_duration = _calculate_average_regime_duration(
            regimes[regimes["regime"] == regime]
        )

        # Alpha vs benchmark
        alpha = 0
        if benchmark_returns is not None:
            bench_rets = regime_data["benchmark_returns"].values
            alpha = annualized_return - ((1 + bench_rets.mean()) ** 252 - 1)

        stats[regime] = {
            "count_days": count_days,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_duration_days": avg_duration,
            "alpha": alpha if benchmark_returns is not None else None,
        }

    # Calculate overall stats
    port_rets = df["portfolio_returns"].values
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_return = port_rets.mean() - daily_rf
    sharpe_ratio = (excess_return / port_rets.std() * np.sqrt(252)) if port_rets.std() > 0 else 0

    cumulative = (1 + port_rets).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

    stats["overall"] = {
        "count_days": len(df),
        "annualized_return": (1 + port_rets.mean()) ** 252 - 1,
        "annualized_volatility": port_rets.std() * np.sqrt(252),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": (port_rets > 0).sum() / len(port_rets),
    }

    return stats


def _calculate_average_regime_duration(regime_df: pd.DataFrame) -> float:
    """
    Calculate average duration of regime episodes.

    Args:
        regime_df: Filtered DataFrame for a single regime

    Returns:
        Average duration in days
    """
    if len(regime_df) == 0:
        return 0

    # Find consecutive date groups
    dates = pd.to_datetime(regime_df["date"]).values
    if len(dates) == 0:
        return 0

    # Calculate gaps between consecutive dates
    date_diffs = np.diff(dates).astype('timedelta64[D]').astype(int)

    # Identify regime breaks (gaps > 1 day)
    breaks = np.where(date_diffs > 1)[0]

    # Split into episodes
    episodes = []
    start_idx = 0

    for break_idx in breaks:
        episodes.append(break_idx - start_idx + 1)
        start_idx = break_idx + 1

    # Add final episode
    episodes.append(len(dates) - start_idx)

    return np.mean(episodes) if episodes else len(dates)


def get_current_regime(regimes: pd.DataFrame) -> Dict:
    """
    Get the current (latest) detected regime.

    Args:
        regimes: DataFrame from detect_regimes()

    Returns:
        Dict with keys:
        - regime: Current regime label ('Bull', 'Bear', 'Sideways')
        - date: Date of regime detection
        - rolling_return: Current rolling return
        - rolling_volatility: Current rolling volatility
    """
    latest = regimes.iloc[-1]

    return {
        "regime": latest["regime"],
        "date": latest["date"],
        "rolling_return": latest["rolling_return"],
        "rolling_volatility": latest["rolling_volatility"],
    }


def get_regime_summary(
    regimes: pd.DataFrame,
    regime_stats: Dict,
) -> Dict:
    """
    Get summary KPI metrics for the current regime.

    Args:
        regimes: DataFrame from detect_regimes()
        regime_stats: Dict from analyze_regime_performance()

    Returns:
        Dict with KPI values for current regime
    """
    current = get_current_regime(regimes)
    regime_name = current["regime"]
    stats = regime_stats.get(regime_name, {})

    return {
        "regime": regime_name,
        "annualized_return": stats.get("annualized_return", 0),
        "sharpe_ratio": stats.get("sharpe_ratio", 0),
        "max_drawdown": stats.get("max_drawdown", 0),
        "win_rate": stats.get("win_rate", 0),
        "volatility": current["rolling_volatility"],
    }


def create_regime_timeline(
    portfolio_returns: pd.Series,
    regimes: pd.DataFrame,
) -> go.Figure:
    """
    Create cumulative returns chart with regime background bands.

    Args:
        portfolio_returns: Series of daily returns
        regimes: DataFrame from detect_regimes()

    Returns:
        Plotly Figure with regime timeline
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Align indices
    returns_df = pd.DataFrame({
        "date": portfolio_returns.index,
        "cumulative_return": cumulative_returns.values,
    })

    regimes = regimes.copy()
    regimes["date"] = pd.to_datetime(regimes["date"])

    merged = returns_df.merge(
        regimes[["date", "regime"]],
        on="date",
        how="left"
    )
    merged["regime"] = merged["regime"].fillna("Sideways")

    fig = go.Figure()

    # Add regime background bands
    for i in range(len(merged) - 1):
        current_regime = merged.iloc[i]["regime"]

        # Check if regime is about to change
        if i == len(merged) - 1 or merged.iloc[i + 1]["regime"] != current_regime:
            # Find start of this regime
            j = i
            while j > 0 and merged.iloc[j - 1]["regime"] == current_regime:
                j -= 1

            start_date = merged.iloc[j]["date"]
            end_date = merged.iloc[i]["date"]

            fig.add_vrect(
                x0=start_date,
                x1=end_date,
                fillcolor=REGIME_COLORS[current_regime],
                layer="below",
                line_width=0,
            )

    # Add cumulative return line
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["cumulative_return"] * 100,
            mode="lines",
            name="Portfolio Return",
            line=dict(color=COLORS["primary"], width=2),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Return: %{y:.2f}%<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.1)",
        )
    )

    fig.update_layout(
        title="Portfolio Returns by Market Regime",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        xaxis=dict(
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_regime_performance_bars(regime_stats: Dict) -> go.Figure:
    """
    Create grouped bar chart of regime performance metrics.

    Args:
        regime_stats: Dict from analyze_regime_performance()

    Returns:
        Plotly Figure with performance bars
    """
    regimes = ["Bull", "Bear", "Sideways"]

    returns = []
    sharpe_ratios = []
    win_rates = []

    for regime in regimes:
        stats = regime_stats.get(regime, {})
        returns.append(stats.get("annualized_return", 0) * 100)
        sharpe_ratios.append(stats.get("sharpe_ratio", 0))
        win_rates.append(stats.get("win_rate", 0) * 100)

    fig = go.Figure()

    # Add return bars
    fig.add_trace(
        go.Bar(
            x=regimes,
            y=returns,
            name="Ann. Return (%)",
            marker_color=[REGIME_LINE_COLORS[r] for r in regimes],
            hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    # Add Sharpe bars (secondary axis)
    fig.add_trace(
        go.Bar(
            x=regimes,
            y=sharpe_ratios,
            name="Sharpe Ratio",
            marker_color=[REGIME_LINE_COLORS[r] for r in regimes],
            marker_pattern_shape="/",
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Regime Performance Comparison",
        xaxis_title="Market Regime",
        yaxis_title="Annualized Return (%)",
        yaxis2=dict(
            title="Sharpe Ratio",
            overlaying="y",
            side="right",
            gridcolor=COLORS["grid"],
        ),
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
        height=400,
        margin=dict(l=50, r=80, t=80, b=50),
        barmode="group",
    )

    return fig


def create_regime_transition_matrix(regimes: pd.DataFrame) -> go.Figure:
    """
    Create heatmap of regime transition probabilities.

    Models transition matrix like a Markov chain.

    Args:
        regimes: DataFrame from detect_regimes()

    Returns:
        Plotly Figure with transition heatmap
    """
    regime_labels = ["Bull", "Bear", "Sideways"]
    n_regimes = len(regime_labels)

    # Build transition matrix
    transition_matrix = np.zeros((n_regimes, n_regimes))

    regime_values = regimes["regime"].values

    for i in range(len(regime_values) - 1):
        current_regime = regime_values[i]
        next_regime = regime_values[i + 1]

        if current_regime == next_regime:
            continue  # Skip same-regime transitions

        current_idx = regime_labels.index(current_regime)
        next_idx = regime_labels.index(next_regime)

        transition_matrix[current_idx, next_idx] += 1

    # Normalize to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_matrix,
        row_sums,
        where=row_sums != 0,
        out=np.zeros_like(transition_matrix),
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=transition_probs * 100,
            x=regime_labels,
            y=regime_labels,
            colorscale=[
                [0, "#0F172A"],
                [1, "#10B981"],
            ],
            hovertemplate="From <b>%{y}</b> to <b>%{x}</b><br>Probability: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Probability (%)"),
        )
    )

    fig.update_layout(
        title="Regime Transition Matrix",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        height=350,
        margin=dict(l=80, r=50, t=80, b=80),
    )

    return fig


def create_regime_duration_chart(regimes: pd.DataFrame) -> go.Figure:
    """
    Create horizontal bar chart showing duration of regime episodes.

    Args:
        regimes: DataFrame from detect_regimes()

    Returns:
        Plotly Figure with duration bars
    """
    # Identify regime episodes
    episodes = []
    current_regime = None
    start_date = None

    for idx, row in regimes.iterrows():
        regime = row["regime"]
        date = row["date"]

        if regime != current_regime:
            if current_regime is not None:
                episodes.append({
                    "regime": current_regime,
                    "start": start_date,
                    "end": regimes.iloc[idx - 1]["date"],
                    "duration": (regimes.iloc[idx - 1]["date"] - start_date).days,
                })
            current_regime = regime
            start_date = date

    # Add final episode
    if current_regime is not None:
        episodes.append({
            "regime": current_regime,
            "start": start_date,
            "end": regimes.iloc[-1]["date"],
            "duration": (regimes.iloc[-1]["date"] - start_date).days,
        })

    if not episodes:
        # Return empty figure if no episodes
        fig = go.Figure()
        fig.update_layout(
            title="Regime Duration Chart",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
        )
        return fig

    episodes_df = pd.DataFrame(episodes)
    episodes_df = episodes_df.sort_values("start")
    episodes_df["label"] = (
        episodes_df["regime"]
        + " ("
        + episodes_df["start"].dt.strftime("%Y-%m-%d")
        + " - "
        + episodes_df["end"].dt.strftime("%Y-%m-%d")
        + ")"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=episodes_df["label"],
            x=episodes_df["duration"],
            orientation="h",
            marker_color=[REGIME_LINE_COLORS[r] for r in episodes_df["regime"]],
            hovertemplate="<b>%{y}</b><br>Duration: %{x} days<extra></extra>",
        )
    )

    fig.update_layout(
        title="Market Regime Duration History",
        xaxis_title="Duration (days)",
        yaxis_title="Regime Episode",
        hovermode="y unified",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
        yaxis=dict(gridcolor=COLORS["grid"]),
        height=400 + max(50, len(episodes_df) * 20),
        margin=dict(l=250, r=50, t=80, b=50),
    )

    return fig


if __name__ == "__main__":
    # Basic smoke test
    print("regime_detection module loaded successfully")

    # Test with sample data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)

    regimes = detect_regimes(returns)
    print(f"Detected {len(regimes)} regime observations")
    print(f"Current regime: {get_current_regime(regimes)['regime']}")
