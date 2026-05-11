"""
Backtesting Engine: 포트폴리오의 과거 성과를 시뮬레이션하는 모듈

핵심 기능:
    1. Single Backtest: 특정 기간에 대한 포트폴리오 성과 분석
    2. Rolling Backtests: 다양한 시작 시점에서의 성과 비교
    3. Visualization: 누적 수익률, 월별 리턴, 성과 히트맵 등

Dependencies: numpy, pandas, plotly (NO scipy)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from visualizations import COLORS, LAYOUT_DEFAULTS


@dataclass
class BacktestResult:
    """백테스트 결과를 담는 데이터클래스"""
    start_date: str
    end_date: str
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    best_month: float
    worst_month: float
    win_rate: float
    cumulative_curve: pd.Series
    monthly_returns: pd.Series
    benchmark_total_return: float
    alpha: float


# =============================================================================
# Core Backtesting Logic
# =============================================================================

def run_backtest(
    prices_df: pd.DataFrame,
    weights: Dict[str, float],
    benchmark_prices: Optional[pd.Series] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    risk_free_rate: float = 0.04,
) -> BacktestResult:
    """
    주어진 가중치와 기간으로 포트폴리오 백테스트 실행

    Parameters:
        prices_df: DataFrame with columns=tickers, index=dates (날짜순 정렬)
        weights: Dict[ticker, weight] 포트폴리오 비중
        benchmark_prices: Series with benchmark daily prices
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD), None이면 마지막 날짜까지
        risk_free_rate: 무위험 이자율 (연환산)

    Returns:
        BacktestResult 객체
    """
    # 0. MultiIndex columns 처리 (OHLCV → Close만 추출)
    if isinstance(prices_df.columns, pd.MultiIndex):
        level0_vals = prices_df.columns.get_level_values(0).unique().tolist()
        if "Close" in level0_vals:
            prices_df = prices_df["Close"]
        elif "Close" in prices_df.columns.get_level_values(1).unique().tolist():
            prices_df = prices_df.xs("Close", axis=1, level=1)
        else:
            prices_df = prices_df.iloc[:, :len(weights)]
        if isinstance(prices_df.columns, pd.MultiIndex):
            prices_df.columns = prices_df.columns.get_level_values(-1)

    # 1. 날짜 인덱스를 datetime으로 변환
    prices_df = prices_df.copy()
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)

    # 2. 날짜 범위 설정
    if start_date is None:
        start_date = prices_df.index[0].strftime("%Y-%m-%d")
    if end_date is None:
        end_date = prices_df.index[-1].strftime("%Y-%m-%d")

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 3. 날짜 범위로 필터링
    mask = (prices_df.index >= start_dt) & (prices_df.index <= end_dt)
    prices_subset = prices_df[mask].copy()

    if len(prices_subset) < 2:
        raise ValueError(
            f"Insufficient data for backtest period {start_date} to {end_date}"
        )

    # 4. 사용 가능한 티커만 필터링 (missing tickers gracefully skip)
    available_tickers = [t for t in weights.keys() if t in prices_subset.columns]

    if not available_tickers:
        raise ValueError(f"No available tickers from {list(weights.keys())} in prices_df")

    # 5. 비중 정규화 (사용 가능한 티커만)
    total_weight = sum(weights[t] for t in available_tickers)
    normalized_weights = {t: weights[t] / total_weight for t in available_tickers}

    # 6. 일일 수익률 계산
    daily_returns = prices_subset[available_tickers].pct_change().dropna()

    # 7. 포트폴리오 일일 수익률 계산
    portfolio_weights = np.array([normalized_weights[t] for t in available_tickers])
    portfolio_daily_returns = (daily_returns[available_tickers].values @ portfolio_weights)

    portfolio_returns_series = pd.Series(
        portfolio_daily_returns,
        index=daily_returns.index,
        name="portfolio_return",
    )

    # 8. 누적 수익률 계산
    cumulative_returns = (1 + portfolio_returns_series).cumprod() - 1
    cumulative_curve = cumulative_returns

    # 9. 기본 메트릭스 계산
    total_return = cumulative_returns.iloc[-1]
    num_years = len(cumulative_returns) / 252  # 252 trading days per year
    annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0

    # 10. 최대 드로다운 계산
    cumulative_wealth = 1 + cumulative_returns
    running_max = cumulative_wealth.expanding().max()
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = drawdown.min()

    # 11. 변동성 (연환산)
    daily_volatility = portfolio_returns_series.std()
    volatility = daily_volatility * np.sqrt(252)

    # 12. Sharpe Ratio (daily risk-free rate)
    daily_risk_free = risk_free_rate / 252
    excess_returns = portfolio_returns_series - daily_risk_free
    sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0

    # 13. 월별 수익률
    monthly_returns = cumulative_returns.resample("ME").last()
    # 첫 달은 따로 처리
    first_month_end = pd.Timestamp(daily_returns.index[0]).to_period("M").to_timestamp("D") + pd.offsets.MonthEnd(0)
    if daily_returns.index[0] < first_month_end and first_month_end <= daily_returns.index[-1]:
        first_month_returns = (1 + portfolio_returns_series[daily_returns.index[0]:first_month_end]).prod() - 1
        monthly_series_list = [first_month_returns]
        monthly_series_list.extend(monthly_returns.diff())
        monthly_returns = pd.Series(monthly_series_list, index=[daily_returns.index[0]] + monthly_returns.index.tolist())
    else:
        # 간단한 방식: monthly로 resample한 누적 수익률의 변화
        monthly_data = cumulative_returns.resample("ME").last()
        monthly_ret = monthly_data.diff()
        monthly_ret.iloc[0] = monthly_data.iloc[0]
        monthly_returns = monthly_ret

    best_month = monthly_returns[monthly_returns > 0].max() if len(monthly_returns[monthly_returns > 0]) > 0 else 0
    worst_month = monthly_returns[monthly_returns < 0].min() if len(monthly_returns[monthly_returns < 0]) > 0 else 0

    win_rate = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0

    # 14. 벤치마크 비교 (있으면)
    benchmark_total_return = 0.0
    alpha = 0.0

    if benchmark_prices is not None:
        benchmark_prices_subset = benchmark_prices.loc[start_dt:end_dt]
        if len(benchmark_prices_subset) > 1:
            benchmark_ret = (benchmark_prices_subset.iloc[-1] / benchmark_prices_subset.iloc[0]) - 1
            benchmark_total_return = benchmark_ret
            alpha = total_return - benchmark_total_return
        else:
            benchmark_total_return = 0.0
            alpha = 0.0

    return BacktestResult(
        start_date=start_date,
        end_date=end_date,
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        volatility=volatility,
        best_month=best_month,
        worst_month=worst_month,
        win_rate=win_rate,
        cumulative_curve=cumulative_curve,
        monthly_returns=monthly_returns,
        benchmark_total_return=benchmark_total_return,
        alpha=alpha,
    )


def run_rolling_backtests(
    prices_df: pd.DataFrame,
    weights: Dict[str, float],
    benchmark_prices: Optional[pd.Series] = None,
    window_months: int = 12,
    step_months: int = 3,
) -> List[BacktestResult]:
    """
    여러 시작 시점에서 rolling backtest 실행 (각 3개월 간격)

    Parameters:
        prices_df: DataFrame with columns=tickers, index=dates
        weights: Dict[ticker, weight]
        benchmark_prices: Series with benchmark prices
        window_months: 각 백테스트 윈도우 길이 (월)
        step_months: 시작 시점 간격 (월)

    Returns:
        List[BacktestResult]
    """
    prices_df = prices_df.copy()
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)

    results = []
    start_idx = 0
    min_data_points = 20

    while start_idx < len(prices_df) - min_data_points:
        start_date = prices_df.index[start_idx]
        end_date = start_date + pd.DateOffset(months=window_months)

        # 종료 날짜가 데이터 범위를 초과하면 데이터 끝까지 사용
        if end_date > prices_df.index[-1]:
            end_date = prices_df.index[-1]

        try:
            result = run_backtest(
                prices_df,
                weights,
                benchmark_prices,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
            results.append(result)
        except ValueError:
            pass

        start_idx += int(len(prices_df) * step_months / 12 / len(prices_df))
        start_idx = max(start_idx + 1, start_idx)  # at least increment by 1

    # 수동으로 date offset 계산하기 위해 다시 작성
    results = []
    current_date = prices_df.index[0]

    while current_date < prices_df.index[-1]:
        end_date = current_date + pd.DateOffset(months=window_months)

        # 데이터 범위 초과 확인
        if end_date > prices_df.index[-1]:
            end_date = prices_df.index[-1]

        start_str = current_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            result = run_backtest(
                prices_df,
                weights,
                benchmark_prices,
                start_date=start_str,
                end_date=end_str,
            )
            results.append(result)
        except ValueError:
            pass

        current_date = current_date + pd.DateOffset(months=step_months)

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def create_backtest_cumulative_chart(
    result: BacktestResult,
    benchmark_curve: Optional[pd.Series] = None,
) -> go.Figure:
    """
    누적 수익률 차트 + 드로다운 영역

    Parameters:
        result: BacktestResult 객체
        benchmark_curve: 벤치마크 누적 수익률 Series (선택사항)

    Returns:
        plotly Figure
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3],
    )

    # 포트폴리오 누적 수익률
    cumulative_pct = (result.cumulative_curve * 100).values
    dates = result.cumulative_curve.index

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cumulative_pct,
            mode="lines",
            name="Portfolio",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.1)",
        ),
        row=1,
        col=1,
    )

    # 벤치마크 (있으면)
    if benchmark_curve is not None:
        benchmark_pct = (benchmark_curve * 100).values
        fig.add_trace(
            go.Scatter(
                x=benchmark_curve.index,
                y=benchmark_pct,
                mode="lines",
                name="Benchmark",
                line=dict(color=COLORS["neutral"], width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # 드로다운 계산
    cumulative_wealth = 1 + result.cumulative_curve
    running_max = cumulative_wealth.expanding().max()
    drawdown = ((cumulative_wealth - running_max) / running_max * 100).values

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color=COLORS["negative"], width=1),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.3)",
        ),
        row=2,
        col=1,
    )

    # 레이아웃
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    fig.update_layout(
        title=f"Backtest Cumulative Returns ({result.start_date} to {result.end_date})",
        **LAYOUT_DEFAULTS,
        hovermode="x unified",
        height=600,
    )

    return fig


def create_rolling_performance_chart(
    results: List[BacktestResult],
) -> go.Figure:
    """
    Rolling backtest 결과 시각화 - 시작 날짜별 총 수익률

    Parameters:
        results: List[BacktestResult]

    Returns:
        plotly Figure
    """
    if not results:
        raise ValueError("No backtest results provided")

    start_dates = [r.start_date for r in results]
    total_returns = [r.total_return * 100 for r in results]
    annualized_returns = [r.annualized_return * 100 for r in results]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=start_dates,
            y=total_returns,
            name="Total Return",
            marker=dict(
                color=total_returns,
                colorscale="RdYlGn",
                showscale=False,
                line=dict(color=COLORS["primary"], width=1),
            ),
            text=[f"{r:.1f}%" for r in total_returns],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Total Return: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Rolling Backtest Performance by Start Date",
        xaxis_title="Start Date",
        yaxis_title="Total Return (%)",
        **LAYOUT_DEFAULTS,
        hovermode="x",
        height=500,
    )

    return fig


def create_backtest_monthly_heatmap(
    result: BacktestResult,
) -> go.Figure:
    """
    월별 수익률 히트맵 (행=연도, 열=월)

    Parameters:
        result: BacktestResult 객체

    Returns:
        plotly Figure
    """
    # 월별 수익률을 연도-월 구조로 변환
    if len(result.monthly_returns) == 0:
        raise ValueError("No monthly returns available for heatmap")

    monthly_pct = (result.monthly_returns * 100).values

    # 월별 인덱스에서 연도와 월 추출
    dates = result.monthly_returns.index
    years = [d.year for d in dates]
    months_num = [d.month for d in dates]
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    # 데이터 재구성 (행=연도, 열=월)
    unique_years = sorted(set(years))
    heatmap_data = []
    for year in unique_years:
        year_data = [None] * 12
        for i, y in enumerate(years):
            if y == year:
                year_data[months_num[i] - 1] = monthly_pct[i]
        heatmap_data.append(year_data)

    # 색상 스케일 결정
    vmin = min([v for row in heatmap_data for v in row if v is not None])
    vmax = max([v for row in heatmap_data for v in row if v is not None])

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=month_names,
            y=[str(y) for y in unique_years],
            colorscale="RdYlGn",
            zmid=0,
            zmin=vmin,
            zmax=vmax,
            text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Return (%)"),
            hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Monthly Returns Heatmap ({result.start_date} to {result.end_date})",
        xaxis_title="Month",
        yaxis_title="Year",
        **LAYOUT_DEFAULTS,
        height=400,
    )

    return fig


def create_performance_metrics_table(result: BacktestResult) -> go.Figure:
    """
    백테스트 성과 지표를 테이블로 표시

    Parameters:
        result: BacktestResult 객체

    Returns:
        plotly Table Figure
    """
    metrics = [
        ["Metric", "Value"],
        ["Total Return", f"{result.total_return * 100:.2f}%"],
        ["Annualized Return", f"{result.annualized_return * 100:.2f}%"],
        ["Volatility (Annual)", f"{result.volatility * 100:.2f}%"],
        ["Sharpe Ratio", f"{result.sharpe_ratio:.2f}"],
        ["Max Drawdown", f"{result.max_drawdown * 100:.2f}%"],
        ["Best Month", f"{result.best_month * 100:.2f}%"],
        ["Worst Month", f"{result.worst_month * 100:.2f}%"],
        ["Win Rate", f"{result.win_rate * 100:.1f}%"],
        ["Benchmark Return", f"{result.benchmark_total_return * 100:.2f}%"],
        ["Alpha", f"{result.alpha * 100:.2f}%"],
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=metrics[0],
                    fill_color=COLORS["primary"],
                    align="left",
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=list(zip(*metrics[1:])),
                    fill_color=COLORS["surface"],
                    align="left",
                    font=dict(color=COLORS["text"], size=11),
                    height=25,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Performance Metrics ({result.start_date} to {result.end_date})",
        **LAYOUT_DEFAULTS,
        height=400,
    )

    return fig
