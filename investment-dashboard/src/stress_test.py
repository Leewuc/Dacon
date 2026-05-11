"""
Stress Test Engine: 역사적 위기 시나리오 시뮬레이션
포트폴리오가 과거 금융 위기에서 어떤 손실을 겪었을지 추정
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import plotly.graph_objects as go


@dataclass
class ScenarioSpec:
    """위기 시나리오 정의"""
    name: str
    description: str
    period: str           # e.g., "2020-02 ~ 2020-03"
    market_decline: float # e.g., -0.34 (S&P500 기준)
    duration_days: int    # trading days
    recovery_days: int    # days to recover
    sector_impacts: Dict[str, float]  # sector -> impact multiplier
    # e.g., {"Technology": 0.8, "Healthcare": 0.6, "Energy": 1.5}


@dataclass
class StressResult:
    """스트레스 테스트 결과"""
    scenario: ScenarioSpec
    portfolio_loss: float         # estimated portfolio loss
    worst_day: float              # worst single day loss
    recovery_estimate_days: int   # estimated recovery time
    sector_losses: Dict[str, float]  # per-sector loss
    daily_path: np.ndarray       # simulated daily cumulative path
    var_95: float                # 95% VaR during crisis


# ========== Built-in Scenarios ==========

SCENARIOS = {
    "COVID-19 Crash (2020)": ScenarioSpec(
        name="COVID-19 Crash",
        description="2020년 2~3월 팬데믹 공포로 인한 급락. S&P500 -34% (23거래일 만에). 가장 빠른 베어마켓 진입.",
        period="2020-02-19 ~ 2020-03-23",
        market_decline=-0.34,
        duration_days=23,
        recovery_days=148,
        sector_impacts={
            "Technology": 0.85, "반도체": 0.85,
            "Healthcare": 0.65, "바이오": 0.65, "제약": 0.55,
            "Energy": 1.60, "에너지": 1.60,
            "Financial": 1.20, "금융": 1.20, "보험": 1.10,
            "Consumer": 1.30, "유통": 1.30, "식품": 0.70,
            "Industrial": 1.10, "건설": 1.10, "중공업": 1.15,
            "IT/플랫폼": 0.80, "게임": 0.75,
            "통신": 0.70, "유틸리티": 0.65,
            "화학": 1.10, "철강": 1.20,
            "자동차": 1.15, "자동차부품": 1.15,
            "2차전지": 0.90,
            "핀테크": 0.95, "화장품": 1.00,
            "방산": 0.80, "조선": 1.10, "항공": 1.80, "해운": 1.40,
            "증권": 1.25, "여행/레저": 1.70,
            "지주": 1.00, "건설/지주": 1.05,
            "전자부품": 0.90, "비철금속": 1.10,
        },
    ),
    "Global Financial Crisis (2008)": ScenarioSpec(
        name="2008 금융위기",
        description="서브프라임 모기지 사태로 촉발된 글로벌 금융위기. S&P500 -57% (17개월). 리먼브라더스 파산.",
        period="2007-10 ~ 2009-03",
        market_decline=-0.57,
        duration_days=352,
        recovery_days=1020,
        sector_impacts={
            "Technology": 0.85, "반도체": 0.90,
            "Financial": 1.80, "금융": 1.80, "보험": 1.50, "증권": 1.70,
            "Energy": 1.40, "에너지": 1.40,
            "Healthcare": 0.60, "바이오": 0.55, "제약": 0.50,
            "Consumer": 1.00, "유통": 1.10,
            "IT/플랫폼": 0.90, "게임": 0.85,
            "통신": 0.70, "유틸리티": 0.65,
            "건설": 1.50, "철강": 1.40, "화학": 1.20,
            "자동차": 1.30, "2차전지": 0.90,
            "핀테크": 1.60, "방산": 0.80, "조선": 1.30,
            "지주": 1.10,
        },
    ),
    "2022 Rate Hike Shock": ScenarioSpec(
        name="2022 금리인상 충격",
        description="Fed의 급격한 금리인상(0%→5.25%)에 따른 성장주 폭락. NASDAQ -33%. 채권도 동반 하락.",
        period="2022-01 ~ 2022-10",
        market_decline=-0.25,
        duration_days=194,
        recovery_days=310,
        sector_impacts={
            "Technology": 1.40, "반도체": 1.30,
            "Financial": 0.70, "금융": 0.70, "보험": 0.60,
            "Energy": 0.30, "에너지": 0.30,  # Energy actually went UP
            "Healthcare": 0.80, "바이오": 1.00, "제약": 0.70,
            "Consumer": 1.10, "유통": 1.20,
            "IT/플랫폼": 1.50, "게임": 1.30,
            "통신": 0.80, "유틸리티": 0.60,
            "2차전지": 1.20, "화학": 1.00, "철강": 0.90,
            "자동차": 0.85, "건설": 0.90,
            "핀테크": 1.60, "화장품": 1.10,
            "방산": 0.50, "조선": 0.70,
            "지주": 0.90, "증권": 1.10,
        },
    ),
    "Dot-com Bubble (2000)": ScenarioSpec(
        name="닷컴 버블 붕괴",
        description="인터넷 버블 붕괴. NASDAQ -78% (2.5년). 기술주 중심의 극심한 하락.",
        period="2000-03 ~ 2002-10",
        market_decline=-0.49,
        duration_days=650,
        recovery_days=1730,
        sector_impacts={
            "Technology": 1.80, "반도체": 1.70,
            "IT/플랫폼": 2.00, "게임": 1.60,
            "Financial": 0.80, "금융": 0.80,
            "Energy": 0.50, "에너지": 0.50,
            "Healthcare": 0.60, "바이오": 0.70,
            "Consumer": 0.70, "통신": 1.40,
            "유틸리티": 0.50, "유통": 0.80,
            "건설": 0.70, "철강": 0.80,
            "자동차": 0.75, "2차전지": 1.30,
            "핀테크": 1.80, "화장품": 0.70,
            "방산": 0.60, "조선": 0.80,
        },
    ),
}


def _get_sector_impact(sector: str, scenario: ScenarioSpec) -> float:
    """
    섹터명으로부터 영향도 배수를 찾음
    Unknown sectors는 1.0 (평균) 반환
    """
    impact = scenario.sector_impacts.get(sector)
    if impact is not None:
        return impact
    return 1.0


def run_stress_test(
    weights: Dict[str, float],
    sector_map: Dict[str, str],
    scenario: ScenarioSpec,
    portfolio_vol: float = 0.20,
) -> StressResult:
    """
    주어진 시나리오에 대해 포트폴리오 스트레스 테스트 실행

    Method:
    1. 각 종목의 섹터에 해당하는 sector_impact를 찾음
    2. portfolio_loss = market_decline * weighted_average(sector_impacts)
    3. daily_path: 시나리오 기간 동안의 시뮬레이션 경로 생성
       - GBM(Geometric Brownian Motion)으로 하락 경로 시뮬레이션
       - 최종 값이 portfolio_loss에 수렴하도록 drift 조정
    4. worst_day: 일간 최대 손실
    5. recovery_estimate: 시나리오의 recovery_days에 비례

    Parameters:
    -----------
    weights : Dict[str, float]
        종목 비중 {종목명: 비중}
    sector_map : Dict[str, str]
        종목->섹터 맵 {종목명: 섹터명}
    scenario : ScenarioSpec
        시나리오 정의
    portfolio_vol : float
        연화 포트폴리오 변동성 (기본값 0.20 = 20%)

    Returns:
    --------
    StressResult : 스트레스 테스트 결과
    """
    np.random.seed(42)  # Reproducibility

    # 1. 각 종목의 섹터 영향도 계산
    sector_impacts_list = []
    for ticker, weight in weights.items():
        sector = sector_map.get(ticker, "Unknown")
        impact = _get_sector_impact(sector, scenario)
        sector_impacts_list.append((ticker, weight, sector, impact))

    # 2. 포트폴리오 평균 영향도와 예상 손실 계산
    total_weight = sum(w for _, w, _, _ in sector_impacts_list)
    if total_weight <= 0:
        total_weight = 1.0

    weighted_impact = sum(w * impact for _, w, _, impact in sector_impacts_list) / total_weight

    # market_decline에 weighted_impact를 곱함
    portfolio_loss = scenario.market_decline * weighted_impact

    # 3. 섹터별 손실 계산
    sector_losses = {}
    for sector in set(s for _, _, s, _ in sector_impacts_list):
        impact = _get_sector_impact(sector, scenario)
        sector_loss = scenario.market_decline * impact
        sector_losses[sector] = sector_loss

    # 4. Daily path 생성 (GBM으로 시뮬레이션)
    # 최종값이 portfolio_loss에 수렴하도록 하는 경로를 생성
    n_days = scenario.duration_days
    daily_vol = portfolio_vol / np.sqrt(252)  # Daily volatility

    # Drift는 portfolio_loss가 n_days 후 도달하도록 설정
    # S(t) = S0 * exp((mu - 0.5*sigma^2)*t + sigma*sqrt(t)*Z)
    # 최종: S(T) = 1 + portfolio_loss = exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z_avg)
    # 평균적으로 Z=0이면: log(1 + portfolio_loss) = (mu - 0.5*sigma^2)*T
    target_return = portfolio_loss

    # GBM 시뮬레이션
    dt = 1 / 252  # 1 trading day
    drift = np.log(1 + target_return) / scenario.duration_days - 0.5 * daily_vol ** 2

    path = np.zeros(n_days + 1)
    path[0] = 1.0  # Start at 1

    # Z-score 샘플 (변동성이 있으면서도 최종값이 target에 수렴)
    z_scores = np.random.standard_normal(n_days)

    for i in range(n_days):
        dW = z_scores[i]
        path[i + 1] = path[i] * np.exp(drift * dt + daily_vol * dW)

    # 최종값을 목표값으로 조정 (시뮬레이션 오차 보정)
    path[-1] = 1 + target_return

    # 5. Worst day 계산
    daily_returns = np.diff(path) / path[:-1]
    worst_day = np.min(daily_returns)

    # 6. 95% VaR 계산
    var_95 = np.percentile(daily_returns, 5)

    # 7. Recovery time estimate
    recovery_estimate_days = scenario.recovery_days

    result = StressResult(
        scenario=scenario,
        portfolio_loss=portfolio_loss,
        worst_day=worst_day,
        recovery_estimate_days=recovery_estimate_days,
        sector_losses=sector_losses,
        daily_path=path,
        var_95=var_95,
    )

    return result


def run_all_stress_tests(
    weights: Dict[str, float],
    sector_map: Dict[str, str],
    portfolio_vol: float = 0.20,
) -> Dict[str, StressResult]:
    """모든 내장 시나리오에 대해 스트레스 테스트 실행"""
    results = {}
    for name, scenario in SCENARIOS.items():
        results[name] = run_stress_test(weights, sector_map, scenario, portfolio_vol)
    return results


# ========== Plotly Visualizations ==========

def create_stress_comparison_bar(results: Dict[str, StressResult]) -> go.Figure:
    """
    시나리오별 예상 손실 비교 바 차트 (가로)
    - 빨간 바: 포트폴리오 예상 손실
    - 회색 바: 시장 평균 손실
    - 다크 테마
    """
    scenarios = []
    portfolio_losses = []
    market_losses = []

    for scenario_name, result in results.items():
        scenarios.append(result.scenario.name)
        portfolio_losses.append(result.portfolio_loss * 100)
        market_losses.append(result.scenario.market_decline * 100)

    fig = go.Figure()

    # 시장 손실
    fig.add_trace(go.Bar(
        y=scenarios,
        x=market_losses,
        name="Market Loss",
        orientation='h',
        marker=dict(color='rgba(107, 114, 128, 0.5)'),
        text=[f"{x:.1f}%" for x in market_losses],
        textposition='outside',
        hovertemplate="<b>Market Loss</b><br>%{x:.2f}%<extra></extra>",
    ))

    # 포트폴리오 손실
    fig.add_trace(go.Bar(
        y=scenarios,
        x=portfolio_losses,
        name="Portfolio Loss",
        orientation='h',
        marker=dict(color='#EF4444'),
        text=[f"{x:.1f}%" for x in portfolio_losses],
        textposition='outside',
        hovertemplate="<b>Portfolio Loss</b><br>%{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Stress Test: Historical Crisis Scenarios</b><br><sub>Portfolio Loss vs Market Loss</sub>",
            font=dict(size=18)
        ),
        xaxis_title="Loss (%)",
        yaxis_title="Scenario",
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        hovermode='closest',
        height=450,
        margin=dict(l=200, r=100, t=100, b=80),
        xaxis=dict(
            gridcolor='#1E293B',
            zeroline=True,
            zerolinecolor='#6366F1',
            zerolinewidth=1,
        ),
        yaxis=dict(
            gridcolor='#1E293B',
        ),
        font=dict(family="Courier New, monospace", size=11, color='#E2E8F0'),
    )

    return fig


def create_stress_path_chart(result: StressResult) -> go.Figure:
    """
    선택된 시나리오의 일별 포트폴리오 가치 경로
    - 시작점=100으로 정규화
    - 최저점 마커
    - 회복 경로 포함
    - 다크 테마
    """
    path_normalized = result.daily_path * 100
    days = np.arange(len(path_normalized))

    # 최저점 찾기
    min_idx = np.argmin(result.daily_path)
    min_val = path_normalized[min_idx]

    fig = go.Figure()

    # 메인 경로
    fig.add_trace(go.Scatter(
        x=days,
        y=path_normalized,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#EF4444', width=3),
        hovertemplate="<b>Day %{x}</b><br>Portfolio: %{y:.1f}<extra></extra>",
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)',
    ))

    # 최저점 마커
    fig.add_trace(go.Scatter(
        x=[min_idx],
        y=[min_val],
        mode='markers+text',
        name='Lowest Point',
        marker=dict(color='#EF4444', size=12),
        text=[f"{min_val:.1f}"],
        textposition='top center',
        textfont=dict(size=10, color='#EF4444'),
        hovertemplate="<b>Lowest Point</b><br>Day: %{x}<br>Value: %{y:.1f}<extra></extra>",
    ))

    # 목표선 (최종값)
    final_val = path_normalized[-1]
    fig.add_hline(
        y=100,
        line_dash='dash',
        line_color='#10B981',
        line_width=1,
        annotation_text='Start',
        annotation_position='right',
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{result.scenario.name}</b><br><sub>{result.scenario.description}</sub>",
            font=dict(size=16)
        ),
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value (Indexed = 100)",
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        hovermode='x unified',
        height=500,
        margin=dict(l=80, r=80, t=120, b=80),
        xaxis=dict(
            gridcolor='#1E293B',
        ),
        yaxis=dict(
            gridcolor='#1E293B',
        ),
        font=dict(family="Courier New, monospace", size=11, color='#E2E8F0'),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(15, 23, 42, 0.8)'),
    )

    # 추가 정보 텍스트
    info_text = (
        f"<b>Loss:</b> {result.portfolio_loss*100:.2f}% | "
        f"<b>Worst Day:</b> {result.worst_day*100:.2f}% | "
        f"<b>VaR(95%):</b> {result.var_95*100:.2f}% | "
        f"<b>Recovery Est.:</b> {result.recovery_estimate_days} days"
    )
    fig.add_annotation(
        text=info_text,
        xref='paper', yref='paper',
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=10, color='#94A3B8'),
        xanchor='center',
    )

    return fig


def create_sector_impact_treemap(
    result: StressResult,
    weights: Dict[str, float],
    sector_map: Dict[str, str],
) -> go.Figure:
    """
    섹터별 위기 영향도 트리맵
    - 크기: 비중
    - 색상: 손실 정도 (빨간색 강도)
    - 다크 테마
    """
    # 섹터별 비중과 손실 계산
    sector_weights = {}
    sector_losses_agg = {}

    for ticker, weight in weights.items():
        sector = sector_map.get(ticker, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0) + weight

    # 손실은 이미 계산되어 있음
    for sector, loss in result.sector_losses.items():
        sector_losses_agg[sector] = loss

    # Unknown 섹터 처리
    if "Unknown" in sector_weights and "Unknown" not in sector_losses_agg:
        sector_losses_agg["Unknown"] = result.scenario.market_decline

    # 트리맵 데이터
    labels = list(sector_weights.keys())
    values = list(sector_weights.values())
    losses = [sector_losses_agg.get(s, result.scenario.market_decline) for s in labels]
    loss_percentages = [l * 100 for l in losses]

    # 색상 맵: 손실이 클수록 빨간색, 작을수록 초록색
    colors = []
    for loss in losses:
        if loss < -0.10:  # > 10% 손실
            colors.append('#EF4444')  # 빨강
        elif loss < 0:
            colors.append('#FB923C')  # 주황
        else:
            colors.append('#10B981')  # 초록

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[''] * len(labels),
        values=values,
        marker=dict(
            colors=loss_percentages,
            colorscale='RdYlGn_r',
            cmid=0,
            cmin=min(loss_percentages) if loss_percentages else -50,
            cmax=max(loss_percentages) if loss_percentages else 0,
            showscale=True,
            colorbar=dict(
                title="Loss %",
                tickformat='.1f',
                len=0.7,
            ),
        ),
        text=[f"<b>{label}</b><br>{val*100:.1f}%<br>Loss: {loss*100:.1f}%"
              for label, val, loss in zip(labels, values, losses)],
        textposition='middle center',
        hovertemplate='<b>%{label}</b><br>Weight: %{value*100:.1f}%<br>Loss: %{customdata:.1f}%<extra></extra>',
        customdata=loss_percentages,
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Sector Impact in {result.scenario.name}</b><br><sub>Size = Weight | Color = Loss Severity</sub>",
            font=dict(size=16)
        ),
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        height=500,
        margin=dict(l=20, r=20, t=100, b=20),
        font=dict(family="Courier New, monospace", size=11, color='#E2E8F0'),
    )

    return fig


def create_recovery_timeline(
    results: Dict[str, StressResult],
) -> go.Figure:
    """
    시나리오별 회복 시간대 간트 차트
    - X축: 거래일
    - 바의 길이: 회복까지 소요 시간
    - 색상: 손실 정도
    """
    scenarios = []
    durations = []
    recovery_times = []
    losses = []

    for scenario_name, result in results.items():
        scenarios.append(result.scenario.name)
        durations.append(result.scenario.duration_days)
        recovery_times.append(result.recovery_estimate_days)
        losses.append(result.portfolio_loss * 100)

    fig = go.Figure()

    # 위기 기간
    fig.add_trace(go.Bar(
        y=scenarios,
        x=durations,
        name='Crisis Duration',
        orientation='h',
        marker=dict(color='#EF4444'),
        text=[f"{d} days" for d in durations],
        textposition='inside',
        hovertemplate="<b>Crisis</b><br>%{x} days<extra></extra>",
    ))

    # 회복 기간
    fig.add_trace(go.Bar(
        y=scenarios,
        x=recovery_times,
        name='Recovery Time',
        orientation='h',
        marker=dict(color='#10B981'),
        text=[f"{r} days" for r in recovery_times],
        textposition='inside',
        hovertemplate="<b>Recovery</b><br>%{x} days<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Crisis Duration vs Recovery Timeline</b>",
            font=dict(size=16)
        ),
        xaxis_title="Trading Days",
        yaxis_title="Scenario",
        barmode='stack',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        hovermode='closest',
        height=400,
        margin=dict(l=200, r=100, t=100, b=80),
        xaxis=dict(
            gridcolor='#1E293B',
        ),
        yaxis=dict(
            gridcolor='#1E293B',
        ),
        font=dict(family="Courier New, monospace", size=11, color='#E2E8F0'),
        legend=dict(x=0.99, y=0.01, xanchor='right', yanchor='bottom'),
    )

    return fig


def create_var_comparison(results: Dict[str, StressResult]) -> go.Figure:
    """
    VaR(95%) 비교 차트
    - 각 시나리오의 95% VaR
    - 가장 극심한 하강 정도
    """
    scenarios = []
    var_values = []
    worst_days = []

    for scenario_name, result in results.items():
        scenarios.append(result.scenario.name)
        var_values.append(result.var_95 * 100)
        worst_days.append(result.worst_day * 100)

    fig = go.Figure()

    # VaR(95%)
    fig.add_trace(go.Bar(
        x=scenarios,
        y=var_values,
        name='VaR (95%)',
        marker=dict(color='#6366F1'),
        text=[f"{v:.2f}%" for v in var_values],
        textposition='outside',
        hovertemplate="<b>VaR (95%)</b><br>%{y:.2f}%<extra></extra>",
    ))

    # Worst Day
    fig.add_trace(go.Bar(
        x=scenarios,
        y=worst_days,
        name='Worst Day',
        marker=dict(color='#EF4444'),
        text=[f"{w:.2f}%" for w in worst_days],
        textposition='outside',
        hovertemplate="<b>Worst Day</b><br>%{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Risk Metrics: VaR & Worst Day Loss</b>",
            font=dict(size=16)
        ),
        xaxis_title="Scenario",
        yaxis_title="Loss (%)",
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        hovermode='x unified',
        height=450,
        margin=dict(l=80, r=80, t=100, b=100),
        xaxis=dict(
            gridcolor='#1E293B',
            tickangle=-45,
        ),
        yaxis=dict(
            gridcolor='#1E293B',
        ),
        font=dict(family="Courier New, monospace", size=11, color='#E2E8F0'),
    )

    return fig
