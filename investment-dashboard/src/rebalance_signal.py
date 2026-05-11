"""
AI Rebalancing Signal Engine: 리밸런싱 신호 생성 및 시각화

포트폴리오 리밸런싱 필요성을 진단하는 모듈:
- 시장 레짐 + 스킬 점수 + 포트폴리오 통계를 조합
- 긴급도(urgency) 0-100 점수 생성
- 방어적/공격적/유지 방향 제시
- 리밸런싱 사유 및 추천 액션 제공
- 시각화: 게이지 차트, 타임라인, 해석 문구
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# ===== Color Palette =====
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

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["background"],
    plot_bgcolor=COLORS["background"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(l=40, r=40, t=60, b=40),
)


@dataclass
class RebalanceSignal:
    """리밸런싱 신호 데이터 클래스"""
    urgency: float  # 0-100 점수
    direction: str  # "방어적", "공격적", "유지"
    reasons: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    signal_date: datetime = field(default_factory=datetime.now)

    # 세부 점수 (투명성)
    regime_score: float = 0.0
    skill_score: float = 0.0
    volatility_score: float = 0.0
    tracking_error_score: float = 0.0
    drawdown_score: float = 0.0

    def __post_init__(self):
        """신호 생성 후 유효성 검사"""
        self.urgency = max(0, min(100, self.urgency))


def generate_rebalance_signal(
    skills_dict: Dict[str, float],
    regime_info: Dict,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: pd.Series,
) -> RebalanceSignal:
    """
    여러 신호를 결합하여 리밸런싱 신호 생성

    Parameters:
        skills_dict: {skill_name: score(0-100)} - 6가지 스킬 점수
        regime_info: {
            'current_regime': str ('Bull'/'Bear'/'Sideways'),
            'days_in_regime': int,
            'regime_probability': float (0-1),
            ...
        }
        portfolio_returns: 포트폴리오 수익률 시계열
        benchmark_returns: 벤치마크 수익률 시계열
        weights: 포트폴리오 비중 (합=1.0)

    Returns:
        RebalanceSignal 객체
    """
    signal = RebalanceSignal(urgency=50, direction="유지")

    # ===== 1. 레짐 점수 =====
    regime = regime_info.get("current_regime", "Sideways")
    regime_score = _calc_regime_score(regime, skills_dict)
    signal.regime_score = regime_score

    # ===== 2. 스킬 점수 =====
    skill_score = _calc_skill_score(skills_dict)
    signal.skill_score = skill_score

    # ===== 3. 변동성 점수 (최근 드로다운) =====
    volatility_score, recent_drawdown = _calc_volatility_score(portfolio_returns)
    signal.volatility_score = volatility_score
    signal.drawdown_score = recent_drawdown

    # ===== 4. 추적오차 점수 =====
    tracking_error_score, tracking_error = _calc_tracking_error_score(
        portfolio_returns, benchmark_returns
    )
    signal.tracking_error_score = tracking_error_score

    # ===== 5. 가중 점수 조합 =====
    weights_combined = {
        "regime": 0.30,
        "skill": 0.25,
        "volatility": 0.20,
        "tracking_error": 0.15,
        "concentration": 0.10,
    }

    # 집중도 점수
    concentration_score = _calc_concentration_score(weights)

    urgency = (
        regime_score * weights_combined["regime"] +
        skill_score * weights_combined["skill"] +
        volatility_score * weights_combined["volatility"] +
        tracking_error_score * weights_combined["tracking_error"] +
        concentration_score * weights_combined["concentration"]
    )

    signal.urgency = urgency

    # ===== 6. 방향 결정 =====
    signal = _determine_direction(signal, regime, skills_dict, recent_drawdown)

    # ===== 7. 사유 및 액션 생성 =====
    signal = _generate_reasons_and_actions(
        signal, regime, skills_dict, recent_drawdown, tracking_error
    )

    return signal


def _calc_regime_score(regime: str, skills_dict: Dict[str, float]) -> float:
    """
    레짐 점수 계산

    Bear + 낮은 적응력 → 높은 긴급도
    Bull + 높은 확신 → 낮은 긴급도
    """
    adaptability = skills_dict.get("Adaptability", 50)
    conviction = skills_dict.get("Conviction", 50)

    if regime == "Bear":
        # Bear 시장에서 적응력이 낮으면 긴급
        score = 70 + (100 - adaptability) * 0.3  # 70~100
        return min(100, score)
    elif regime == "Bull":
        # Bull 시장에서 확신이 높으면 안정적
        if conviction > 70:
            score = 30  # 낮은 긴급도
        else:
            score = 50
        return score
    else:  # Sideways
        return 55  # 중간 수준


def _calc_skill_score(skills_dict: Dict[str, float]) -> float:
    """
    스킬 점수 계산

    전반적인 역량이 높으면 안정적
    전반적으로 낮으면 긴급
    """
    scores = list(skills_dict.values())
    avg_skill = np.mean(scores)

    # 역함수: 스킬이 높을수록 낮은 긴급도
    score = 100 - avg_skill
    return max(0, min(100, score))


def _calc_volatility_score(portfolio_returns: pd.Series) -> Tuple[float, float]:
    """
    변동성 점수 + 최근 드로다운

    최근 20일 드로다운 > 10% → 높은 긴급도
    """
    if len(portfolio_returns) < 2:
        return 50.0, 0.0

    # 누적 수익률
    cumulative = (1 + portfolio_returns).cumprod()

    # 20일 rolling drawdown
    rolling_max = cumulative.rolling(window=20, min_periods=1).max()
    drawdown = (cumulative - rolling_max) / rolling_max
    recent_drawdown = abs(drawdown.iloc[-1] * 100)  # 퍼센트

    # 점수: 드로다운이 크면 높은 점수
    if recent_drawdown > 10:
        score = min(100, 50 + recent_drawdown * 2)
    else:
        score = max(0, 50 - recent_drawdown * 2)

    return score, recent_drawdown


def _calc_tracking_error_score(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Tuple[float, float]:
    """
    추적오차 점수

    추적오차 > 15% → 리밸런싱 필요
    """
    if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
        return 50.0, 0.0

    # 길이 맞추기
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    port_ret = portfolio_returns.iloc[-min_len:].values
    bench_ret = benchmark_returns.iloc[-min_len:].values

    # 추적오차 = 초과수익률의 표준편차
    excess_returns = port_ret - bench_ret
    tracking_error = np.std(excess_returns) * np.sqrt(252) * 100  # 연환산

    # 점수: 추적오차가 크면 높은 점수
    if tracking_error > 15:
        score = min(100, 50 + tracking_error * 2)
    else:
        score = max(0, 50 - tracking_error)

    return score, tracking_error


def _calc_concentration_score(weights: pd.Series) -> float:
    """
    포트폴리오 집중도 점수

    집중도가 높으면 리밸런싱 권장
    HHI (Herfindahl-Hirschman Index) 사용
    """
    if weights is None or len(weights) == 0:
        return 50.0

    # HHI: sum(weight^2)
    # 분산: 1/n
    # HHI = 0 (완전 분산) ~ 1 (완전 집중)
    hhi = np.sum(weights.values ** 2)

    # HHI > 0.25 (상위 4개 집중) → 높은 긴급도
    if hhi > 0.25:
        score = min(100, 50 + (hhi - 0.25) * 100)
    else:
        score = max(0, 50 - (0.25 - hhi) * 100)

    return score


def _determine_direction(
    signal: RebalanceSignal,
    regime: str,
    skills_dict: Dict[str, float],
    recent_drawdown: float,
) -> RebalanceSignal:
    """리밸런싱 방향 결정"""

    adaptability = skills_dict.get("Adaptability", 50)
    conviction = skills_dict.get("Conviction", 50)

    # 규칙 1: Bear + 낮은 적응력 + 큰 드로다운 → 방어적
    if regime == "Bear" and adaptability < 50 and recent_drawdown > 10:
        signal.direction = "방어적"
    # 규칙 2: Bull + 높은 확신 + 낮은 드로다운 → 공격적
    elif regime == "Bull" and conviction > 70 and recent_drawdown < 5:
        signal.direction = "공격적"
    # 규칙 3: 긴급도 > 70 → 방어적
    elif signal.urgency > 70:
        signal.direction = "방어적"
    # 규칙 4: 긴급도 < 30 → 공격적
    elif signal.urgency < 30:
        signal.direction = "공격적"
    # 기본값: 유지
    else:
        signal.direction = "유지"

    return signal


def _generate_reasons_and_actions(
    signal: RebalanceSignal,
    regime: str,
    skills_dict: Dict[str, float],
    recent_drawdown: float,
    tracking_error: float,
) -> RebalanceSignal:
    """사유 및 추천 액션 생성"""

    reasons = []
    actions = []

    # 사유 1: 레짐
    if regime == "Bear":
        reasons.append("현재 약세장(Bear) 국면")
    elif regime == "Bull":
        reasons.append("현재 강세장(Bull) 국면")
    else:
        reasons.append("횡보장(Sideways) 국면 지속")

    # 사유 2: 드로다운
    if recent_drawdown > 10:
        reasons.append(f"최근 {recent_drawdown:.1f}% 낙폭 발생")
    elif recent_drawdown > 5:
        reasons.append(f"최근 {recent_drawdown:.1f}% 정도의 낙폭")

    # 사유 3: 추적오차
    if tracking_error > 15:
        reasons.append(f"추적오차({tracking_error:.1f}%) 초과 - 리밸런싱 필요")

    # 사유 4: 스킬
    weakest_skill = min(skills_dict, key=skills_dict.get)
    weakest_score = skills_dict.get(weakest_skill, 0)
    if weakest_score < 40:
        reasons.append(f"약점 스킬: {weakest_skill} ({weakest_score:.0f}점)")

    signal.reasons = reasons

    # 액션
    if signal.direction == "방어적":
        actions.append("고위험 자산 비중 감소 검토")
        actions.append("방어적 자산(채권, 안정성) 비중 증가")
        actions.append("손실 제한 (stop-loss) 설정")
    elif signal.direction == "공격적":
        actions.append("고성장 자산 비중 증가 검토")
        actions.append("포트폴리오 집중도 조정")
        actions.append("수익 목표 상향 조정")
    else:  # 유지
        actions.append("현재 비중 유지")
        actions.append("약한 스킬 개선에 집중")
        actions.append("정기 리밸런싱(분기/반기) 진행")

    # 추적오차 관련 액션
    if tracking_error > 15:
        actions.append("포트폴리오 구성 재검토")
        actions.append("벤치마크와의 괴리도 축소")

    signal.suggested_actions = actions[:3]  # 상위 3개만

    return signal


def create_signal_gauge(signal: RebalanceSignal) -> go.Figure:
    """
    리밸런싱 긴급도 게이지 차트

    0-30: 초록색 (낮음)
    30-60: 노란색 (중간)
    60-100: 빨강색 (높음)
    """

    # 방향별 색상
    if signal.urgency < 30:
        color = COLORS["positive"]  # 초록
    elif signal.urgency < 60:
        color = COLORS["neutral"]  # 노랑
    else:
        color = COLORS["negative"]  # 빨강

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=signal.urgency,
        title=dict(text="리밸런싱 긴급도", font=dict(size=18)),
        delta=dict(reference=50, suffix=" pt"),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=color, thickness=0.25),
            steps=[
                dict(range=[0, 30], color="rgba(16, 185, 129, 0.1)"),
                dict(range=[30, 60], color="rgba(245, 158, 11, 0.1)"),
                dict(range=[60, 100], color="rgba(239, 68, 68, 0.1)"),
            ],
            threshold=dict(
                line=dict(color="white", width=4),
                thickness=0.75,
                value=70,
            ),
        ),
        number=dict(font=dict(size=36, color=color)),
    ))

    layout_kw = {**LAYOUT_DEFAULTS, "margin": dict(l=20, r=20, t=80, b=20)}
    fig.update_layout(
        **layout_kw,
        height=400,
    )

    return fig


def create_signal_timeline(signals_history: List[RebalanceSignal]) -> go.Figure:
    """
    과거 리밸런싱 신호 타임라인

    X축: 날짜
    Y축: 긴급도 (0-100)
    색상: 방향 (방어적/유지/공격적)
    """

    if not signals_history or len(signals_history) == 0:
        # 빈 차트 반환
        fig = go.Figure()
        fig.add_annotation(
            text="신호 이력이 없습니다.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text"]),
        )
        fig.update_layout(
            **LAYOUT_DEFAULTS,
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        )
        return fig

    # 데이터 준비
    dates = [s.signal_date for s in signals_history]
    urgencies = [s.urgency for s in signals_history]
    directions = [s.direction for s in signals_history]

    # 방향별 색상
    direction_colors = {
        "방어적": COLORS["negative"],
        "공격적": COLORS["positive"],
        "유지": COLORS["neutral"],
    }
    colors = [direction_colors.get(d, COLORS["primary"]) for d in directions]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=urgencies,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=10, color=colors, line=dict(width=2, color=COLORS["text"])),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.1)",
        name="긴급도",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>긴급도: %{y:.0f}<extra></extra>",
    ))

    # 배경 영역
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["positive"],
                  annotation_text="낮음", annotation_position="right")
    fig.add_hline(y=60, line_dash="dash", line_color=COLORS["negative"],
                  annotation_text="높음", annotation_position="right")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="리밸런싱 신호 이력", x=0.5, font=dict(size=16)),
        xaxis=dict(title="날짜", gridcolor=COLORS["grid"]),
        yaxis=dict(title="긴급도 점수", range=[0, 100], gridcolor=COLORS["grid"]),
        hovermode="x unified",
        height=300,
    )

    return fig


def get_signal_interpretation(signal: RebalanceSignal) -> str:
    """
    리밸런싱 신호를 한글로 해석

    신호 강도 + 방향 + 사유 + 추천 액션을 통합한 텍스트
    """

    # 긴급도 해석
    if signal.urgency < 30:
        urgency_text = "현재 포트폴리오 상태가 양호하며 리밸런싱이 시급하지 않습니다."
    elif signal.urgency < 60:
        urgency_text = "포트폴리오 조정을 검토할 시점입니다."
    else:
        urgency_text = "리밸런싱이 긴급히 필요합니다."

    # 방향 해석
    direction_detail = {
        "방어적": "위험 자산을 축소하고 안정적인 자산을 강화해야 합니다.",
        "공격적": "성장 자산을 확대하고 포트폴리오 수익성을 추구할 수 있습니다.",
        "유지": "현재 포트폴리오 구성을 유지하면서 지속적으로 모니터링하세요.",
    }

    direction_text = direction_detail.get(signal.direction, "포트폴리오 유지")

    # 종합 해석
    interpretation = f"""{urgency_text}

【리밸런싱 방향: {signal.direction}】
{direction_text}

【신호 사유】
{chr(10).join([f'• {r}' for r in signal.reasons])}

【추천 액션】
{chr(10).join([f'• {a}' for a in signal.suggested_actions])}

【상세 점수】
- 레짐 점수: {signal.regime_score:.1f}/100
- 스킬 점수: {signal.skill_score:.1f}/100
- 변동성 점수: {signal.volatility_score:.1f}/100
- 추적오차 점수: {signal.tracking_error_score:.1f}/100

⚠️ 이 신호는 데이터 기반 분석이며, 투자 결정은 개인의 상황과 판단에 따라 결정해주세요.
"""

    return interpretation
