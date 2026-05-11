"""
Portfolio DNA Fingerprint: 포트폴리오의 고유 특성을 통합한 시각적 신원증명

포트폴리오를 12개 이상의 차원으로 분석하여:
- 독특한 DNA 핑거프린트 생성 (극좌표 레이더 차트)
- 벤치마크와의 비교
- 포트폴리오 아키타입 분류 (예: "안정형 가치투자자", "공격형 모멘텀 트레이더")

차원:
1. Timing (타이밍) - 매매 타이밍 역량
2. Diversification (분산) - 분산투자 역량
3. Risk Management (리스크관리) - 리스크 관리 역량
4. Conviction (확신) - 확신도
5. Adaptability (적응력) - 시장 변화 적응력
6. Consistency (일관성) - 수익 일관성
7. Value (가치) - 가치주 비중/역량
8. Growth (성장) - 성장주 비중/역량
9. Momentum (모멘텀) - 모멘텀 팩터 노출
10. Volatility (변동성) - 저변동성/고변동성
11. Concentration (집중도) - 포트폴리오 집중도 (역함수)
12. Sector Diversity (섹터다양) - 섹터 다양화
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


# ===== Color Palette =====
COLORS = {
    "background": "#0F172A",
    "surface": "#1E293B",
    "text": "#F8FAFC",
    "grid": "#334155",
    "primary": "#6366F1",
    "secondary": "#8B5CF6",
    "accent": "#EC4899",
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

# DNA 차원별 색상 (rainbow gradient)
DNA_COLORS = {
    "Timing": "#FF6B6B",        # 빨강
    "Diversification": "#FF8C42",   # 주황
    "Risk Management": "#FFD93D",   # 노랑
    "Conviction": "#6BCB77",    # 초록
    "Adaptability": "#4D96FF",  # 파랑
    "Consistency": "#9B59B6",   # 보라
    "Value": "#E74C3C",         # 진빨강
    "Growth": "#3498DB",        # 진파랑
    "Momentum": "#F39C12",      # 진주황
    "Volatility": "#1ABC9C",    # 청록
    "Concentration": "#E67E22", # 갈색
    "Sector Diversity": "#95A5A6",  # 회색
}


@dataclass
class PortfolioDNA:
    """포트폴리오 DNA 데이터 클래스"""

    # 6가지 기본 스킬 (0-100)
    skills: Dict[str, float]  # {Timing, Diversification, RiskManagement, Conviction, Adaptability, Consistency}

    # 스타일 정보
    style: str  # "value", "growth", "blend"
    value_score: float  # 0-100
    growth_score: float  # 0-100

    # 팩터 노출
    factor_exposure: Dict[str, float]  # {momentum, quality, dividend, ...}

    # 포트폴리오 특성
    concentration: float  # 0-1 (0=완전분산, 1=완전집중)
    sector_diversity: float  # 0-1 (0=단일섹터, 1=완전분산)

    # 변동성 레짐
    volatility_regime: str  # "low" (방어적), "medium", "high" (공격적)

    # 종합 점수
    archetype: str = ""  # 예: "안정형 가치투자자"
    dna_hash: str = ""  # DNA 핑거프린트 고유 ID

    def __post_init__(self):
        """DNA 아키타입 및 해시 생성"""
        if not self.archetype:
            self.archetype = get_dna_archetype(self)
        if not self.dna_hash:
            self.dna_hash = _generate_dna_hash(self)


def generate_dna(
    skills_dict: Dict[str, float],
    style_info: Dict,
    factor_info: Dict,
    weights: pd.Series,
    vol_regime: str,
) -> PortfolioDNA:
    """
    포트폴리오 DNA 생성

    Parameters:
        skills_dict: {skill_name: score (0-100)}
        style_info: {
            'style': str ('value'/'growth'/'blend'),
            'value_score': float (0-100),
            'growth_score': float (0-100),
        }
        factor_info: {
            'momentum': float (0-100),
            'quality': float (0-100),
            'dividend': float (0-100),
            ...
        }
        weights: 포트폴리오 비중 pd.Series
        vol_regime: 변동성 레짐 ('low'/'medium'/'high')

    Returns:
        PortfolioDNA 객체
    """

    # 기본 스킬
    skills = skills_dict.copy()

    # 스타일
    style = style_info.get("style", "blend")
    value_score = style_info.get("value_score", 50)
    growth_score = style_info.get("growth_score", 50)

    # 팩터
    factor_exposure = factor_info.copy()

    # 집중도 (HHI 기반)
    if weights is not None and len(weights) > 0:
        hhi = np.sum(weights.values ** 2)
        concentration = hhi
    else:
        concentration = 0.5

    # 섹터 다양성 (가정: 8개 이상 섹터면 높음)
    if weights is not None:
        num_sectors = len(weights)
        sector_diversity = min(1.0, num_sectors / 10.0)
    else:
        sector_diversity = 0.5

    dna = PortfolioDNA(
        skills=skills,
        style=style,
        value_score=value_score,
        growth_score=growth_score,
        factor_exposure=factor_exposure,
        concentration=concentration,
        sector_diversity=sector_diversity,
        volatility_regime=vol_regime,
    )

    return dna


def create_dna_fingerprint(dna: PortfolioDNA) -> go.Figure:
    """
    포트폴리오 DNA 핑거프린트 생성 (극좌표 레이더 차트)

    12개 차원을 모두 포함하여 시각적으로 striking한 차트
    - 스킬 6개
    - 스타일 2개 (Value, Growth)
    - 팩터 3개 (Momentum, Quality, Volatility)
    - 포트폴리오 특성 (Concentration 역함수, SectorDiv)

    filled area with gradient-like appearance
    """

    # ===== 12개 차원 데이터 구성 =====
    dimensions = {
        "Timing": dna.skills.get("Timing", 50),
        "Diversification": dna.skills.get("Diversification", 50),
        "Risk Management": dna.skills.get("Risk Management", 50),
        "Conviction": dna.skills.get("Conviction", 50),
        "Adaptability": dna.skills.get("Adaptability", 50),
        "Consistency": dna.skills.get("Consistency", 50),
        "Value": dna.value_score,
        "Growth": dna.growth_score,
        "Momentum": dna.factor_exposure.get("momentum", 50),
        "Volatility": dna.factor_exposure.get("volatility", 50),
        "Concentration": (1 - dna.concentration) * 100,  # 역함수: 집중도 낮을수록 높음
        "Sector Diversity": dna.sector_diversity * 100,
    }

    categories = list(dimensions.keys())
    values = list(dimensions.values())

    # 레이더 닫기
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    # ===== 배경 레이어 (등급 구간) =====
    # S, A, B, C 등급별 색상 배경
    for threshold, color, grade_name in [
        (90, "rgba(16, 185, 129, 0.05)", "S"),
        (75, "rgba(99, 102, 241, 0.05)", "A"),
        (55, "rgba(245, 158, 11, 0.05)", "B"),
        (35, "rgba(239, 68, 68, 0.05)", "C"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=[threshold] * (len(categories) + 1),
            theta=categories_closed,
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="",
        ))

    # ===== 메인 DNA 핑거프린트 =====
    # Gradient 효과를 위해 색상을 동적으로 생성
    main_color = COLORS["primary"]
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.35)",  # 반투명 인디고
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(
            size=12,
            color=COLORS["primary"],
            line=dict(width=2, color=COLORS["text"]),
        ),
        name="포트폴리오 DNA",
        text=[f"{v:.0f}" for v in values_closed],
        textposition="top center",
        mode="lines+markers+text",
        textfont=dict(size=11, color=COLORS["text"], family="monospace"),
        hovertemplate="<b>%{theta}</b><br>점수: %{r:.0f}<extra></extra>",
    ))

    # ===== 레이아웃 =====
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"포트폴리오 DNA 핑거프린트<br><sub>{dna.archetype}</sub>",
            x=0.5,
            font=dict(size=18),
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                gridcolor=COLORS["grid"],
                tickfont=dict(size=9),
            ),
            angularaxis=dict(
                gridcolor=COLORS["grid"],
                tickfont=dict(size=11, color=COLORS["text"]),
            ),
            bgcolor=COLORS["background"],
        ),
        height=650,
        showlegend=True,
        legend=dict(
            x=1.15, y=1.0,
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["grid"],
            borderwidth=1,
        ),
    )

    return fig


def create_dna_comparison(dna: PortfolioDNA, benchmark_dna: PortfolioDNA) -> go.Figure:
    """
    포트폴리오 DNA와 벤치마크 DNA를 겹쳐서 비교

    두 개의 극좌표 영역을 같은 차트에 표시
    """

    # ===== 차원 데이터 =====
    portfolio_dims = {
        "Timing": dna.skills.get("Timing", 50),
        "Diversification": dna.skills.get("Diversification", 50),
        "Risk Management": dna.skills.get("Risk Management", 50),
        "Conviction": dna.skills.get("Conviction", 50),
        "Adaptability": dna.skills.get("Adaptability", 50),
        "Consistency": dna.skills.get("Consistency", 50),
        "Value": dna.value_score,
        "Growth": dna.growth_score,
        "Momentum": dna.factor_exposure.get("momentum", 50),
        "Volatility": dna.factor_exposure.get("volatility", 50),
        "Concentration": (1 - dna.concentration) * 100,
        "Sector Diversity": dna.sector_diversity * 100,
    }

    benchmark_dims = {
        "Timing": benchmark_dna.skills.get("Timing", 50),
        "Diversification": benchmark_dna.skills.get("Diversification", 50),
        "Risk Management": benchmark_dna.skills.get("Risk Management", 50),
        "Conviction": benchmark_dna.skills.get("Conviction", 50),
        "Adaptability": benchmark_dna.skills.get("Adaptability", 50),
        "Consistency": benchmark_dna.skills.get("Consistency", 50),
        "Value": benchmark_dna.value_score,
        "Growth": benchmark_dna.growth_score,
        "Momentum": benchmark_dna.factor_exposure.get("momentum", 50),
        "Volatility": benchmark_dna.factor_exposure.get("volatility", 50),
        "Concentration": (1 - benchmark_dna.concentration) * 100,
        "Sector Diversity": benchmark_dna.sector_diversity * 100,
    }

    categories = list(portfolio_dims.keys())
    portfolio_values = list(portfolio_dims.values())
    benchmark_values = list(benchmark_dims.values())

    # 닫기
    categories_closed = categories + [categories[0]]
    portfolio_closed = portfolio_values + [portfolio_values[0]]
    benchmark_closed = benchmark_values + [benchmark_values[0]]

    fig = go.Figure()

    # 벤치마크 (background)
    fig.add_trace(go.Scatterpolar(
        r=benchmark_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(245, 158, 11, 0.15)",  # 노랑 (벤치마크)
        line=dict(color=COLORS["neutral"], width=2, dash="dash"),
        marker=dict(size=8, color=COLORS["neutral"]),
        name="벤치마크",
        text=[f"{v:.0f}" for v in benchmark_closed],
        textposition="middle center",
        mode="lines+markers+text",
        textfont=dict(size=9, color=COLORS["neutral"]),
        hovertemplate="<b>벤치마크: %{theta}</b><br>점수: %{r:.0f}<extra></extra>",
    ))

    # 포트폴리오 (foreground)
    fig.add_trace(go.Scatterpolar(
        r=portfolio_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.25)",  # 인디고 (포트폴리오)
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=10, color=COLORS["primary"], line=dict(width=2, color=COLORS["text"])),
        name="포트폴리오",
        text=[f"{v:.0f}" for v in portfolio_closed],
        textposition="top center",
        mode="lines+markers+text",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="<b>포트폴리오: %{theta}</b><br>점수: %{r:.0f}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text="포트폴리오 DNA 비교분석",
            x=0.5,
            font=dict(size=18),
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                gridcolor=COLORS["grid"],
                tickfont=dict(size=9),
            ),
            angularaxis=dict(
                gridcolor=COLORS["grid"],
                tickfont=dict(size=10, color=COLORS["text"]),
            ),
            bgcolor=COLORS["background"],
        ),
        height=600,
        showlegend=True,
        legend=dict(
            x=1.15, y=1.0,
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["grid"],
            borderwidth=1,
        ),
    )

    return fig


def get_dna_archetype(dna: PortfolioDNA) -> str:
    """
    포트폴리오 DNA를 분석하여 아키타입 분류 (한글)

    지배적인 특성 기반:
    - Skills dominance: 가장 높은 스킬
    - Style: Value vs Growth
    - Volatility regime: 방어/중립/공격
    - 기타 특성

    반환 예시:
    - "안정형 가치투자자"
    - "공격형 모멘텀 트레이더"
    - "균형잡힌 분산투자자"
    - "고수익 추구형 성장 전문가"
    """

    # 1. 스킬 프로필 분석
    strongest_skill = max(dna.skills, key=dna.skills.get)
    weakest_skill = min(dna.skills, key=dna.skills.get)
    avg_skill = np.mean(list(dna.skills.values()))

    # 2. 스타일 분석
    if dna.value_score > dna.growth_score + 10:
        style_type = "가치"
    elif dna.growth_score > dna.value_score + 10:
        style_type = "성장"
    else:
        style_type = "혼합"

    # 3. 변동성 분석
    if dna.volatility_regime == "low":
        volatility_type = "안정형"
    elif dna.volatility_regime == "high":
        volatility_type = "공격형"
    else:
        volatility_type = "중립형"

    # 4. 팩터 분석
    momentum = dna.factor_exposure.get("momentum", 50)
    quality = dna.factor_exposure.get("quality", 50)

    has_momentum = momentum > 60
    has_quality = quality > 60

    # 5. 다양성 분석
    is_diversified = dna.sector_diversity > 0.6

    # 6. 아키타입 결정 (규칙 기반)
    archetype = ""

    # 규칙 1: 강한 스킬 + 스타일 + 변동성 조합
    if strongest_skill == "Consistency" and dna.skills["Consistency"] > 75:
        if style_type == "가치":
            archetype = "안정적 수익창출 가치투자자"
        else:
            archetype = "일관된 성과 추구 투자자"

    elif strongest_skill == "Timing" and dna.skills["Timing"] > 75:
        if volatility_type == "공격형":
            archetype = "타이밍 감각 우수 트레이더"
        else:
            archetype = "신중한 타이밍 투자자"

    elif strongest_skill == "Conviction" and dna.skills["Conviction"] > 75:
        if has_momentum:
            archetype = "공격형 모멘텀 트레이더"
        else:
            archetype = "확신의 집중투자자"

    elif strongest_skill == "Diversification" and dna.skills["Diversification"] > 75:
        if is_diversified:
            archetype = "균형잡힌 분산투자자"
        else:
            archetype = "포트폴리오 최적화 전문가"

    elif strongest_skill == "Adaptability" and dna.skills["Adaptability"] > 75:
        archetype = "시장변화 적응형 투자자"

    elif strongest_skill == "Risk Management" and dna.skills["Risk Management"] > 75:
        archetype = "리스크 관리 중심 투자자"

    # 규칙 2: 평균 역량 기반
    if not archetype:
        if avg_skill > 70:
            if volatility_type == "공격형":
                if style_type == "성장":
                    archetype = "고수익 추구형 성장전문가"
                else:
                    archetype = "공격적 포트폴리오 운용자"
            else:
                if style_type == "가치":
                    archetype = "안정형 가치투자자"
                else:
                    archetype = "우수한 균형투자자"
        elif avg_skill > 55:
            if volatility_type == "공격형":
                archetype = "중도 공격형 투자자"
            else:
                archetype = "중도 방어형 투자자"
        else:
            archetype = "학습 중인 신규 투자자"

    # 규칙 3: 특수 패턴
    if not archetype:
        if has_momentum and has_quality:
            archetype = "모멘텀 + 퀄리티 추구형 투자자"
        elif has_momentum:
            archetype = "모멘텀 트레이더"
        elif has_quality:
            archetype = "퀄리티 가치 추구형 투자자"

    # 기본값
    if not archetype:
        archetype = "균형잡힌 일반 투자자"

    return archetype


def _generate_dna_hash(dna: PortfolioDNA) -> str:
    """
    포트폴리오 DNA의 고유 해시 생성

    12개 차원의 점수를 기반으로 고유한 ID 생성
    """

    dimensions = {
        "Timing": dna.skills.get("Timing", 50),
        "Diversification": dna.skills.get("Diversification", 50),
        "Risk Management": dna.skills.get("Risk Management", 50),
        "Conviction": dna.skills.get("Conviction", 50),
        "Adaptability": dna.skills.get("Adaptability", 50),
        "Consistency": dna.skills.get("Consistency", 50),
        "Value": dna.value_score,
        "Growth": dna.growth_score,
        "Momentum": dna.factor_exposure.get("momentum", 50),
        "Volatility": dna.factor_exposure.get("volatility", 50),
        "Concentration": (1 - dna.concentration) * 100,
        "Sector Diversity": dna.sector_diversity * 100,
    }

    # 각 차원을 0-9 범위로 양자화
    quantized = [str(int(v / 10)) for v in dimensions.values()]
    hash_string = "DNA-" + "".join(quantized)

    return hash_string
