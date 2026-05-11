"""
Geopolitical Scenario Engine: 지정학적 위기 시나리오 시뮬레이션

핵심 아이디어:
    매크로 충격(유가, 금리, 환율 등) → 섹터 전파 체인 → 포트폴리오 영향 분석

기능:
    1. 내장 지정학 시나리오 (이란/호르무즈, 대만해협, 우크라이나 확전 등)
    2. 사용자 정의 매크로 충격 (슬라이더로 유가 +80%, 원달러 +15% 등 조합)
    3. 충격 전파 체인: 매크로변수 → 섹터 영향 → 종목 영향
    4. 포트폴리오 손익 시뮬레이션 + 시각화
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# 매크로 충격 변수 정의
# =============================================================================

@dataclass
class MacroShock:
    """단일 매크로 충격"""
    variable: str       # "oil_price", "interest_rate", "usd_krw", etc.
    display_name: str   # "유가 (Brent)"
    change_pct: float   # +0.80 = +80%
    unit: str           # "%", "$/bbl", "원"


# 매크로 변수 → 섹터 영향 전파 매트릭스
# 값: 탄력성 (매크로 변수 1% 변동 → 섹터 수익률 변동폭 %)
# 양수: 같은 방향 (유가 상승 → 에너지주 수혜)
# 음수: 반대 방향 (유가 상승 → 항공주 타격)

SECTOR_SENSITIVITY: Dict[str, Dict[str, float]] = {
    # ── 유가 (Brent) ──
    "oil_price": {
        "에너지": 0.45,
        "Energy": 0.45,
        "유틸리티": -0.15,
        "항공": -0.55,
        "해운": -0.30,
        "화학": -0.25,
        "철강": -0.10,
        "자동차": -0.15,
        "자동차부품": -0.12,
        "방산": 0.20,
        "조선": 0.10,
        "2차전지": 0.15,    # 대체에너지 수혜
        "반도체": -0.05,
        "IT/플랫폼": -0.03,
        "게임": -0.02,
        "금융": -0.08,
        "보험": -0.05,
        "증권": -0.08,
        "바이오": -0.03,
        "제약": -0.02,
        "통신": -0.02,
        "식품": -0.08,
        "화장품": -0.05,
        "유통": -0.10,
        "건설": -0.08,
        "건설/지주": -0.07,
        "여행/레저": -0.35,
        "핀테크": -0.05,
        "지주": -0.05,
        "비철금속": 0.10,
        "중공업": 0.05,
        "전자부품": -0.05,
        "Technology": -0.05,
        "Healthcare": -0.03,
        "Financial": -0.08,
        "Consumer": -0.10,
        "Industrial": -0.05,
    },

    # ── 금리 (기준금리 변동 bp) ──
    "interest_rate": {
        "금융": 0.30,
        "Financial": 0.30,
        "보험": 0.25,
        "증권": -0.15,
        "IT/플랫폼": -0.35,
        "게임": -0.25,
        "바이오": -0.30,
        "Technology": -0.35,
        "Healthcare": -0.15,
        "2차전지": -0.30,
        "반도체": -0.20,
        "건설": -0.25,
        "건설/지주": -0.20,
        "유틸리티": 0.10,
        "통신": 0.05,
        "에너지": 0.05,
        "Energy": 0.05,
        "식품": -0.05,
        "화장품": -0.10,
        "유통": -0.15,
        "자동차": -0.15,
        "자동차부품": -0.12,
        "철강": -0.10,
        "화학": -0.10,
        "핀테크": -0.35,
        "방산": -0.05,
        "조선": -0.10,
        "항공": -0.15,
        "해운": -0.10,
        "여행/레저": -0.15,
        "제약": -0.10,
        "지주": -0.10,
        "비철금속": -0.10,
        "중공업": -0.10,
        "전자부품": -0.15,
        "Consumer": -0.10,
        "Industrial": -0.10,
    },

    # ── 원/달러 환율 ──
    "usd_krw": {
        "반도체": 0.25,     # 수출 수혜
        "자동차": 0.20,
        "자동차부품": 0.18,
        "조선": 0.20,
        "해운": 0.10,
        "철강": 0.12,
        "화학": 0.08,
        "2차전지": 0.15,
        "전자부품": 0.15,
        "IT/플랫폼": 0.05,
        "게임": 0.10,        # 해외매출 환효과
        "에너지": -0.15,     # 원자재 수입 비용 증가
        "Energy": -0.15,
        "항공": -0.30,       # 유류비+리스비 달러 지출
        "여행/레저": -0.20,
        "식품": -0.10,       # 원자재 수입
        "유통": -0.08,
        "금융": -0.05,
        "Financial": -0.05,
        "보험": -0.05,
        "바이오": 0.10,      # 해외매출
        "제약": 0.05,
        "방산": 0.10,
        "통신": -0.03,
        "유틸리티": -0.08,
        "화장품": 0.08,      # 해외매출
        "건설": -0.05,
        "건설/지주": -0.05,
        "핀테크": -0.05,
        "중공업": 0.10,
        "비철금속": 0.08,
        "지주": 0.00,
        "증권": -0.05,
        "Technology": 0.15,
        "Healthcare": 0.05,
        "Consumer": -0.05,
        "Industrial": 0.08,
    },

    # ── 글로벌 공급망 스트레스 지수 ──
    "supply_chain": {
        "반도체": -0.30,
        "전자부품": -0.25,
        "자동차": -0.25,
        "자동차부품": -0.30,
        "2차전지": -0.20,
        "화학": -0.15,
        "철강": -0.10,
        "에너지": 0.15,
        "Energy": 0.15,
        "해운": 0.40,        # 운임 급등 수혜
        "조선": 0.15,
        "항공": -0.20,       # 화물 운송 비용
        "식품": -0.15,
        "유통": -0.20,
        "IT/플랫폼": -0.10,
        "게임": -0.05,
        "금융": -0.05,
        "Financial": -0.05,
        "바이오": -0.10,
        "제약": -0.12,
        "방산": -0.05,
        "통신": -0.05,
        "유틸리티": -0.05,
        "중공업": -0.10,
        "건설": -0.10,
        "Technology": -0.20,
        "Healthcare": -0.10,
        "Consumer": -0.15,
        "Industrial": -0.15,
    },

    # ── 지정학적 리스크 프리미엄 (VIX 상승 등) ──
    "geopolitical_risk": {
        "방산": 0.50,        # 방산주 수혜
        "에너지": 0.20,
        "Energy": 0.20,
        "금": 0.30,          # 안전자산
        "금융": -0.20,
        "Financial": -0.20,
        "보험": -0.15,
        "증권": -0.25,
        "IT/플랫폼": -0.20,
        "게임": -0.15,
        "반도체": -0.15,
        "자동차": -0.15,
        "자동차부품": -0.12,
        "2차전지": -0.12,
        "바이오": -0.08,
        "제약": -0.05,
        "유틸리티": 0.05,
        "통신": 0.03,
        "식품": 0.05,        # 필수소비재 방어
        "화장품": -0.10,
        "유통": -0.10,
        "건설": -0.15,
        "건설/지주": -0.12,
        "철강": -0.10,
        "화학": -0.10,
        "조선": 0.10,
        "해운": 0.05,
        "항공": -0.25,
        "여행/레저": -0.35,
        "핀테크": -0.18,
        "지주": -0.10,
        "비철금속": 0.05,
        "중공업": 0.10,
        "전자부품": -0.10,
        "Technology": -0.15,
        "Healthcare": -0.05,
        "Consumer": -0.08,
        "Industrial": -0.05,
    },

    # ── 글로벌 경기침체 ──
    "recession": {
        "금융": -0.30,
        "Financial": -0.30,
        "보험": -0.20,
        "증권": -0.35,
        "반도체": -0.25,
        "IT/플랫폼": -0.20,
        "게임": -0.10,
        "자동차": -0.30,
        "자동차부품": -0.25,
        "건설": -0.30,
        "건설/지주": -0.25,
        "철강": -0.25,
        "화학": -0.20,
        "에너지": -0.20,
        "Energy": -0.20,
        "유통": -0.20,
        "화장품": -0.15,
        "여행/레저": -0.30,
        "항공": -0.25,
        "해운": -0.20,
        "2차전지": -0.15,
        "핀테크": -0.25,
        "바이오": -0.10,
        "제약": 0.05,        # 경기방어
        "통신": 0.05,        # 경기방어
        "유틸리티": 0.10,    # 경기방어
        "식품": 0.08,        # 필수소비재
        "방산": 0.00,
        "조선": -0.15,
        "비철금속": -0.20,
        "중공업": -0.15,
        "전자부품": -0.20,
        "지주": -0.15,
        "Technology": -0.20,
        "Healthcare": -0.05,
        "Consumer": -0.15,
        "Industrial": -0.20,
    },
}

# 매크로 변수 메타 정보
MACRO_VARIABLES = {
    "oil_price": {"name": "유가 (Brent)", "unit": "%", "range": (-50, 150), "default": 0},
    "interest_rate": {"name": "기준금리 변동", "unit": "bp", "range": (-200, 500), "default": 0},
    "usd_krw": {"name": "원/달러 환율", "unit": "%", "range": (-20, 40), "default": 0},
    "supply_chain": {"name": "공급망 스트레스", "unit": "지수", "range": (0, 100), "default": 0},
    "geopolitical_risk": {"name": "지정학 리스크", "unit": "지수", "range": (0, 100), "default": 0},
    "recession": {"name": "경기침체 강도", "unit": "지수", "range": (0, 100), "default": 0},
}


# =============================================================================
# 내장 지정학 시나리오
# =============================================================================

@dataclass
class GeopoliticalScenario:
    """지정학 시나리오 정의"""
    name: str
    description: str
    narrative: str               # 상세 스토리라인
    macro_shocks: Dict[str, float]  # macro_variable -> change value
    probability: str             # "낮음", "중간", "높음"
    time_horizon: str            # "단기(1~3개월)", "중기(3~12개월)"
    key_triggers: List[str]      # 주요 트리거 이벤트


GEOPOLITICAL_SCENARIOS: Dict[str, GeopoliticalScenario] = {
    "🇮🇷 이란-호르무즈 해협 봉쇄": GeopoliticalScenario(
        name="이란-호르무즈 해협 봉쇄",
        description="이란이 호르무즈 해협을 봉쇄하여 글로벌 원유 수송의 20%가 차단되는 시나리오",
        narrative=(
            "이란-이스라엘 갈등이 전면전으로 확대되면서, 이란이 호르무즈 해협을 봉쇄합니다. "
            "글로벌 원유 수송량의 약 20%(일일 2,100만 배럴)가 차단되면서 유가가 급등합니다. "
            "OPEC 긴급 증산에도 불구하고 단기 공급 부족이 발생하고, "
            "글로벌 인플레이션 우려로 중앙은행의 금리인하 기대가 후퇴합니다. "
            "방산주와 에너지주가 급등하는 반면, 항공·화학·자동차 섹터가 타격을 받습니다. "
            "한국은 원유 수입 의존도가 높아 원/달러 환율이 급등하고, "
            "수출 기업의 원자재 비용 증가로 실적 하향 조정이 예상됩니다."
        ),
        macro_shocks={
            "oil_price": 80,           # 유가 +80%
            "interest_rate": 50,       # 금리 +50bp (인플레 우려)
            "usd_krw": 12,             # 원달러 +12%
            "supply_chain": 60,        # 공급망 스트레스 60
            "geopolitical_risk": 85,   # 지정학 리스크 85
            "recession": 25,           # 경기침체 위험 25
        },
        probability="중간",
        time_horizon="단기(1~3개월)",
        key_triggers=[
            "이란-이스라엘 직접 군사 충돌",
            "호르무즈 해협 기뢰 부설 또는 선박 나포",
            "미국의 이란 석유 시설 공격",
            "IRGC의 걸프만 유조선 공격",
        ],
    ),

    "🇹🇼 대만해협 위기": GeopoliticalScenario(
        name="대만해협 위기",
        description="중국의 대만 봉쇄/침공 시나리오. 글로벌 반도체 공급망 마비",
        narrative=(
            "중국이 대만을 해상 봉쇄하면서 TSMC 등 대만 반도체 공장이 가동 중단됩니다. "
            "글로벌 반도체 공급의 90% 이상이 영향을 받으며, 자동차·전자·IT 산업에 "
            "연쇄 타격이 발생합니다. 미-중 경제 디커플링이 가속화되고, "
            "한국 반도체 기업은 단기적으로 공급 대체 수혜를 받을 수 있으나, "
            "동아시아 지정학 리스크로 한국 시장 전체에 대한 외국인 매도가 진행됩니다. "
            "원/달러 환율이 급등하고, 안전자산 선호로 금값이 치솟습니다."
        ),
        macro_shocks={
            "oil_price": 35,
            "interest_rate": -25,       # 경기 우려로 금리인하 기대
            "usd_krw": 18,
            "supply_chain": 95,         # 공급망 거의 마비
            "geopolitical_risk": 95,
            "recession": 50,
        },
        probability="낮음",
        time_horizon="중기(3~12개월)",
        key_triggers=[
            "중국 인민해방군의 대만 해상 봉쇄 선언",
            "대만해협 실탄 훈련 또는 미사일 발사",
            "미국의 대만 방어 의지 공식 표명",
            "TSMC 생산 중단 발표",
        ],
    ),

    "🇺🇦 우크라이나 전쟁 확전": GeopoliticalScenario(
        name="우크라이나 전쟁 NATO 확전",
        description="러시아-우크라이나 전쟁이 NATO 직접 개입으로 확대",
        narrative=(
            "우크라이나 전쟁이 NATO 회원국으로 확전되면서 유럽 전역이 전시 상태에 진입합니다. "
            "러시아의 추가 에너지 수출 금지로 유럽 에너지 가격이 재폭등하고, "
            "글로벌 곡물 공급망이 다시 경색됩니다. 핵 위협 고조로 안전자산 선호가 극대화되며, "
            "주식시장 전반의 대폭 하락이 예상됩니다. 방산주와 에너지주만이 수혜를 받습니다."
        ),
        macro_shocks={
            "oil_price": 60,
            "interest_rate": -50,
            "usd_krw": 15,
            "supply_chain": 70,
            "geopolitical_risk": 90,
            "recession": 45,
        },
        probability="낮음",
        time_horizon="중기(3~12개월)",
        key_triggers=[
            "러시아의 NATO 회원국 영토 공격",
            "전술핵 사용 위협 현실화",
            "발트3국 또는 폴란드 국경 충돌",
            "러시아 에너지 수출 전면 차단",
        ],
    ),

    "🇰🇵 한반도 긴장 고조": GeopoliticalScenario(
        name="한반도 긴장 고조",
        description="북한 ICBM 발사 + 서해 NLL 도발로 한반도 위기 고조",
        narrative=(
            "북한이 ICBM을 발사하고 NLL 인근에서 군사 도발을 감행합니다. "
            "한반도 전쟁 위험이 고조되면서 외국인 투자자의 한국 자산 대거 매도가 진행됩니다. "
            "KOSPI가 급락하고 원/달러 환율이 급등합니다. "
            "방산주(한화에어로스페이스, 한화오션)가 급등하는 반면, "
            "한국 내수 관련 종목들은 일제히 하락합니다. "
            "다만 과거 패턴상 군사적 긴장이 실제 전쟁으로 이어지지 않을 경우, "
            "2~4주 내 회복하는 '코리아 디스카운트 패턴'이 나타날 수 있습니다."
        ),
        macro_shocks={
            "oil_price": 15,
            "interest_rate": -25,
            "usd_krw": 10,
            "supply_chain": 30,
            "geopolitical_risk": 75,
            "recession": 15,
        },
        probability="중간",
        time_horizon="단기(1~3개월)",
        key_triggers=[
            "북한 ICBM 태평양 발사",
            "NLL 인근 포격 또는 해상 충돌",
            "주한미군 전개 태세 변경 (DEFCON 상향)",
            "서울/수도권 대피 명령",
        ],
    ),

    "💹 글로벌 AI 버블 붕괴": GeopoliticalScenario(
        name="글로벌 AI 버블 붕괴",
        description="AI 과대평가 논란으로 기술주 급락, 닷컴 버블 2.0",
        narrative=(
            "NVIDIA, Microsoft 등 AI 관련 대형 기술주의 실적이 시장 기대에 미치지 못하면서 "
            "'AI 버블' 논란이 본격화됩니다. AI 반도체 수요 둔화 신호에 반도체 섹터가 급락하고, "
            "연쇄적으로 기술주 전반의 멀티플 축소가 진행됩니다. "
            "한국 반도체(삼성전자, SK하이닉스)도 AI 수혜 기대감이 후퇴하면서 동반 하락하고, "
            "성장주에서 가치주로의 대규모 로테이션이 발생합니다. "
            "금융, 필수소비재, 유틸리티 등 방어적 섹터가 상대적 수혜를 받습니다."
        ),
        macro_shocks={
            "oil_price": -15,
            "interest_rate": -75,       # 경기 둔화로 금리인하 기대
            "usd_krw": 5,
            "supply_chain": 15,
            "geopolitical_risk": 20,
            "recession": 40,
        },
        probability="중간",
        time_horizon="중기(3~12개월)",
        key_triggers=[
            "NVIDIA 실적 가이던스 대폭 하향",
            "AI 관련 대형 투자 취소/연기",
            "AI 규제 강화 법안 통과",
            "기술주 ETF(QQQ) -20% 이상 하락",
        ],
    ),

    "🦠 팬데믹 2.0 (신종 감염병)": GeopoliticalScenario(
        name="팬데믹 2.0",
        description="신종 감염병 발생으로 글로벌 봉쇄 조치 재시행",
        narrative=(
            "새로운 고치사율 감염병이 전 세계로 확산되면서 "
            "각국이 국경 봉쇄와 이동 제한을 재시행합니다. "
            "항공, 여행, 오프라인 유통이 직격탄을 맞는 반면, "
            "바이오/제약(백신·치료제), 게임, IT 플랫폼은 수혜를 받습니다. "
            "공급망 마비로 인한 인플레이션과 경기 침체가 동시에 우려되는 "
            "스태그플레이션 시나리오가 펼쳐집니다."
        ),
        macro_shocks={
            "oil_price": -30,
            "interest_rate": -100,
            "usd_krw": 8,
            "supply_chain": 80,
            "geopolitical_risk": 40,
            "recession": 65,
        },
        probability="낮음",
        time_horizon="중기(3~12개월)",
        key_triggers=[
            "WHO 팬데믹 선언",
            "주요국 국경 봉쇄",
            "글로벌 공장 가동 중단",
            "사망률 2% 이상 감염병 확인",
        ],
    ),
}


# =============================================================================
# 충격 전파 엔진
# =============================================================================

@dataclass
class SectorImpact:
    """섹터별 충격 영향"""
    sector: str
    total_impact_pct: float   # 총 영향 (%)
    breakdown: Dict[str, float]  # macro_variable -> contribution
    tickers: List[str]        # 해당 섹터 종목들
    weight_in_portfolio: float


@dataclass
class GeopoliticalResult:
    """지정학 시나리오 분석 결과"""
    scenario_name: str
    narrative: str
    macro_shocks: Dict[str, float]
    sector_impacts: List[SectorImpact]
    portfolio_impact_pct: float    # 포트폴리오 전체 예상 수익률 변동
    worst_sector: str
    best_sector: str
    daily_path: np.ndarray         # 시뮬레이션 일별 경로 (30일)
    risk_score: float              # 0~100 위험 점수


def propagate_shocks(
    macro_shocks: Dict[str, float],
    weights: Dict[str, float],
    sector_map: Dict[str, str],
    duration_days: int = 30,
) -> GeopoliticalResult:
    """
    매크로 충격 → 섹터 전파 → 포트폴리오 영향 계산

    Step 1: 각 매크로 변수의 충격을 섹터 탄력성으로 전파
    Step 2: 각 종목의 섹터 기반으로 종목 영향도 계산
    Step 3: 비중 가중 평균으로 포트폴리오 전체 영향 산출
    Step 4: GBM으로 일별 경로 시뮬레이션
    """

    # ── Step 1: 섹터별 총 충격 계산 ──
    sector_total_impact: Dict[str, Dict[str, float]] = {}  # sector -> {macro_var: impact}

    for macro_var, shock_value in macro_shocks.items():
        if shock_value == 0:
            continue

        sensitivity = SECTOR_SENSITIVITY.get(macro_var, {})

        # 충격 크기 정규화 (유가는 %, 금리는 bp, 지수는 0~100)
        meta = MACRO_VARIABLES.get(macro_var, {})
        var_range = meta.get("range", (-100, 100))
        range_span = var_range[1] - var_range[0]
        normalized_shock = shock_value / range_span * 2 if range_span > 0 else 0

        for sector, elasticity in sensitivity.items():
            if sector not in sector_total_impact:
                sector_total_impact[sector] = {}
            # 섹터 영향 = 충격 × 탄력성
            impact = normalized_shock * elasticity * 100  # percentage
            sector_total_impact[sector][macro_var] = impact

    # ── Step 2: 종목별 영향 & 섹터 집계 ──
    ticker_impacts: Dict[str, float] = {}
    sector_weight_sum: Dict[str, float] = {}
    sector_tickers: Dict[str, List[str]] = {}

    for ticker, weight in weights.items():
        sector = sector_map.get(ticker, "Unknown")

        # 섹터 영향 합산
        breakdown = sector_total_impact.get(sector, {})
        total_impact = sum(breakdown.values())
        ticker_impacts[ticker] = total_impact

        # 섹터별 비중 합산
        sector_weight_sum[sector] = sector_weight_sum.get(sector, 0) + weight
        if sector not in sector_tickers:
            sector_tickers[sector] = []
        sector_tickers[sector].append(ticker)

    # ── Step 3: 포트폴리오 전체 영향 ──
    portfolio_impact = 0.0
    for ticker, weight in weights.items():
        portfolio_impact += weight * ticker_impacts.get(ticker, 0)

    # ── SectorImpact 리스트 생성 ──
    sector_impacts = []
    for sector in set(sector_map.values()):
        if sector in sector_weight_sum:
            breakdown = sector_total_impact.get(sector, {})
            sector_impacts.append(SectorImpact(
                sector=sector,
                total_impact_pct=sum(breakdown.values()),
                breakdown=breakdown,
                tickers=sector_tickers.get(sector, []),
                weight_in_portfolio=sector_weight_sum.get(sector, 0),
            ))

    # Unknown 섹터도 포함
    if "Unknown" in sector_weight_sum:
        sector_impacts.append(SectorImpact(
            sector="Unknown",
            total_impact_pct=0,
            breakdown={},
            tickers=sector_tickers.get("Unknown", []),
            weight_in_portfolio=sector_weight_sum.get("Unknown", 0),
        ))

    # 영향도 순 정렬
    sector_impacts.sort(key=lambda x: x.total_impact_pct)

    worst_sector = sector_impacts[0].sector if sector_impacts else "N/A"
    best_sector = sector_impacts[-1].sector if sector_impacts else "N/A"

    # ── Step 4: 일별 경로 시뮬레이션 (GBM) ──
    target_total = portfolio_impact / 100  # decimal
    daily_drift = target_total / duration_days
    daily_vol = abs(target_total) / (duration_days ** 0.5) * 1.5

    np.random.seed(42)
    daily_returns = np.random.normal(daily_drift, max(daily_vol, 0.005), duration_days)
    # 첫날에 큰 충격, 점점 약해지는 패턴
    shock_decay = np.exp(-np.arange(duration_days) / (duration_days * 0.3))
    daily_returns = daily_returns * (0.3 + 0.7 * shock_decay)
    # 최종 수익률이 target에 수렴하도록 보정
    cum_ret = np.cumprod(1 + daily_returns)
    scale = (1 + target_total) / cum_ret[-1]
    daily_returns = daily_returns * (scale ** (1 / duration_days))
    daily_path = np.cumprod(1 + daily_returns) * 100  # 100 기준

    # ── 위험 점수 ──
    active_shocks = [v for v in macro_shocks.values() if v != 0]
    risk_score = min(100, abs(portfolio_impact) * 2 + len(active_shocks) * 5)

    return GeopoliticalResult(
        scenario_name="Custom" if not hasattr(macro_shocks, '__scenario_name__') else "",
        narrative="",
        macro_shocks=macro_shocks,
        sector_impacts=sector_impacts,
        portfolio_impact_pct=portfolio_impact,
        worst_sector=worst_sector,
        best_sector=best_sector,
        daily_path=daily_path,
        risk_score=risk_score,
    )


def run_geopolitical_scenario(
    scenario: GeopoliticalScenario,
    weights: Dict[str, float],
    sector_map: Dict[str, str],
) -> GeopoliticalResult:
    """내장 시나리오 실행"""
    result = propagate_shocks(
        macro_shocks=scenario.macro_shocks,
        weights=weights,
        sector_map=sector_map,
    )
    result.scenario_name = scenario.name
    result.narrative = scenario.narrative
    return result


# =============================================================================
# Plotly 시각화
# =============================================================================

def create_macro_impact_waterfall(result: GeopoliticalResult) -> "go.Figure":
    """
    매크로 충격 → 포트폴리오 영향 워터폴 차트
    각 매크로 변수의 기여도를 누적으로 표시
    """
    import plotly.graph_objects as go

    # 각 매크로 변수의 포트폴리오 영향 분해
    macro_contributions = {}
    for si in result.sector_impacts:
        for macro_var, impact in si.breakdown.items():
            weighted_impact = impact * si.weight_in_portfolio
            macro_contributions[macro_var] = macro_contributions.get(macro_var, 0) + weighted_impact

    # 정렬 (큰 영향순)
    sorted_vars = sorted(macro_contributions.items(), key=lambda x: x[1])

    labels = []
    values = []
    colors = []

    for var, contrib in sorted_vars:
        meta = MACRO_VARIABLES.get(var, {"name": var})
        shock_val = result.macro_shocks.get(var, 0)
        labels.append(f"{meta['name']}\n({shock_val:+.0f}{meta.get('unit', '')})")
        values.append(contrib)
        colors.append("#EF4444" if contrib < 0 else "#10B981")

    # Total bar
    labels.append("📊 포트폴리오 총 영향")
    values.append(result.portfolio_impact_pct)
    colors.append("#6366F1")

    measures = ["relative"] * (len(labels) - 1) + ["total"]

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=measures,
        textposition="outside",
        text=[f"{v:+.1f}%" for v in values],
        textfont=dict(size=11, color="#E2E8F0"),
        connector=dict(line=dict(color="#475569", width=1)),
        decreasing=dict(marker=dict(color="#EF4444")),
        increasing=dict(marker=dict(color="#10B981")),
        totals=dict(marker=dict(color="#6366F1")),
    ))

    fig.update_layout(
        title="매크로 충격 → 포트폴리오 영향 분해",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=450,
        yaxis_title="포트폴리오 영향 (%)",
        showlegend=False,
        margin=dict(b=100),
    )

    return fig


def create_sector_impact_bar(result: GeopoliticalResult) -> "go.Figure":
    """섹터별 영향도 수평 바 차트"""
    import plotly.graph_objects as go

    sectors = [si.sector for si in result.sector_impacts]
    impacts = [si.total_impact_pct for si in result.sector_impacts]
    weights = [si.weight_in_portfolio * 100 for si in result.sector_impacts]
    colors = ["#EF4444" if v < 0 else "#10B981" for v in impacts]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=sectors,
        x=impacts,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in impacts],
        textposition="outside",
        textfont=dict(size=10, color="#E2E8F0"),
        name="영향도",
        hovertemplate="<b>%{y}</b><br>영향: %{x:+.1f}%<br>비중: %{customdata:.1f}%",
        customdata=weights,
    ))

    fig.update_layout(
        title="섹터별 예상 영향",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=max(350, len(sectors) * 35 + 100),
        xaxis_title="예상 수익률 변동 (%)",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        margin=dict(l=120),
    )

    # 제로라인 강조
    fig.add_vline(x=0, line_dash="dash", line_color="#475569", line_width=1)

    return fig


def create_geopolitical_path(result: GeopoliticalResult) -> "go.Figure":
    """시나리오 발생 시 포트폴리오 가치 경로"""
    import plotly.graph_objects as go

    days = list(range(len(result.daily_path)))

    fig = go.Figure()

    # 기준선
    fig.add_hline(y=100, line_dash="dash", line_color="#475569", line_width=1,
                  annotation_text="기준(100)")

    # 경로
    path_color = "#EF4444" if result.portfolio_impact_pct < 0 else "#10B981"
    fig.add_trace(go.Scatter(
        x=days,
        y=result.daily_path,
        mode="lines",
        fill="tozeroy" if result.portfolio_impact_pct < 0 else None,
        fillcolor=f"rgba({','.join(str(int(path_color[i:i+2], 16)) for i in (1,3,5))},0.1)" if result.portfolio_impact_pct < 0 else None,
        line=dict(color=path_color, width=2.5),
        name="포트폴리오 가치",
        hovertemplate="Day %{x}<br>가치: %{y:.1f}<extra></extra>",
    ))

    # 최저점 마커
    min_idx = np.argmin(result.daily_path)
    fig.add_trace(go.Scatter(
        x=[min_idx],
        y=[result.daily_path[min_idx]],
        mode="markers+text",
        marker=dict(color="#EF4444", size=12, symbol="triangle-down"),
        text=[f"최저: {result.daily_path[min_idx]:.1f}"],
        textposition="bottom center",
        textfont=dict(color="#EF4444", size=11),
        name="최저점",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"시나리오 발생 후 30일 포트폴리오 경로",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=400,
        xaxis_title="거래일",
        yaxis_title="포트폴리오 가치 (시작=100)",
        showlegend=False,
    )

    return fig


def create_scenario_comparison_radar(
    results: Dict[str, GeopoliticalResult],
) -> "go.Figure":
    """여러 시나리오의 매크로 충격 비교 레이더"""
    import plotly.graph_objects as go

    colors = ["#EF4444", "#F59E0B", "#6366F1", "#10B981", "#8B5CF6", "#06B6D4"]
    macro_labels = [MACRO_VARIABLES[k]["name"] for k in MACRO_VARIABLES.keys()]
    macro_keys = list(MACRO_VARIABLES.keys())

    fig = go.Figure()

    for i, (name, result) in enumerate(results.items()):
        values = []
        for key in macro_keys:
            val = abs(result.macro_shocks.get(key, 0))
            meta = MACRO_VARIABLES[key]
            # 정규화 (0~100)
            var_range = meta["range"]
            normalized = val / max(abs(var_range[0]), abs(var_range[1])) * 100
            values.append(min(100, normalized))

        values.append(values[0])  # close

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=macro_labels + [macro_labels[0]],
            name=name[:20],
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="시나리오별 매크로 충격 비교",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=450,
        polar=dict(
            bgcolor="#1E293B",
            radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.35,
            xanchor="center", x=0.5,
            font=dict(size=10),
        ),
    )

    return fig


def create_impact_heatmap(
    results: Dict[str, GeopoliticalResult],
) -> "go.Figure":
    """시나리오 × 섹터 영향도 히트맵"""
    import plotly.graph_objects as go

    # 모든 섹터 수집
    all_sectors = set()
    for result in results.values():
        for si in result.sector_impacts:
            if si.weight_in_portfolio > 0:
                all_sectors.add(si.sector)
    all_sectors = sorted(all_sectors)

    scenario_names = list(results.keys())
    z_data = []

    for name in scenario_names:
        result = results[name]
        sector_dict = {si.sector: si.total_impact_pct for si in result.sector_impacts}
        row = [sector_dict.get(s, 0) for s in all_sectors]
        z_data.append(row)

    # Short names for display
    short_names = [n.split(" ")[1] if len(n.split(" ")) > 1 else n[:10] for n in scenario_names]

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=all_sectors,
        y=short_names,
        colorscale=[
            [0.0, "#EF4444"],
            [0.3, "#FCA5A5"],
            [0.5, "#1E293B"],
            [0.7, "#6EE7B7"],
            [1.0, "#10B981"],
        ],
        zmid=0,
        text=[[f"{v:+.1f}%" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="시나리오: %{y}<br>섹터: %{x}<br>영향: %{z:+.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="시나리오 × 섹터 영향도 매트릭스",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=max(300, len(scenario_names) * 50 + 150),
        xaxis=dict(tickangle=45),
        margin=dict(b=80, l=120),
    )

    return fig
