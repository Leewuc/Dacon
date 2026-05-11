"""
AI Commentary Engine: Skills 프로필 기반 자연어 투자 진단 생성

API 키 없이 룰 기반으로 동작하는 NLG (Natural Language Generation) 엔진.
Skills 점수 조합 패턴을 분석하여 맞춤형 코멘터리를 생성한다.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Commentary:
    """AI 코멘터리 결과"""
    summary: str              # 한줄 종합 평가
    diagnosis: str            # 상세 진단 (2~3문단)
    strengths: List[str]      # 강점 리스트
    weaknesses: List[str]     # 약점 리스트
    recommendations: List[str] # 개선 제안 리스트
    risk_alert: Optional[str] # 리스크 경고 (있을 때만)


# =============================================================================
# 등급/점수 유틸리티
# =============================================================================

def _grade(score: float) -> str:
    if score >= 90: return "S"
    elif score >= 75: return "A"
    elif score >= 55: return "B"
    elif score >= 35: return "C"
    else: return "D"


def _grade_kr(grade: str) -> str:
    """등급의 한국어 설명"""
    return {
        "S": "최상위", "A": "우수", "B": "양호",
        "C": "개선 필요", "D": "취약",
    }.get(grade, "")


# =============================================================================
# 개별 Skill 코멘터리
# =============================================================================

def _comment_timing(score: float) -> Tuple[str, Optional[str]]:
    """Timing Skill 코멘터리 → (진단, 제안 or None)"""
    g = _grade(score)
    if g in ("S", "A"):
        return (
            f"매수/매도 타이밍 역량이 {_grade_kr(g)}합니다({score:.0f}점). "
            "시장 저점 근처에서 매수하고 고점 근처에서 매도하는 패턴이 관찰됩니다.",
            None,
        )
    elif g == "B":
        return (
            f"타이밍 역량은 평균 수준입니다({score:.0f}점).",
            "기술적 지표(RSI, MACD 등)를 참고하여 진입/퇴출 시점을 보완해보세요.",
        )
    else:
        return (
            f"타이밍 역량에 개선 여지가 있습니다({score:.0f}점). "
            "고점 매수 또는 저점 매도 경향이 보입니다.",
            "분할 매수/매도 전략을 도입하여 타이밍 리스크를 분산하는 것을 권장합니다. "
            "예: 목표 금액의 30%/30%/40%를 3회에 걸쳐 매수.",
        )


def _comment_diversification(
    score: float,
    sector_map: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[str, Optional[str]]:
    """Diversification Skill 코멘터리"""
    g = _grade(score)

    # 섹터 편중 분석
    sector_warning = ""
    if sector_map and weights:
        sector_weights: Dict[str, float] = {}
        for ticker, weight in weights.items():
            sector = sector_map.get(ticker, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        top_sector = max(sector_weights, key=sector_weights.get)
        top_pct = sector_weights[top_sector] * 100

        if top_pct > 50:
            sector_warning = f" {top_sector} 섹터에 {top_pct:.0f}%가 집중되어 있어 섹터 리스크가 높습니다."
        elif top_pct > 35:
            sector_warning = f" {top_sector} 섹터 비중이 {top_pct:.0f}%로 다소 편중되어 있습니다."

    if g in ("S", "A"):
        return (
            f"분산투자가 {_grade_kr(g)}하게 이루어지고 있습니다({score:.0f}점). "
            "종목과 섹터 모두 고르게 분배되어 있습니다.",
            None,
        )
    elif g == "B":
        return (
            f"분산투자 수준은 양호합니다({score:.0f}점).{sector_warning}",
            "추가 섹터 분산을 위해 현재 포트폴리오에 없는 업종(예: 헬스케어, 유틸리티)을 검토해보세요." if sector_warning else None,
        )
    else:
        return (
            f"포트폴리오가 소수 종목/섹터에 집중되어 있습니다({score:.0f}점).{sector_warning}",
            "최소 7~10개 종목, 4개 이상의 섹터에 분산하는 것을 권장합니다. "
            "ETF를 활용하면 적은 비용으로 즉시 분산 효과를 얻을 수 있습니다.",
        )


def _comment_risk_management(score: float, detail: Dict) -> Tuple[str, Optional[str]]:
    """Risk Management Skill 코멘터리"""
    g = _grade(score)
    mdd = detail.get("max_drawdown", 0)
    sharpe = detail.get("sharpe_ratio", 0)

    if g in ("S", "A"):
        return (
            f"리스크 관리 역량이 {_grade_kr(g)}합니다({score:.0f}점). "
            f"Sharpe Ratio {sharpe:.2f}, MDD {mdd:.1f}%로 위험 대비 수익이 효율적입니다.",
            None,
        )
    elif g == "B":
        return (
            f"리스크 관리는 평균 수준입니다({score:.0f}점). MDD {mdd:.1f}%.",
            "손절 기준(예: -10% trailing stop)을 설정하면 MDD를 줄일 수 있습니다.",
        )
    else:
        risk_text = f"리스크 관리에 주의가 필요합니다({score:.0f}점). MDD {mdd:.1f}%"
        if mdd > 30:
            risk_text += "로, 원금의 30% 이상이 한때 손실된 적이 있습니다."
        else:
            risk_text += "입니다."
        return (
            risk_text,
            "포트폴리오의 변동성을 줄이기 위해 채권 ETF(예: TLT, KOSEF 국고채) 또는 "
            "금(GLD) 같은 안전자산 10~20% 편입을 고려하세요. "
            "또한 주기적 리밸런싱(분기 1회)으로 리스크를 관리하세요.",
        )


def _comment_conviction(score: float, detail: Dict) -> Tuple[str, Optional[str]]:
    """Conviction Skill 코멘터리"""
    g = _grade(score)
    alpha = detail.get("conviction_alpha", 0)
    top_holdings = detail.get("top_holdings", [])

    if g in ("S", "A"):
        top_str = ", ".join(top_holdings[:3]) if top_holdings else "상위 종목"
        return (
            f"확신 투자 역량이 {_grade_kr(g)}합니다({score:.0f}점). "
            f"집중 투자한 {top_str}이(가) 나머지 종목 대비 {alpha:.1f}%p 초과 수익을 기록했습니다.",
            None,
        )
    elif g == "B":
        return (
            f"확신 투자 역량은 평균적입니다({score:.0f}점).",
            "확신이 높은 종목의 비중을 점진적으로 올리되, "
            "한 종목 비중이 25%를 넘지 않도록 관리하세요.",
        )
    else:
        return (
            f"집중 투자한 종목들이 기대만큼 성과를 내지 못했습니다({score:.0f}점). "
            f"Conviction Alpha: {alpha:.1f}%p.",
            "비중을 높인 종목에 대한 분석을 재검토하세요. "
            "확신 종목 선정 시 펀더멘털(실적, 밸류에이션)과 기술적 신호를 함께 확인하는 것이 좋습니다.",
        )


def _comment_adaptability(score: float, detail: Dict) -> Tuple[str, Optional[str]]:
    """Adaptability Skill 코멘터리"""
    g = _grade(score)
    bear_days = detail.get("bear_market_days", 0)

    if g in ("S", "A"):
        return (
            f"시장 변화 적응력이 {_grade_kr(g)}합니다({score:.0f}점). "
            f"베어마켓 구간({bear_days}거래일)에서 벤치마크 대비 방어에 성공했습니다.",
            None,
        )
    elif g == "B":
        return (
            f"시장 적응력은 평균 수준입니다({score:.0f}점).",
            "시장 하락 신호(VIX 급등, 이동평균 이탈) 시 현금 비중을 10~20% 확보하는 "
            "전술적 자산배분 전략을 고려해보세요.",
        )
    else:
        return (
            f"시장 하락기에 포트폴리오 방어가 미흡했습니다({score:.0f}점). "
            "벤치마크보다 더 큰 손실을 기록한 구간이 있습니다.",
            "하락장 방어를 위해: (1) 인버스 ETF를 소량 헤지용으로 편입하거나, "
            "(2) 분기별 리밸런싱으로 승자를 일부 매도하고 패자를 매수하는 역추세 전략을 검토하세요.",
        )


def _comment_consistency(score: float, detail: Dict) -> Tuple[str, Optional[str]]:
    """Consistency Skill 코멘터리"""
    g = _grade(score)
    win_rate = detail.get("win_rate", 0)
    max_streak = detail.get("max_positive_streak", 0)

    if g in ("S", "A"):
        return (
            f"수익의 일관성이 {_grade_kr(g)}합니다({score:.0f}점). "
            f"월간 승률 {win_rate:.0f}%, 최대 연속 양의 수익 {max_streak}개월.",
            None,
        )
    elif g == "B":
        return (
            f"수익 일관성은 양호합니다({score:.0f}점). 월간 승률 {win_rate:.0f}%.",
            "변동성이 큰 종목의 비중을 줄이고, 안정적 배당주나 채권을 편입하면 "
            "일관성을 높일 수 있습니다.",
        )
    else:
        return (
            f"수익이 불안정합니다({score:.0f}점). 월간 승률 {win_rate:.0f}%로, "
            "절반 이상의 달에서 손실이 발생했습니다.",
            "일관성 향상 전략: (1) 자산군을 주식+채권+대안자산으로 분산, "
            "(2) 적립식 투자로 매입 단가 평준화, "
            "(3) 단기 트레이딩보다 중장기 보유 전략 검토.",
        )


# =============================================================================
# 종합 코멘터리 생성
# =============================================================================

def generate_commentary(
    skills_dict: Dict[str, float],
    skill_details: Dict[str, Dict],
    weights: Optional[Dict[str, float]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    total_return: float = 0,
    benchmark_return: float = 0,
) -> Commentary:
    """
    6-Skills 프로필을 기반으로 종합 AI 코멘터리 생성

    Parameters:
        skills_dict: {skill_name: score}
        skill_details: {skill_name: detail_dict}
        weights: 포트폴리오 비중
        sector_map: 섹터 매핑
        total_return: 포트폴리오 총 수익률
        benchmark_return: 벤치마크 총 수익률
    """
    overall = sum(skills_dict.values()) / len(skills_dict)
    overall_grade = _grade(overall)

    # 강점/약점 분류
    sorted_skills = sorted(skills_dict.items(), key=lambda x: x[1], reverse=True)
    strong = [(k, v) for k, v in sorted_skills if v >= 55]
    weak = [(k, v) for k, v in sorted_skills if v < 55]

    # ─── 한줄 요약 ───
    alpha = total_return - benchmark_return
    alpha_str = f"벤치마크 대비 {alpha*100:+.1f}%p" if alpha != 0 else "벤치마크와 동등"

    summary = (
        f"종합 투자 역량 {overall_grade}등급({overall:.0f}점). "
        f"총 수익률 {total_return*100:+.1f}% ({alpha_str}). "
    )

    if strong and weak:
        summary += f"{strong[0][0]}이(가) 가장 강하고, {weak[-1][0]}에 개선 여지가 있습니다."
    elif strong:
        summary += "전반적으로 균형 잡힌 포트폴리오입니다."
    else:
        summary += "전반적인 투자 전략 재점검이 필요합니다."

    # ─── 상세 진단 ───
    diagnosis_parts = []

    # 성과 개요
    if alpha > 0.05:
        diagnosis_parts.append(
            f"분석 기간 동안 포트폴리오는 {total_return*100:.1f}%의 수익을 달성하며 "
            f"벤치마크를 {alpha*100:.1f}%p 초과했습니다. 이는 긍정적인 성과입니다."
        )
    elif alpha < -0.05:
        diagnosis_parts.append(
            f"분석 기간 동안 포트폴리오 수익률은 {total_return*100:.1f}%로, "
            f"벤치마크 대비 {abs(alpha)*100:.1f}%p 부진했습니다."
        )
    else:
        diagnosis_parts.append(
            f"포트폴리오 수익률은 {total_return*100:.1f}%로 벤치마크와 유사한 수준입니다."
        )

    # 개별 Skill 진단
    skill_comments = []
    skill_recomms = []

    # Timing
    diag, rec = _comment_timing(skills_dict.get("Timing", 50))
    skill_comments.append(diag)
    if rec: skill_recomms.append(rec)

    # Diversification
    diag, rec = _comment_diversification(
        skills_dict.get("Diversification", 50), sector_map, weights
    )
    skill_comments.append(diag)
    if rec: skill_recomms.append(rec)

    # Risk Management
    diag, rec = _comment_risk_management(
        skills_dict.get("Risk Management", 50),
        skill_details.get("Risk Management", {}),
    )
    skill_comments.append(diag)
    if rec: skill_recomms.append(rec)

    # Conviction
    diag, rec = _comment_conviction(
        skills_dict.get("Conviction", 50),
        skill_details.get("Conviction", {}),
    )
    skill_comments.append(diag)
    if rec: skill_recomms.append(rec)

    # Adaptability
    diag, rec = _comment_adaptability(
        skills_dict.get("Adaptability", 50),
        skill_details.get("Adaptability", {}),
    )
    skill_comments.append(diag)
    if rec: skill_recomms.append(rec)

    # Consistency
    diag, rec = _comment_consistency(
        skills_dict.get("Consistency", 50),
        skill_details.get("Consistency", {}),
    )
    skill_comments.append(diag)
    if rec: skill_recomms.append(rec)

    diagnosis = " ".join(diagnosis_parts) + "\n\n" + " ".join(skill_comments)

    # ─── 강점/약점 리스트 ───
    strengths = [
        f"{name} ({_grade(score)}등급, {score:.0f}점)"
        for name, score in strong
    ]
    weaknesses = [
        f"{name} ({_grade(score)}등급, {score:.0f}점)"
        for name, score in weak
    ]

    # ─── 리스크 경고 ───
    risk_alert = None
    mdd = skill_details.get("Risk Management", {}).get("max_drawdown", 0)
    div_score = skills_dict.get("Diversification", 100)

    if mdd > 30 and div_score < 40:
        risk_alert = (
            f"⚠️ 고위험 경고: MDD가 {mdd:.1f}%에 달하며 분산투자도 부족합니다. "
            "포트폴리오 전면 재검토를 강력히 권장합니다."
        )
    elif mdd > 25:
        risk_alert = (
            f"⚠️ 주의: 최대 낙폭이 {mdd:.1f}%입니다. "
            "손실 허용 범위를 재확인하고, 방어 자산 편입을 검토하세요."
        )

    return Commentary(
        summary=summary,
        diagnosis=diagnosis,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=skill_recomms,
        risk_alert=risk_alert,
    )
