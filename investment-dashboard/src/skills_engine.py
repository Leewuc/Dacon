"""
Skills Engine: 투자자의 6가지 역량(Skills)을 정량적으로 산출하는 핵심 모듈

Skills:
    1. Timing - 매수/매도 타이밍 역량
    2. Diversification - 분산투자 역량
    3. Risk Management - 리스크 관리 역량
    4. Conviction - 확신 포지션 운용 역량
    5. Adaptability - 시장 변화 적응력
    6. Consistency - 수익 일관성
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class SkillScore:
    """개별 Skill 점수를 담는 데이터 클래스"""
    name: str
    score: float  # 0~100
    detail: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    grade: str = field(init=False)

    def __post_init__(self):
        self.grade = self._calc_grade()
        if not self.description:
            self.description = self._default_description()

    def _calc_grade(self) -> str:
        if self.score >= 90:
            return "S"
        elif self.score >= 75:
            return "A"
        elif self.score >= 55:
            return "B"
        elif self.score >= 35:
            return "C"
        else:
            return "D"

    def _default_description(self) -> str:
        _kr = {
            "Timing": "타이밍",
            "Diversification": "분산투자",
            "Risk Management": "리스크관리",
            "Conviction": "확신도",
            "Adaptability": "적응력",
            "Consistency": "일관성",
        }
        kr_name = _kr.get(self.name, self.name)
        descriptions = {
            "S": f"{kr_name} 역량이 최상위 수준입니다.",
            "A": f"{kr_name} 역량이 우수합니다.",
            "B": f"{kr_name} 역량이 평균 이상입니다.",
            "C": f"{kr_name} 역량에 개선 여지가 있습니다.",
            "D": f"{kr_name} 역량 강화가 필요합니다.",
        }
        return descriptions.get(self.grade, "")


@dataclass
class SkillsProfile:
    """6가지 Skills 종합 프로필"""
    timing: SkillScore
    diversification: SkillScore
    risk_management: SkillScore
    conviction: SkillScore
    adaptability: SkillScore
    consistency: SkillScore

    def to_dict(self) -> Dict[str, float]:
        return {
            "Timing": self.timing.score,
            "Diversification": self.diversification.score,
            "Risk Management": self.risk_management.score,
            "Conviction": self.conviction.score,
            "Adaptability": self.adaptability.score,
            "Consistency": self.consistency.score,
        }

    def overall_score(self) -> float:
        scores = list(self.to_dict().values())
        return np.mean(scores)

    def strongest_skill(self) -> str:
        d = self.to_dict()
        return max(d, key=d.get)

    def weakest_skill(self) -> str:
        d = self.to_dict()
        return min(d, key=d.get)


# =============================================================================
# 유틸리티 함수
# =============================================================================

def calc_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.035, periods: int = 252
) -> float:
    """연환산 Sharpe Ratio 계산"""
    if returns.std() == 0:
        return 0.0
    excess = returns.mean() - risk_free_rate / periods
    return (excess / returns.std()) * np.sqrt(periods)


def calc_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.035, periods: int = 252
) -> float:
    """연환산 Sortino Ratio 계산 (하방 변동성만 사용)"""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    excess = returns.mean() - risk_free_rate / periods
    return (excess / downside.std()) * np.sqrt(periods)


def calc_max_drawdown(returns: pd.Series) -> float:
    """최대 낙폭(MDD) 계산"""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return abs(drawdown.min())


def calc_calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Calmar Ratio = 연환산 수익률 / MDD"""
    mdd = calc_max_drawdown(returns)
    if mdd == 0:
        return 0.0
    annual_return = returns.mean() * periods
    return annual_return / mdd


# =============================================================================
# Skills 산출 함수
# =============================================================================

def calc_timing_skill(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    lookback_days: int = 30,
) -> SkillScore:
    """
    Timing Skill: 매수/매도 타이밍 역량 측정

    Parameters:
        trades_df: columns=['date', 'ticker', 'side', 'price', 'quantity']
        price_df: MultiIndex or single ticker OHLCV DataFrame
        lookback_days: 타이밍 평가 윈도우 (default 30일)

    Returns:
        SkillScore with timing assessment
    """
    if trades_df.empty:
        return SkillScore(name="Timing", score=50.0, detail={"note": "거래 이력 없음"})

    buy_scores = []
    sell_scores = []

    for _, trade in trades_df.iterrows():
        ticker = trade["ticker"]
        trade_date = pd.Timestamp(trade["date"])
        end_date = trade_date + pd.Timedelta(days=lookback_days)

        # 해당 종목의 가격 데이터 추출
        try:
            if isinstance(price_df.columns, pd.MultiIndex):
                ticker_prices = price_df.xs(ticker, axis=1, level=1)
            else:
                ticker_prices = price_df
        except KeyError:
            continue

        window = ticker_prices.loc[
            (ticker_prices.index >= trade_date)
            & (ticker_prices.index <= end_date)
        ]

        if window.empty:
            continue

        if trade["side"].upper() == "BUY":
            best_possible = window["Low"].min()
            if trade["price"] > 0:
                score = 1 - (trade["price"] - best_possible) / trade["price"]
                buy_scores.append(max(0, min(1, score)))
        else:  # SELL
            best_possible = window["High"].max()
            if trade["price"] > 0:
                score = 1 - (best_possible - trade["price"]) / trade["price"]
                sell_scores.append(max(0, min(1, score)))

    all_scores = buy_scores + sell_scores
    if not all_scores:
        return SkillScore(name="Timing", score=50.0, detail={"note": "평가 불가"})

    final_score = np.mean(all_scores) * 100

    return SkillScore(
        name="Timing",
        score=round(final_score, 1),
        detail={
            "avg_buy_timing": round(np.mean(buy_scores) * 100, 1) if buy_scores else 0,
            "avg_sell_timing": round(np.mean(sell_scores) * 100, 1) if sell_scores else 0,
            "total_trades_evaluated": len(all_scores),
        },
    )


def calc_timing_from_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: Dict[str, float],
    returns_by_ticker: Dict[str, float],
) -> SkillScore:
    """
    매매 이력 없이 가격/수익률 데이터만으로 Timing Skill을 추정.

    Method:
    1. Entry Timing Score (진입 타이밍):
       - 분석 기간 시작점 대비 이후 30일 평균 가격이 더 높은지 측정
       - 시작점이 상대적 저점이었다면 좋은 진입 타이밍

    2. Relative Timing Score (상대 타이밍):
       - 포트폴리오의 초반 vs 후반 성과 비교
       - 후반이 더 좋으면 "시장이 올라갈 때 들어갔다" → 타이밍 양호

    3. Dip-buying Score (저점 매수 대리):
       - 벤치마크 하락일에 포트폴리오가 덜 빠지면 저점 대응 능력 우수

    4. Top-performing 종목 비중 (확신 타이밍):
       - 가장 많이 오른 종목에 비중이 높으면 진입 타이밍이 좋았다는 proxy

    Returns:
        SkillScore with timing assessment
    """
    detail = {}

    # --- 1) Entry Timing: 시작점이 저점이었는지 ---
    cum = (1 + portfolio_returns).cumprod()
    n = len(cum)
    if n > 30:
        # 시작 30일 평균 vs 전체 평균
        early_avg = cum.iloc[:30].mean()
        total_avg = cum.mean()
        # 시작점이 전체 평균보다 낮으면 좋은 진입
        entry_ratio = total_avg / early_avg if early_avg > 0 else 1.0
        entry_score = min(100, max(0, (entry_ratio - 0.9) * 500))  # 0.9~1.1 → 0~100
    else:
        entry_score = 50.0
        entry_ratio = 1.0
    detail["진입타이밍 비율 (전체평균/초기평균)"] = round(entry_ratio, 3)
    detail["진입타이밍 점수"] = round(entry_score, 1)

    # --- 2) Relative Timing: 전반 vs 후반 alpha ---
    half = n // 2
    if half > 20:
        first_half_port = (1 + portfolio_returns.iloc[:half]).prod() - 1
        second_half_port = (1 + portfolio_returns.iloc[half:]).prod() - 1
        first_half_bench = (1 + benchmark_returns.iloc[:half]).prod() - 1
        second_half_bench = (1 + benchmark_returns.iloc[half:]).prod() - 1

        alpha_first = first_half_port - first_half_bench
        alpha_second = second_half_port - second_half_bench

        # 후반 알파가 전반보다 높으면 → 시장에 적응하며 나아진 것
        alpha_improvement = alpha_second - alpha_first
        relative_score = min(100, max(0, 50 + alpha_improvement * 300))
    else:
        relative_score = 50.0
        alpha_improvement = 0.0
    detail["알파 개선폭 (후반-전반, %)"] = round(alpha_improvement * 100, 2)
    detail["상대타이밍 점수"] = round(relative_score, 1)

    # --- 3) Dip-buying: 벤치마크 하락일 대비 방어력 ---
    bench_down_days = benchmark_returns < -0.005  # 벤치마크 -0.5% 이상 하락일
    if bench_down_days.sum() > 10:
        port_on_down = portfolio_returns[bench_down_days].mean()
        bench_on_down = benchmark_returns[bench_down_days].mean()
        # 포트폴리오가 벤치마크보다 덜 빠지면 좋음
        dip_alpha = port_on_down - bench_on_down
        dip_score = min(100, max(0, 50 + dip_alpha * 1000))
        n_down_days = int(bench_down_days.sum())
    else:
        dip_score = 50.0
        dip_alpha = 0.0
        n_down_days = 0
    detail["하락방어 알파 (%)"] = round(dip_alpha * 100, 3)
    detail["저점매수 점수"] = round(dip_score, 1)
    detail["벤치마크 하락일 수"] = n_down_days

    # --- 4) Winner Allocation: 상위 종목 비중 점수 ---
    if returns_by_ticker and weights:
        sorted_by_return = sorted(returns_by_ticker.items(), key=lambda x: -x[1])
        top_tickers = [t for t, r in sorted_by_return if r > 0][:3]
        top_weight = sum(weights.get(t, 0) for t in top_tickers)
        total_weight = sum(weights.values())
        winner_ratio = top_weight / total_weight if total_weight > 0 else 0
        winner_score = min(100, max(0, winner_ratio * 200))
        detail["상위3종목 비중 (%)"] = round(top_weight * 100, 1)
        detail["승자배분 점수"] = round(winner_score, 1)
    else:
        winner_score = 50.0
        detail["승자배분 점수"] = 50.0

    # --- 종합 ---
    final_score = (
        0.25 * entry_score
        + 0.25 * relative_score
        + 0.25 * dip_score
        + 0.25 * winner_score
    )
    final_score = min(100, max(0, final_score))
    detail["분석 방법"] = "가격 기반 추정 (매매 이력 없음)"

    return SkillScore(
        name="Timing",
        score=round(final_score, 1),
        detail=detail,
    )


def calc_diversification_skill(
    weights: Dict[str, float],
    sector_map: Optional[Dict[str, str]] = None,
) -> SkillScore:
    """
    Diversification Skill: 분산투자 역량 측정
    HHI(허핀달-허쉬만 지수) + 섹터 엔트로피 결합

    Parameters:
        weights: {ticker: weight} 포트폴리오 비중
        sector_map: {ticker: sector} 섹터 매핑
    """
    if not weights:
        return SkillScore(name="Diversification", score=0.0, detail={"note": "비중 데이터 없음"})

    w = np.array(list(weights.values()))
    w = w / w.sum()  # 정규화

    # HHI 기반 종목 분산도
    hhi = float(np.sum(w ** 2))
    n = len(w)
    min_hhi = 1 / n if n > 0 else 1
    # 정규화: HHI가 min(1/n)이면 100, 1이면 0
    hhi_score = (1 - hhi) / (1 - min_hhi) * 100 if (1 - min_hhi) > 0 else 0

    # 섹터 엔트로피
    entropy_score = 50.0  # 기본값
    if sector_map:
        sector_weights: Dict[str, float] = {}
        for ticker, weight in weights.items():
            sector = sector_map.get(ticker, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        sw = np.array(list(sector_weights.values()))
        sw = sw / sw.sum()
        sw = sw[sw > 0]

        if len(sw) > 1:
            entropy = float(-np.sum(sw * np.log2(sw)))
            max_entropy = np.log2(len(sw))
            entropy_score = (entropy / max_entropy) * 100 if max_entropy > 0 else 0
        else:
            entropy_score = 0.0

    final_score = 0.5 * hhi_score + 0.5 * entropy_score

    return SkillScore(
        name="Diversification",
        score=round(final_score, 1),
        detail={
            "hhi": round(hhi, 4),
            "hhi_score": round(hhi_score, 1),
            "entropy_score": round(entropy_score, 1),
            "num_holdings": n,
        },
    )


def calc_risk_management_skill(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> SkillScore:
    """
    Risk Management Skill: 리스크 관리 역량 측정
    Sharpe, Sortino, MDD를 벤치마크 대비 상대 비교
    """
    sharpe = calc_sharpe_ratio(returns)
    sortino = calc_sortino_ratio(returns)
    mdd = calc_max_drawdown(returns)
    calmar = calc_calmar_ratio(returns)

    bench_sharpe = calc_sharpe_ratio(benchmark_returns)
    bench_mdd = calc_max_drawdown(benchmark_returns)

    # Sharpe 상대 점수
    if bench_sharpe > 0 and bench_sharpe != 0:
        sharpe_score = min(100, max(0, (sharpe / bench_sharpe) * 50))
    elif bench_sharpe == 0 and sharpe > 0:
        sharpe_score = 70
    elif bench_sharpe == 0 and sharpe == 0:
        sharpe_score = 50
    else:
        sharpe_score = 30

    # MDD 상대 점수 (MDD가 작을수록 좋음)
    if mdd > 0 and bench_mdd > 0:
        mdd_score = min(100, max(0, (bench_mdd / mdd) * 50))
    elif mdd == 0:
        mdd_score = 100
    else:
        mdd_score = 30

    # Sortino 절대 점수
    sortino_score = min(100, max(0, 50 + sortino * 15))

    final_score = 0.4 * sharpe_score + 0.3 * mdd_score + 0.3 * sortino_score

    return SkillScore(
        name="Risk Management",
        score=round(final_score, 1),
        detail={
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown": round(mdd * 100, 2),
            "calmar_ratio": round(calmar, 3),
            "benchmark_sharpe": round(bench_sharpe, 3),
            "benchmark_mdd": round(bench_mdd * 100, 2),
        },
    )


def calc_conviction_skill(
    weights: Dict[str, float],
    returns_by_ticker: Dict[str, float],
    top_n: int = 5,
) -> SkillScore:
    """
    Conviction Skill: 확신 포지션 운용 역량
    비중 상위 N개 종목의 성과 vs 나머지
    """
    if not weights or not returns_by_ticker:
        return SkillScore(name="Conviction", score=50.0, detail={"note": "데이터 부족"})

    # 비중 기준 정렬
    sorted_holdings = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top = sorted_holdings[:top_n]
    rest = sorted_holdings[top_n:]

    # 가중 수익률 계산
    top_weighted_return = sum(
        w * returns_by_ticker.get(t, 0) for t, w in top
    )
    top_total_weight = sum(w for _, w in top)

    rest_weighted_return = sum(
        w * returns_by_ticker.get(t, 0) for t, w in rest
    )
    rest_total_weight = sum(w for _, w in rest)

    top_avg = top_weighted_return / top_total_weight if top_total_weight > 0 else 0
    rest_avg = rest_weighted_return / rest_total_weight if rest_total_weight > 0 else 0

    # 확신 알파
    conviction_alpha = top_avg - rest_avg
    final_score = min(100, max(0, 50 + conviction_alpha * 120))

    return SkillScore(
        name="Conviction",
        score=round(final_score, 1),
        detail={
            "top_n_return": round(top_avg * 100, 2),
            "rest_return": round(rest_avg * 100, 2),
            "conviction_alpha": round(conviction_alpha * 100, 2),
            "top_holdings": [t for t, _ in top],
            "concentration_ratio": round(top_total_weight * 100, 1),
        },
    )


def calc_adaptability_skill(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rebalance_dates: Optional[List[pd.Timestamp]] = None,
) -> SkillScore:
    """
    Adaptability Skill: 시장 변화 적응력
    베어마켓 방어력 + 리밸런싱 효과
    """
    # 벤치마크 누적 수익률 기반 드로다운 구간 식별
    bench_cum = (1 + benchmark_returns).cumprod()
    bench_rolling_max = bench_cum.cummax()
    bench_dd = (bench_cum - bench_rolling_max) / bench_rolling_max

    # 베어마켓 = 벤치마크 드로다운 threshold 이하 구간 (상대 임계값)
    bench_volatility = benchmark_returns.std()
    bear_threshold = max(-0.10, -2 * bench_volatility * np.sqrt(252) / 10)
    bear_mask = bench_dd < bear_threshold

    if bear_mask.any():
        # 공통 인덱스만 사용
        common_idx = returns.index.intersection(benchmark_returns.index)
        bear_common = bear_mask.reindex(common_idx).fillna(False)

        port_bear = returns.reindex(common_idx)[bear_common]
        bench_bear = benchmark_returns.reindex(common_idx)[bear_common]

        if len(port_bear) > 0:
            relative_perf = port_bear.mean() - bench_bear.mean()
            bear_score = min(100, max(0, 50 + relative_perf * 500))
        else:
            bear_score = 50
    else:
        bear_score = 50

    # 리밸런싱 효과
    rebal_score = 50.0
    if rebalance_dates and len(rebalance_dates) >= 2:
        post_rebal_returns = []
        for date in rebalance_dates:
            post_window = returns.loc[
                (returns.index >= date)
                & (returns.index < date + pd.Timedelta(days=30))
            ]
            if len(post_window) > 0:
                post_rebal_returns.append(post_window.sum())

        if post_rebal_returns:
            avg_post = np.mean(post_rebal_returns)
            avg_overall = returns.mean() * 21  # ~1개월
            rebal_alpha = avg_post - avg_overall
            rebal_score = min(100, max(0, 50 + rebal_alpha * 300))

    final_score = 0.6 * bear_score + 0.4 * rebal_score

    return SkillScore(
        name="Adaptability",
        score=round(final_score, 1),
        detail={
            "bear_market_score": round(bear_score, 1),
            "rebalancing_score": round(rebal_score, 1),
            "bear_market_days": int(bear_mask.sum()),
        },
    )


def calc_consistency_skill(monthly_returns: pd.Series) -> SkillScore:
    """
    Consistency Skill: 수익 일관성
    양의 수익 월 비율 + 변동계수 + 연속 양의 수익 streak
    """
    if monthly_returns.empty:
        return SkillScore(name="Consistency", score=50.0, detail={"note": "데이터 부족"})

    # 양의 수익 월 비율
    win_rate = float((monthly_returns > 0).mean() * 100)

    # 변동계수 (CV) - 낮을수록 일관성 높음
    mean_ret = monthly_returns.mean()
    std_ret = monthly_returns.std()
    if mean_ret != 0 and not np.isnan(mean_ret):
        cv = abs(std_ret / mean_ret)
        cv_score = 100 / (1 + cv)
    else:
        cv = float("inf")
        cv_score = 0

    # 연속 양의 수익 최대 streak
    streaks = []
    current = 0
    for r in monthly_returns:
        if r > 0:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    max_streak = max(streaks) if streaks else 0
    total_months = len(monthly_returns)
    streak_score = min(100, (max_streak / max(1, total_months * 0.4)) * 100)

    final_score = 0.4 * win_rate + 0.3 * cv_score + 0.3 * streak_score

    return SkillScore(
        name="Consistency",
        score=round(final_score, 1),
        detail={
            "win_rate": round(win_rate, 1),
            "coefficient_of_variation": round(cv, 3) if cv != float("inf") else "N/A",
            "max_positive_streak": max_streak,
            "total_months": len(monthly_returns),
        },
    )


# =============================================================================
# 종합 프로필 생성
# =============================================================================

def generate_skills_profile(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: Dict[str, float],
    returns_by_ticker: Dict[str, float],
    trades_df: Optional[pd.DataFrame] = None,
    price_df: Optional[pd.DataFrame] = None,
    sector_map: Optional[Dict[str, str]] = None,
    rebalance_dates: Optional[List[pd.Timestamp]] = None,
) -> SkillsProfile:
    """
    모든 입력을 받아 6-Skills 종합 프로필 생성

    Parameters:
        portfolio_returns: 일간 포트폴리오 수익률
        benchmark_returns: 일간 벤치마크 수익률
        weights: {ticker: weight} 현재 포트폴리오 비중
        returns_by_ticker: {ticker: total_return} 종목별 누적 수익률
        trades_df: 매매 이력 (optional)
        price_df: 가격 데이터 (optional, timing skill 필요시)
        sector_map: 섹터 매핑 (optional)
        rebalance_dates: 리밸런싱 날짜 (optional)
    """
    # 월간 수익률 변환
    monthly_returns = portfolio_returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )

    # Timing
    if trades_df is not None and price_df is not None:
        timing = calc_timing_skill(trades_df, price_df)
    else:
        # 매매 이력 없을 때: 가격 기반 Timing 추정
        timing = calc_timing_from_returns(
            portfolio_returns, benchmark_returns, weights, returns_by_ticker
        )

    # Diversification
    diversification = calc_diversification_skill(weights, sector_map)

    # Risk Management
    risk_mgmt = calc_risk_management_skill(portfolio_returns, benchmark_returns)

    # Conviction
    conviction = calc_conviction_skill(weights, returns_by_ticker)

    # Adaptability
    adaptability = calc_adaptability_skill(
        portfolio_returns, benchmark_returns, rebalance_dates
    )

    # Consistency
    consistency = calc_consistency_skill(monthly_returns)

    return SkillsProfile(
        timing=timing,
        diversification=diversification,
        risk_management=risk_mgmt,
        conviction=conviction,
        adaptability=adaptability,
        consistency=consistency,
    )
