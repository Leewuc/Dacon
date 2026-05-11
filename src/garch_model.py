"""
GARCH(1,1) 변동성 예측 모듈
━━━━━━━━━━━━━━━━━━━━━━━━━━━
순수 NumPy/SciPy 구현 (arch 패키지 불필요)

GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
- ω (omega): 장기 분산에 대한 가중치
- α (alpha): 직전 충격(뉴스)에 대한 반응도
- β (beta):  이전 변동성의 지속도
- α + β < 1 (안정성 조건)

사용처:
- Tail Risk: 동적 VaR 계산
- Monte Carlo: 변동성 클러스터링 반영 시뮬레이션
- Stress Test: 미래 변동성 시나리오
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# 데이터 클래스
# =============================================================================

@dataclass
class GarchResult:
    """GARCH(1,1) 피팅 결과"""
    omega: float          # 상수항
    alpha: float          # ARCH 파라미터 (충격 반응)
    beta: float           # GARCH 파라미터 (변동성 지속)
    persistence: float    # α + β (1에 가까울수록 변동성 지속)
    long_run_var: float   # 장기 분산 = ω / (1 - α - β)
    long_run_vol: float   # 장기 변동성 (연환산)
    conditional_vol: np.ndarray   # 조건부 변동성 시계열
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int

    # 예측
    forecast_vol_1d: float   # 1일 후 예측 변동성
    forecast_vol_5d: float   # 5일 후 예측 변동성
    forecast_vol_20d: float  # 20일(1개월) 후 예측 변동성

    # 현재 vs 장기
    current_vol: float       # 현재 조건부 변동성 (연환산)
    vol_regime: str          # "높음" / "보통" / "낮음"


# =============================================================================
# GARCH(1,1) 피팅
# =============================================================================

def _garch_variance(params: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """GARCH(1,1) 조건부 분산 계산"""
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)  # 초기값 = 표본 분산

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    return sigma2


def _neg_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    """음의 로그우도 (최소화 대상)"""
    omega, alpha, beta = params

    # 안정성 조건
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10

    sigma2 = _garch_variance(params, returns)

    # σ² ≤ 0 방지
    sigma2 = np.maximum(sigma2, 1e-12)

    # 가우시안 로그우도
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns ** 2 / sigma2)
    return -ll  # 최소화이므로 부호 반전


def fit_garch(returns: pd.Series, max_iter: int = 500) -> Optional[GarchResult]:
    """
    GARCH(1,1) 모델 피팅

    Parameters:
        returns: 일간 수익률 시계열
        max_iter: 최적화 최대 반복 횟수

    Returns:
        GarchResult 또는 None (데이터 부족 시)
    """
    r = returns.dropna().values
    n = len(r)

    if n < 50:
        return None

    # 초기 파라미터 추정
    var_r = np.var(r)

    if HAS_SCIPY:
        # SciPy 최적화
        bounds = [(1e-8, var_r * 10), (0.01, 0.5), (0.3, 0.99)]
        result = scipy_minimize(
            _neg_log_likelihood,
            x0=[var_r * 0.05, 0.08, 0.85],
            args=(r,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter},
        )
        if not result.success:
            result = scipy_minimize(
                _neg_log_likelihood,
                x0=[var_r * 0.05, 0.08, 0.85],
                args=(r,),
                method="Nelder-Mead",
                options={"maxiter": max_iter * 2},
            )
        omega, alpha, beta = result.x
    else:
        # SciPy 없을 때: Grid Search + 미세 탐색
        best_nll = 1e10
        omega, alpha, beta = var_r * 0.05, 0.08, 0.85
        for a in np.arange(0.02, 0.35, 0.03):
            for b in np.arange(0.50, 0.97, 0.03):
                if a + b >= 0.99:
                    continue
                o = var_r * (1 - a - b) * 0.8
                nll = _neg_log_likelihood(np.array([o, a, b]), r)
                if nll < best_nll:
                    best_nll = nll
                    omega, alpha, beta = o, a, b
        # 미세 탐색
        for da in np.arange(-0.02, 0.025, 0.005):
            for db in np.arange(-0.02, 0.025, 0.005):
                a2, b2 = alpha + da, beta + db
                if a2 <= 0 or b2 <= 0 or a2 + b2 >= 0.99:
                    continue
                o2 = var_r * (1 - a2 - b2) * 0.8
                nll = _neg_log_likelihood(np.array([o2, a2, b2]), r)
                if nll < best_nll:
                    best_nll = nll
                    omega, alpha, beta = o2, a2, b2
    persistence = alpha + beta

    # 안정성 보정
    if persistence >= 1.0:
        scale = 0.99 / persistence
        alpha *= scale
        beta *= scale
        persistence = alpha + beta

    # 조건부 분산 계산
    sigma2 = _garch_variance(np.array([omega, alpha, beta]), r)
    cond_vol = np.sqrt(sigma2)

    # 장기 분산
    long_run_var = omega / (1 - persistence) if persistence < 1 else var_r
    long_run_vol = np.sqrt(long_run_var) * np.sqrt(252)

    # 현재 변동성
    current_vol = cond_vol[-1] * np.sqrt(252)

    # 변동성 예측 (multi-step)
    last_sigma2 = sigma2[-1]
    last_eps2 = r[-1] ** 2

    # h-step 예측: σ²_{t+h} = ω·Σ(α+β)^i + (α+β)^h · σ²_t
    def _forecast_var(h):
        if persistence >= 1:
            return last_sigma2
        return long_run_var + (persistence ** h) * (last_sigma2 - long_run_var)

    forecast_1d = np.sqrt(_forecast_var(1)) * np.sqrt(252)
    forecast_5d = np.sqrt(_forecast_var(5)) * np.sqrt(252)
    forecast_20d = np.sqrt(_forecast_var(20)) * np.sqrt(252)

    # 변동성 레짐 판단
    hist_vol_median = np.median(cond_vol) * np.sqrt(252)
    hist_vol_75 = np.percentile(cond_vol, 75) * np.sqrt(252)
    if current_vol > hist_vol_75:
        vol_regime = "높음"
    elif current_vol > hist_vol_median:
        vol_regime = "보통"
    else:
        vol_regime = "낮음"

    # 정보 기준
    k = 3  # 파라미터 수
    neg_ll = _neg_log_likelihood(np.array([omega, alpha, beta]), r)
    ll = -neg_ll
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return GarchResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=persistence,
        long_run_var=long_run_var,
        long_run_vol=long_run_vol,
        conditional_vol=pd.Series(cond_vol * np.sqrt(252), index=returns.dropna().index),
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        n_obs=n,
        forecast_vol_1d=forecast_1d,
        forecast_vol_5d=forecast_5d,
        forecast_vol_20d=forecast_20d,
        current_vol=current_vol,
        vol_regime=vol_regime,
    )


# =============================================================================
# GARCH 기반 Monte Carlo 시뮬레이션
# =============================================================================

def garch_monte_carlo(
    returns: pd.Series,
    garch_result: GarchResult,
    n_days: int = 60,
    n_sims: int = 1000,
    initial_value: float = 10000,
) -> pd.DataFrame:
    """
    GARCH 변동성을 반영한 Monte Carlo 시뮬레이션

    기존 Monte Carlo는 변동성이 일정하다고 가정하지만,
    이 함수는 변동성 클러스터링을 반영합니다.
    """
    r = returns.dropna().values
    omega = garch_result.omega
    alpha = garch_result.alpha
    beta = garch_result.beta
    mu = np.mean(r)

    # 마지막 상태
    last_sigma2 = (garch_result.conditional_vol.iloc[-1] / np.sqrt(252)) ** 2
    last_eps = r[-1]

    paths = np.zeros((n_sims, n_days))

    for i in range(n_sims):
        sigma2_t = last_sigma2
        price = initial_value
        for t in range(n_days):
            z = np.random.standard_normal()
            eps = np.sqrt(sigma2_t) * z
            ret = mu + eps
            paths[i, t] = price * (1 + ret)
            price = paths[i, t]
            sigma2_t = omega + alpha * eps ** 2 + beta * sigma2_t

    return pd.DataFrame(paths)


# =============================================================================
# 시각화
# =============================================================================

def create_garch_vol_chart(
    garch_result: GarchResult,
    returns: pd.Series,
) -> go.Figure:
    """조건부 변동성 vs 실현 변동성 비교 차트"""
    cond_vol = garch_result.conditional_vol
    # 20일 실현 변동성
    realized_vol = returns.rolling(20).std() * np.sqrt(252)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cond_vol.index, y=cond_vol.values,
        name="GARCH 조건부 변동성",
        line=dict(color="#6366F1", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=realized_vol.index, y=realized_vol.values,
        name="20일 실현 변동성",
        line=dict(color="#94A3B8", width=1, dash="dash"),
    ))

    # 장기 변동성 수평선
    fig.add_hline(
        y=garch_result.long_run_vol,
        line_dash="dot", line_color="#F59E0B",
        annotation_text=f"장기 변동성: {garch_result.long_run_vol:.1%}",
    )

    fig.update_layout(
        title="GARCH(1,1) 조건부 변동성",
        yaxis_title="연환산 변동성",
        yaxis_tickformat=".0%",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=400,
        legend=dict(orientation="h", y=1.1),
    )

    return fig


def create_vol_forecast_chart(garch_result: GarchResult) -> go.Figure:
    """변동성 예측 퍼널 차트"""
    horizons = ["현재", "1일 후", "5일 후", "20일 후", "장기"]
    vols = [
        garch_result.current_vol,
        garch_result.forecast_vol_1d,
        garch_result.forecast_vol_5d,
        garch_result.forecast_vol_20d,
        garch_result.long_run_vol,
    ]

    colors = []
    for v in vols:
        if v > garch_result.long_run_vol * 1.3:
            colors.append("#EF4444")
        elif v > garch_result.long_run_vol * 0.8:
            colors.append("#F59E0B")
        else:
            colors.append("#10B981")

    fig = go.Figure(go.Bar(
        x=horizons, y=vols,
        marker_color=colors,
        text=[f"{v:.1%}" for v in vols],
        textposition="outside",
    ))

    fig.update_layout(
        title="변동성 예측 (GARCH)",
        yaxis_title="연환산 변동성",
        yaxis_tickformat=".0%",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=350,
    )

    return fig


def create_garch_params_chart(garch_result: GarchResult) -> go.Figure:
    """GARCH 파라미터 도넛 차트 (α, β, 1-α-β)"""
    remainder = max(0, 1 - garch_result.alpha - garch_result.beta)
    labels = [
        f"α (충격 반응): {garch_result.alpha:.3f}",
        f"β (변동성 지속): {garch_result.beta:.3f}",
        f"잔여: {remainder:.3f}",
    ]
    values = [garch_result.alpha, garch_result.beta, remainder]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=["#EF4444", "#6366F1", "#334155"]),
    ))

    fig.update_layout(
        title="GARCH 파라미터 구성",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=350,
        annotations=[dict(
            text=f"지속성<br>{garch_result.persistence:.3f}",
            x=0.5, y=0.5, font_size=14, showarrow=False,
        )],
    )

    return fig
