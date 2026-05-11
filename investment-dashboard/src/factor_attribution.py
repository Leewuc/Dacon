"""
Factor Attribution Engine: Fama-French 팩터 모델 기반 수익률 분해

포트폴리오 수익률을 다음 요소로 분해:
    - Market (Beta): 시장 전체 움직임에 의한 수익
    - SMB (Size): 소형주 팩터 (Small Minus Big)
    - HML (Value): 가치주 팩터 (High Minus Low)
    - Alpha: 위 팩터로 설명되지 않는 순수 초과수익

Kenneth French Data Library에서 무료 팩터 데이터를 가져오거나,
오프라인 시 합성 팩터 데이터를 생성한다.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from visualizations import COLORS

# Kenneth French Data Library URL
FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


@dataclass
class FactorResult:
    """팩터 분석 결과"""
    alpha: float           # 연환산 알파 (일간 alpha * 252)
    alpha_daily: float     # 일간 알파
    beta_market: float     # 시장 베타
    beta_smb: float        # SMB 베타
    beta_hml: float        # HML 베타
    beta_wml: float = 0.0  # Momentum (WML) 베타
    r_squared: float = 0.0       # 결정계수
    adj_r_squared: float = 0.0   # 수정 결정계수

    # 수익률 분해 (비율, signed)
    pct_market: float = 0.0      # 시장으로 설명되는 비율
    pct_smb: float = 0.0         # SMB로 설명되는 비율
    pct_hml: float = 0.0         # HML로 설명되는 비율
    pct_wml: float = 0.0         # WML로 설명되는 비율
    pct_alpha: float = 0.0       # 알파 비율

    # 통계
    t_stat_alpha: float = 0.0
    p_value_alpha: float = 0.0
    residual_std: float = 0.0

    # 시계열
    factor_contributions: Optional[pd.DataFrame] = None


# =============================================================================
# 팩터 데이터 수집
# =============================================================================

def fetch_ff3_factors(
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Kenneth French 3-Factor 일간 데이터 다운로드

    Returns:
        DataFrame with columns: ['Mkt-RF', 'SMB', 'HML', 'RF']
        모든 값은 소수점 비율 (퍼센트 → 소수)
    """
    try:
        import io, zipfile, urllib.request

        response = urllib.request.urlopen(FF3_URL, timeout=10)
        zip_data = io.BytesIO(response.read())

        with zipfile.ZipFile(zip_data) as zf:
            csv_name = [f for f in zf.namelist() if f.endswith('.CSV')][0]
            with zf.open(csv_name) as f:
                lines = f.read().decode('utf-8').splitlines()

        # 헤더 찾기
        data_start = None
        for i, line in enumerate(lines):
            if 'Mkt-RF' in line:
                data_start = i
                break

        if data_start is None:
            return None

        # 데이터 파싱
        records = []
        for line in lines[data_start + 1:]:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            try:
                date_str = parts[0].strip()
                if len(date_str) != 8:
                    continue
                date = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
                mkt_rf = float(parts[1]) / 100
                smb = float(parts[2]) / 100
                hml = float(parts[3]) / 100
                rf = float(parts[4]) / 100
                records.append({
                    'Date': date, 'Mkt-RF': mkt_rf, 'SMB': smb,
                    'HML': hml, 'RF': rf,
                })
            except (ValueError, IndexError):
                continue

        df = pd.DataFrame(records).set_index('Date').sort_index()

        # 기간 필터
        mask = df.index >= pd.Timestamp(start_date)
        if end_date:
            mask &= df.index <= pd.Timestamp(end_date)

        return df[mask]

    except Exception as e:
        print(f"FF3 데이터 다운로드 실패: {e}")
        return None


def generate_synthetic_factors(
    dates: pd.DatetimeIndex,
    seed: int = 123,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    오프라인용 합성 팩터 데이터 생성
    실제 Fama-French 팩터의 통계적 특성을 모방

    Args:
        dates: DatetimeIndex for factor data
        seed: Random seed for reproducibility
        benchmark_returns: Optional benchmark returns for momentum factor generation
    """
    np.random.seed(seed)
    n = len(dates)

    # 실제 FF3 팩터의 대략적 통계 (일간)
    # Mkt-RF: mean ~0.04%, std ~1.0%
    # SMB: mean ~0.01%, std ~0.5%
    # HML: mean ~0.01%, std ~0.5%
    mkt_rf = np.random.normal(0.0004, 0.010, n)
    smb = np.random.normal(0.0001, 0.005, n)
    hml = np.random.normal(0.0001, 0.005, n)
    rf = np.full(n, 0.035 / 252)  # 연 3.5% → 일간

    # 간헐적 큰 움직임 (fat tails)
    crash_days = np.random.choice(n, size=max(1, int(n * 0.02)), replace=False)
    mkt_rf[crash_days] -= np.random.uniform(0.02, 0.04, len(crash_days))

    # Generate Momentum (WML) factor
    if benchmark_returns is not None and len(benchmark_returns) > 252:
        # WML = long-term momentum - short-term momentum
        momentum = (benchmark_returns.rolling(252).mean() -
                   benchmark_returns.rolling(21).mean())
        # Normalize to match scale of other factors
        momentum = momentum.fillna(0)
        momentum_std = momentum.std()
        if momentum_std > 1e-10:
            momentum = momentum / momentum_std * 0.005  # Scale to ~0.5% std
    else:
        # Synthetic momentum factor
        momentum = np.random.normal(0.0001, 0.005, n)

    return pd.DataFrame({
        'Mkt-RF': mkt_rf,
        'SMB': smb,
        'HML': hml,
        'WML': momentum,
        'RF': rf,
    }, index=dates)


# =============================================================================
# 팩터 회귀분석
# =============================================================================

def run_factor_regression(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
) -> FactorResult:
    """
    OLS 회귀: R_p - R_f = alpha + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML + beta_wml*WML + e

    Supports both 3-factor (Fama-French) and 4-factor (Carhart) models.
    Uses condition number check to detect ill-conditioned matrices.

    numpy만으로 구현 (scikit-learn 의존성 제거)
    """
    # 인덱스 정렬
    common = portfolio_returns.index.intersection(factor_data.index)
    if len(common) < 30:
        raise ValueError(f"공통 데이터 부족: {len(common)}일 (최소 30일 필요)")

    y = portfolio_returns.reindex(common).values - factor_data['RF'].reindex(common).values

    # Determine if 4-factor model is available
    factor_cols = ['Mkt-RF', 'SMB', 'HML']
    if 'WML' in factor_data.columns:
        factor_cols.append('WML')
        n_factors = 4
    else:
        n_factors = 3

    X_raw = factor_data[factor_cols].reindex(common).values

    # 상수항 추가 (intercept = alpha)
    n_obs = len(y)
    X = np.column_stack([np.ones(n_obs), X_raw])

    # OLS: beta = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y

    # Check condition number
    cond_number = np.linalg.cond(XtX)
    if cond_number > 1e10:
        print(f"Warning: High condition number ({cond_number:.2e}), using pseudo-inverse")
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XtX) @ Xty
    else:
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

    alpha_daily = beta[0]
    beta_market = beta[1]
    beta_smb = beta[2]
    beta_hml = beta[3]
    beta_wml = beta[4] if n_factors == 4 else 0.0

    # 잔차 및 R²
    y_hat = X @ beta
    residuals = y - y_hat
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n_params = n_factors + 1
    adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - n_params) if n_obs > n_params else r_squared

    # t-통계량 (alpha)
    residual_std = np.sqrt(ss_res / (n_obs - n_params)) if n_obs > n_params else 0
    if residual_std > 0:
        try:
            inv_XtX = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            inv_XtX = np.linalg.pinv(XtX)
        se_beta = residual_std * np.sqrt(np.diag(inv_XtX))
        t_stat_alpha = alpha_daily / se_beta[0] if se_beta[0] > 0 else 0
    else:
        t_stat_alpha = 0

    # p-value 근사 (정규분포)
    p_value_alpha = 2 * (1 - _norm_cdf(abs(t_stat_alpha)))

    # 수익률 분해 (기여도) - using SIGNED values
    factor_means = factor_data[factor_cols].reindex(common).mean()
    contrib_market = beta_market * factor_means['Mkt-RF'] * 252
    contrib_smb = beta_smb * factor_means['SMB'] * 252
    contrib_hml = beta_hml * factor_means['HML'] * 252
    contrib_wml = beta_wml * factor_means.get('WML', 0) * 252 if n_factors == 4 else 0
    contrib_alpha = alpha_daily * 252

    # Total with signed values
    total_explained = contrib_market + contrib_smb + contrib_hml + contrib_wml + contrib_alpha

    # Percentages using SIGNED contributions
    if abs(total_explained) > 1e-10:
        pct_market = contrib_market / total_explained * 100
        pct_smb = contrib_smb / total_explained * 100
        pct_hml = contrib_hml / total_explained * 100
        pct_wml = contrib_wml / total_explained * 100 if n_factors == 4 else 0
        pct_alpha = contrib_alpha / total_explained * 100
    else:
        pct_market = pct_smb = pct_hml = pct_wml = pct_alpha = 0.0

    # 팩터 기여 시계열
    dates = factor_data.reindex(common).index
    contrib_dict = {
        'Market': beta_market * factor_data['Mkt-RF'].reindex(common).values,
        'SMB': beta_smb * factor_data['SMB'].reindex(common).values,
        'HML': beta_hml * factor_data['HML'].reindex(common).values,
        'Alpha': np.full(n_obs, alpha_daily),
        'Residual': residuals,
    }
    if n_factors == 4:
        contrib_dict['WML'] = beta_wml * factor_data['WML'].reindex(common).values

    contributions = pd.DataFrame(contrib_dict, index=dates)

    return FactorResult(
        alpha=alpha_daily * 252,
        alpha_daily=alpha_daily,
        beta_market=beta_market,
        beta_smb=beta_smb,
        beta_hml=beta_hml,
        beta_wml=beta_wml,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        pct_market=pct_market,
        pct_smb=pct_smb,
        pct_hml=pct_hml,
        pct_wml=pct_wml,
        pct_alpha=pct_alpha,
        t_stat_alpha=t_stat_alpha,
        p_value_alpha=p_value_alpha,
        residual_std=residual_std * np.sqrt(252),  # 연환산
        factor_contributions=contributions,
    )


def _norm_cdf(x: float) -> float:
    """표준정규분포 CDF 근사 (Abramowitz & Stegun)"""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


# =============================================================================
# 통합 실행
# =============================================================================

def run_factor_analysis(
    portfolio_returns: pd.Series,
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    use_synthetic: bool = False,
    benchmark_returns: Optional[pd.Series] = None,
) -> FactorResult:
    """
    팩터 분석 전체 실행

    1. FF3/4-Factor 데이터 수집 (또는 합성)
    2. 회귀분석
    3. 결과 반환

    Args:
        portfolio_returns: Portfolio returns time series
        start_date: Start date for factor data
        end_date: End date for factor data
        use_synthetic: Force synthetic factor generation
        benchmark_returns: Optional benchmark returns for momentum factor generation
    """
    if use_synthetic:
        factors = generate_synthetic_factors(portfolio_returns.index, benchmark_returns=benchmark_returns)
    else:
        factors = fetch_ff3_factors(start_date, end_date)
        if factors is None or factors.empty:
            print("FF3 데이터 수집 실패, 합성 데이터 사용 (4-Factor model)")
            factors = generate_synthetic_factors(portfolio_returns.index, benchmark_returns=benchmark_returns)
        else:
            # Add momentum factor if benchmark returns are available
            if benchmark_returns is not None and len(benchmark_returns) > 252:
                momentum = (benchmark_returns.rolling(252).mean() -
                           benchmark_returns.rolling(21).mean())
                momentum = momentum.fillna(0)
                momentum_std = momentum.std()
                if momentum_std > 1e-10:
                    momentum = momentum / momentum_std * 0.005
                factors['WML'] = momentum.reindex(factors.index, fill_value=0)

    return run_factor_regression(portfolio_returns, factors)


# =============================================================================
# 시각화
# =============================================================================

def create_factor_donut(result: FactorResult) -> go.Figure:
    """팩터별 수익률 기여 도넛 차트"""
    labels = ["Market (Beta)", "Size (SMB)", "Value (HML)", "Alpha"]
    values = [result.pct_market, result.pct_smb, result.pct_hml, result.pct_alpha]
    colors_list = ["#6366F1", "#8B5CF6", "#EC4899", "#10B981"]

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors_list, line=dict(color=COLORS["background"], width=2)),
        textinfo="label+percent",
        texttemplate="%{label}<br>%{percent:.1%}",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="<b>%{label}</b><br>Contribution: %{percent:.1%}<extra></extra>",
        pull=[0, 0, 0, 0.05],  # Alpha를 약간 분리
    ))

    # 중앙 텍스트
    alpha_str = f"{result.alpha:+.2f}%"
    fig.add_annotation(
        text=f"<b>Alpha</b><br>{alpha_str}",
        x=0.5, y=0.5,
        font=dict(size=16, color=COLORS["positive"] if result.alpha >= 0 else COLORS["negative"]),
        showarrow=False,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        title=dict(text="Return Attribution (Fama-French 3-Factor)", x=0.5, font=dict(size=18)),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )

    return fig


def create_factor_exposure_bar(result: FactorResult) -> go.Figure:
    """팩터 노출도(베타) 수평 바 차트"""
    factors = ["Market (Beta)", "Size (SMB)", "Value (HML)"]
    betas = [result.beta_market, result.beta_smb, result.beta_hml]

    colors_list = [
        COLORS["primary"] if b >= 0 else COLORS["negative"]
        for b in betas
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=factors,
        x=betas,
        orientation="h",
        marker=dict(color=colors_list, line=dict(width=0)),
        text=[f"{b:.3f}" for b in betas],
        textposition="outside",
        textfont=dict(size=13, color=COLORS["text"]),
        hovertemplate="<b>%{y}</b><br>Beta: %{x:.4f}<extra></extra>",
    ))

    # Beta = 1.0 참조선
    fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text="Beta=1.0")
    fig.add_vline(x=0, line_color=COLORS["grid"], line_width=0.5)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        title=dict(text="Factor Exposures (Betas)", x=0.5, font=dict(size=18)),
        xaxis=dict(title="Beta Coefficient", gridcolor=COLORS["grid"]),
        yaxis=dict(autorange="reversed"),
        height=250,
        showlegend=False,
        margin=dict(l=40, r=60, t=60, b=40),
    )

    return fig


def create_cumulative_attribution(result: FactorResult) -> go.Figure:
    """팩터별 누적 기여 수익률 stacked area chart"""
    if result.factor_contributions is None:
        fig = go.Figure()
        fig.add_annotation(text="Factor contribution data not available", showarrow=False)
        return fig

    contrib = result.factor_contributions.copy()

    # 누적
    cum_market = (contrib['Market']).cumsum() * 100
    cum_smb = (contrib['SMB']).cumsum() * 100
    cum_hml = (contrib['HML']).cumsum() * 100
    cum_alpha = (contrib['Alpha']).cumsum() * 100
    cum_total = (cum_market + cum_smb + cum_hml + cum_alpha)

    fig = go.Figure()

    traces = [
        ("Market", cum_market, "#6366F1"),
        ("Size (SMB)", cum_smb, "#8B5CF6"),
        ("Value (HML)", cum_hml, "#EC4899"),
        ("Alpha", cum_alpha, "#10B981"),
    ]

    for name, data, color in traces:
        fig.add_trace(go.Scatter(
            x=data.index, y=data.values,
            name=name,
            line=dict(color=color, width=2),
            stackgroup='one',
            hovertemplate=f"{name}: " + "%{y:.2f}%<extra></extra>",
        ))

    # Total 수익률 선
    fig.add_trace(go.Scatter(
        x=cum_total.index, y=cum_total.values,
        name="Total",
        line=dict(color=COLORS["text"], width=2, dash="dot"),
        hovertemplate="Total: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        title=dict(text="Cumulative Factor Attribution", x=0.5, font=dict(size=18)),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(title="Cumulative Contribution (%)", gridcolor=COLORS["grid"], ticksuffix="%"),
        legend=dict(x=0.02, y=0.98),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
    )

    return fig


def create_alpha_significance_gauge(result: FactorResult) -> go.Figure:
    """알파의 통계적 유의성을 게이지로 표현"""
    # t-stat 기반 유의성 (|t| > 2.0이면 유의)
    significance = min(100, abs(result.t_stat_alpha) / 3.0 * 100)

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=result.alpha * 100,  # 퍼센트로 표시될 수 있게
        number=dict(
            suffix="%",
            font=dict(size=28, color=COLORS["positive"] if result.alpha >= 0 else COLORS["negative"]),
        ),
        title=dict(text="Annualized Alpha", font=dict(size=14, color=COLORS["text"])),
        delta=dict(
            reference=0,
            increasing=dict(color=COLORS["positive"]),
            decreasing=dict(color=COLORS["negative"]),
        ),
        gauge=dict(
            axis=dict(range=[-20, 20], tickcolor=COLORS["text"], ticksuffix="%"),
            bar=dict(color=COLORS["primary"]),
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["grid"],
            steps=[
                dict(range=[-20, -5], color="rgba(239, 68, 68, 0.2)"),
                dict(range=[-5, 5], color="rgba(107, 114, 128, 0.2)"),
                dict(range=[5, 20], color="rgba(16, 185, 129, 0.2)"),
            ],
            threshold=dict(
                line=dict(color=COLORS["text"], width=2),
                thickness=0.8,
                value=result.alpha * 100,
            ),
        ),
    ))

    # 유의성 주석
    sig_text = "Significant" if abs(result.t_stat_alpha) > 2.0 else "Not Significant"
    sig_color = COLORS["positive"] if abs(result.t_stat_alpha) > 2.0 else COLORS["neutral"]

    fig.add_annotation(
        text=f"t-stat: {result.t_stat_alpha:.2f} ({sig_text})",
        x=0.5, y=-0.15,
        font=dict(size=11, color=sig_color),
        showarrow=False,
        xref="paper", yref="paper",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        height=280,
        margin=dict(l=30, r=30, t=40, b=40),
    )

    return fig
