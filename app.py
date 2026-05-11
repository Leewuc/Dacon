"""
Investment Skills Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━
투자 데이터를 시각화하는 Skills 기반 대시보드

실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline import (
    load_portfolio_data,
    generate_sample_portfolio,
    generate_sample_sector_map,
    parse_portfolio_csv,
    PortfolioData,
)
from skills_engine import (
    generate_skills_profile,
    SkillsProfile,
    calc_sharpe_ratio,
    calc_sortino_ratio,
    calc_max_drawdown,
    calc_calmar_ratio,
)
from visualizations import (
    create_skills_radar,
    create_cumulative_returns,
    create_drawdown_chart,
    create_sector_treemap,
    create_monthly_heatmap,
    create_risk_return_scatter,
    create_monte_carlo_chart,
    create_skills_detail_bars,
    COLORS,
)
from whatif_engine import (
    generate_equal_weight,
    generate_concentrated,
    generate_defensive,
    generate_momentum,
    run_scenario,
    create_comparison_radar,
    create_change_waterfall,
    create_weight_comparison_bar,
)
from factor_attribution import (
    run_factor_analysis,
    create_factor_donut,
    create_factor_exposure_bar,
    create_cumulative_attribution,
    create_alpha_significance_gauge,
)
from kr_stocks import (
    resolve_kr_ticker,
    is_korean_ticker,
    build_kr_portfolio,
    get_popular_kr_portfolios,
    search_kr_stock,
    KR_BENCHMARKS,
    ALL_SEARCHABLE,
    parse_stock_selection,
)
from ai_commentary import generate_commentary, Commentary
from efficient_frontier import (
    compute_efficient_frontier,
    create_frontier_scatter,
    create_optimal_weights_bar,
    create_frontier_summary_table,
)
from stress_test import (
    SCENARIOS,
    run_stress_test,
    run_all_stress_tests,
    create_stress_comparison_bar,
    create_stress_path_chart,
    create_sector_impact_treemap,
    create_recovery_timeline,
)
from geopolitical_engine import (
    GEOPOLITICAL_SCENARIOS,
    MACRO_VARIABLES,
    GeopoliticalScenario,
    propagate_shocks,
    run_geopolitical_scenario,
    create_macro_impact_waterfall,
    create_sector_impact_bar as create_geo_sector_bar,
    create_geopolitical_path,
    create_scenario_comparison_radar,
    create_impact_heatmap,
)
from correlation_network import (
    compute_correlation_network,
    create_network_graph,
    create_correlation_heatmap,
    create_cluster_summary,
)
from multi_portfolio import (
    COMPARISON_PORTFOLIOS,
    compute_portfolio_summary,
    create_multi_radar,
    create_performance_comparison_bar as create_perf_comparison_bar,
    create_comparison_table,
    create_risk_return_scatter as create_multi_risk_return,
)
from risk_contribution import (
    compute_risk_contribution,
    create_risk_contribution_bar,
    create_risk_donut,
    create_risk_budget_table,
    get_risk_summary,
    identify_risk_outliers,
)
from style_analysis import (
    analyze_portfolio_style,
    analyze_style_drift,
    create_style_box,
    create_style_radar,
    create_style_timeline,
    create_style_comparison,
    format_style_result,
)
from market_events import (
    get_events_in_range,
    create_annotated_returns_chart,
    create_event_impact_table,
)
from regime_detection import (
    detect_regimes,
    analyze_regime_performance,
    get_current_regime,
    get_regime_summary,
    create_regime_timeline,
    create_regime_performance_bars,
    create_regime_transition_matrix,
    create_regime_duration_chart,
)
from tail_risk import (
    cornish_fisher_var,
    multi_confidence_var,
    tail_risk_analysis,
    create_var_comparison_chart,
    create_return_distribution,
    create_tail_qq_plot,
    create_rolling_var_chart,
)
from garch_model import (
    fit_garch,
    garch_monte_carlo,
    create_garch_vol_chart,
    create_vol_forecast_chart,
    create_garch_params_chart,
)
from black_litterman import (
    black_litterman_analysis,
    create_bl_comparison_chart,
    create_bl_weights_chart,
    create_bl_impact_chart,
)
from performance_metrics import (
    calc_all_metrics,
    create_capture_ratio_chart,
    create_performance_comparison_chart,
)
from rebalance_signal import (
    generate_rebalance_signal,
    create_signal_gauge,
    get_signal_interpretation,
)
from portfolio_dna import (
    generate_dna,
    create_dna_fingerprint,
    get_dna_archetype,
)
from backtest_engine import (
    run_backtest,
    run_rolling_backtests,
    create_backtest_cumulative_chart,
    create_rolling_performance_chart,
    create_backtest_monthly_heatmap,
)

# =============================================================================
# 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="Investment Skills Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 커스텀 CSS
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background-color: #0F172A;
    }

    /* 사이드바 */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
    }

    /* 메트릭 카드 */
    [data-testid="stMetric"] {
        background-color: #1E293B;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #334155;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 8px 16px;
        color: #94A3B8;
    }

    .stTabs [aria-selected="true"] {
        background-color: #6366F1;
        color: white;
    }

    /* 구분선 */
    hr {
        border-color: #334155;
    }

    /* 스킬 뱃지 */
    .skill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 2px;
    }
    .grade-S { background: #10B981; color: white; }
    .grade-A { background: #6366F1; color: white; }
    .grade-B { background: #F59E0B; color: black; }
    .grade-C { background: #F97316; color: white; }
    .grade-D { background: #EF4444; color: white; }

    /* ── 텍스트 잘림(ellipsis) 방지 ── */

    /* 메트릭 라벨: 줄바꿈 허용, 잘리지 않게 */
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        min-width: 0;
    }
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] label {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: keep-all;
        overflow-wrap: break-word;
    }

    /* 메트릭 값: 줄바꿈 허용 */
    [data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-all;
    }

    /* 메트릭 delta 텍스트 */
    [data-testid="stMetricDelta"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    /* 데이터프레임 셀 */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th,
    .stDataFrame td, .stDataFrame th {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        max-width: none !important;
    }

    /* expander 내부 텍스트 */
    [data-testid="stExpander"] div {
        white-space: normal !important;
        overflow-wrap: break-word;
    }

    /* 탭 라벨 */
    .stTabs [data-baseweb="tab"] button {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    /* 일반 마크다운 텍스트 */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        white-space: normal !important;
        overflow-wrap: break-word;
        word-break: keep-all;
    }

    /* 컬럼 내 콘텐츠 줄바꿈 */
    [data-testid="column"] {
        overflow: visible !important;
        min-width: 0;
    }

    /* 카테고리 라디오 버튼 스타일 */
    div[data-testid="stHorizontalBlock"]:has([data-testid="stWidgetLabel"]) {
        gap: 0 !important;
    }
    div.row-widget.stRadio > div[role="radiogroup"] {
        gap: 6px;
        flex-wrap: wrap;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.15s;
        font-size: 0.92rem;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        border-color: #6366F1;
        background-color: #283548;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #6366F1;
        border-color: #6366F1;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 사이드바: 데이터 입력
# =============================================================================

with st.sidebar:
    st.markdown("## 🎯 Investment Skills Dashboard")
    st.markdown("---")

    # 데이터 소스 선택
    data_source = st.radio(
        "📂 데이터 소스",
        ["샘플 포트폴리오", "직접 입력", "CSV 업로드", "합성 데이터 (오프라인)", "🇰🇷 한국 주식 포트폴리오"],
        index=0,
    )

    st.markdown("---")

    # 기간 설정
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작일",
            value=datetime.now() - timedelta(days=365*2),
            max_value=datetime.now(),
        )
    with col2:
        end_date = st.date_input(
            "종료일",
            value=datetime.now(),
            max_value=datetime.now(),
        )

    # 벤치마크
    benchmark_options = ["SPY", "QQQ", "IWM", "EFA", "VT", "KOSPI (^KS11)", "KOSDAQ (^KQ11)"]
    benchmark_raw = st.selectbox(
        "📊 벤치마크",
        benchmark_options,
        index=0,
        help="포트폴리오 성과를 비교할 벤치마크",
    )
    # Parse benchmark ticker
    if "KOSPI" in benchmark_raw:
        benchmark = "^KS11"
    elif "KOSDAQ" in benchmark_raw:
        benchmark = "^KQ11"
    else:
        benchmark = benchmark_raw

    st.markdown("---")

    # 데이터 소스별 입력
    weights = None
    sector_map = None
    use_synthetic = False

    if data_source == "샘플 포트폴리오":
        weights = generate_sample_portfolio()
        sector_map = generate_sample_sector_map()
        st.markdown("### 📋 샘플 포트폴리오")
        for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
            st.markdown(f"**{ticker}**: {w*100:.1f}%")

    elif data_source == "직접 입력":
        st.markdown("### ✏️ 포트폴리오 입력")
        st.caption("종목명을 검색하거나 직접 티커를 입력하세요")
        n_holdings = st.number_input("종목 수", 2, 20, 5)

        weights = {}
        for i in range(n_holdings):
            col_t, col_w = st.columns([2, 1])
            with col_t:
                selection = st.selectbox(
                    f"종목 {i+1}",
                    options=ALL_SEARCHABLE,
                    index=0,
                    key=f"stock_search_{i}",
                    placeholder="종목명 검색...",
                )
                # "직접 입력" 선택 시 수동 텍스트 입력
                if selection and selection.startswith("직접 입력"):
                    manual_ticker = st.text_input(
                        "티커 직접 입력",
                        key=f"manual_ticker_{i}",
                        placeholder="예: AAPL, 005930.KS",
                    )
                    if manual_ticker:
                        resolved_ticker, display_name, sector = resolve_kr_ticker(manual_ticker)
                        selection = None  # flag to use manual
                    else:
                        resolved_ticker = None
                else:
                    resolved_ticker, display_name, sector = parse_stock_selection(selection)
                    manual_ticker = None

            with col_w:
                weight = st.number_input(f"비중(%)", 1, 100, 20, key=f"weight_{i}")

            if resolved_ticker:
                weights[resolved_ticker] = weight / 100

    elif data_source == "CSV 업로드":
        st.markdown("### 📁 CSV 업로드")
        st.markdown("""
        지원 형식:
        - `ticker, weight`
        - `ticker, shares, price`
        - `ticker, amount`
        """)
        uploaded = st.file_uploader("CSV 파일", type=["csv"])
        if uploaded:
            try:
                weights = parse_portfolio_csv(uploaded)
                st.success(f"{len(weights)}개 종목 로드 완료!")
                for t, w in weights.items():
                    st.markdown(f"**{t}**: {w*100:.1f}%")
            except Exception as e:
                st.error(f"CSV 파싱 오류: {e}")

    elif data_source == "합성 데이터 (오프라인)":
        use_synthetic = True
        st.info("인터넷 없이 합성 데이터로 데모합니다.")

    elif data_source == "🇰🇷 한국 주식 포트폴리오":
        kr_portfolios = get_popular_kr_portfolios()
        kr_choice = st.selectbox(
            "🇰🇷 한국 포트폴리오 선택",
            list(kr_portfolios.keys()),
        )
        kr_raw = kr_portfolios[kr_choice]
        weights_result, sector_map_result, display_names = build_kr_portfolio(kr_raw)
        weights = weights_result
        sector_map = sector_map_result
        st.session_state["kr_display_names"] = display_names

        st.markdown(f"### 📋 {kr_choice}")
        for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
            display = display_names.get(ticker, ticker)
            st.markdown(f"**{display}** ({ticker}): {w*100:.1f}%")

    st.markdown("---")

    # 시뮬레이션 설정
    st.markdown("### 🎲 Monte Carlo 설정")
    n_simulations = st.slider("시뮬레이션 횟수", 100, 2000, 500, 100)
    simulation_days = st.slider("예측 기간 (거래일)", 63, 504, 252, 21)

    # 실행 버튼
    run_analysis = st.button("🚀 분석 실행", type="primary", use_container_width=True)


# =============================================================================
# 메인 영역
# =============================================================================

# 헤더
st.markdown("""
# 🎯 InvestScope — Investment Skills Dashboard
**17가지 분석 엔진으로 투자 역량을 진단하고, 데이터 기반 인사이트를 발견하세요.**
""")

if not run_analysis and "portfolio_data" not in st.session_state:
    # Enhanced Landing Page
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0;">🎯</h1>
        <h2 style="color: #A5B4FC; margin-top: 8px;">Welcome to InvestScope v6.0</h2>
        <p style="color: #94A3B8; font-size: 1.1rem; max-width: 650px; margin: 0 auto;">
            6-Skills 프레임워크를 핵심으로, 23개 분석 탭과 AI 진단 엔진이<br>
            포트폴리오의 강점·약점·리스크·기회를 입체적으로 진단합니다.
        </p>
        <div style="margin-top: 16px;">
            <span style="background: #6366F1; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: white; margin: 4px;">🇰🇷 한국 112종목</span>
            <span style="background: #334155; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #F8FAFC; margin: 4px;">🌍 글로벌 100종목</span>
            <span style="background: #334155; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #F8FAFC; margin: 4px;">📊 17 분석 탭</span>
            <span style="background: #334155; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #F8FAFC; margin: 4px;">🔑 API 키 불필요</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Core Analysis ──
    st.markdown("<h3 style='color: #A5B4FC; margin-bottom: 4px;'>🧠 Core Analysis</h3>", unsafe_allow_html=True)
    core_features = [
        ("🎯", "6-Skills Radar", "Timing · Diversification · Risk Management · Conviction · Adaptability · Consistency — 핵심 투자 역량을 레이더로 시각화"),
        ("🤖", "AI 투자 진단", "룰 기반 NLG 엔진이 강점/약점/개선 제안을 자동 생성하고 PDF 리포트로 다운로드"),
        ("📊", "Performance 분석", "누적수익률, 드로다운, 월별 히트맵 + 시장 이벤트 어노테이션 (39개 글로벌 이벤트)"),
    ]

    cols = st.columns(3)
    for j, (icon, title, desc) in enumerate(core_features):
        with cols[j]:
            st.markdown(f"""
            <div style="background: #1E293B; padding: 20px; border-radius: 12px;
                        border: 1px solid #6366F1; min-height: 150px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
                <h4 style="color: #F8FAFC; margin: 0 0 8px 0;">{title}</h4>
                <p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Risk & Factor ──
    st.markdown("<h3 style='color: #F59E0B; margin-bottom: 4px;'>⚡ Risk & Factor 분석</h3>", unsafe_allow_html=True)
    risk_features = [
        ("🧬", "Carhart 4-Factor", "Market · SMB · HML · Momentum으로 수익원천을 분해하고 알파 유의성을 검증"),
        ("⚡", "Risk Contribution", "Euler 분해로 종목별 MRC/CRC를 산출하고 비중 대비 리스크 이상치를 탐지"),
        ("🎲", "Tail Risk (CF-VaR)", "왜도·첨도 보정 Cornish-Fisher VaR + Q-Q Plot + Jarque-Bera 정규성 검정"),
        ("📐", "Efficient Frontier", "Ledoit-Wolf 축소 추정량 기반 Markowitz 최적화로 최적 비중을 계산"),
    ]

    cols = st.columns(4)
    for j, (icon, title, desc) in enumerate(risk_features):
        with cols[j]:
            st.markdown(f"""
            <div style="background: #1E293B; padding: 16px; border-radius: 12px;
                        border: 1px solid #334155; min-height: 160px;">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">{icon}</div>
                <h4 style="color: #F8FAFC; margin: 0 0 8px 0; font-size: 0.95rem;">{title}</h4>
                <p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Scenario & Simulation ──
    st.markdown("<h3 style='color: #10B981; margin-bottom: 4px;'>🌍 시나리오 & 시뮬레이션</h3>", unsafe_allow_html=True)
    scenario_features = [
        ("🔮", "Monte Carlo", "확률적 미래 포트폴리오 가치 예측 + VaR/CVaR 지표"),
        ("🔄", "What-If 시나리오", "4가지 전략(동일비중/집중/방어/모멘텀)으로 Skills 변화 시뮬레이션"),
        ("🔥", "Stress Test", "COVID, 2008 GFC, 금리인상 등 역사적 위기 시뮬레이션"),
        ("🌍", "Geopolitical", "호르무즈 봉쇄, 대만 위기 등 매크로 충격 → 섹터 전파 시뮬레이션"),
    ]

    cols = st.columns(4)
    for j, (icon, title, desc) in enumerate(scenario_features):
        with cols[j]:
            st.markdown(f"""
            <div style="background: #1E293B; padding: 16px; border-radius: 12px;
                        border: 1px solid #334155; min-height: 150px;">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">{icon}</div>
                <h4 style="color: #F8FAFC; margin: 0 0 8px 0; font-size: 0.95rem;">{title}</h4>
                <p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Advanced Insight ──
    st.markdown("<h3 style='color: #EC4899; margin-bottom: 4px;'>🔬 Advanced Insight</h3>", unsafe_allow_html=True)
    insight_features = [
        ("📈", "Skills Evolution", "롤링 윈도우로 투자 역량의 시간적 변화를 추적하고 등급 밴드를 오버레이"),
        ("🎨", "Investment Style", "Morningstar 3×3 Style Box + 6축 스타일 레이더 + Style Drift 추적"),
        ("🔀", "Regime Detection", "Bull/Bear/Sideways 시장 국면 자동 탐지 + 국면별 성과 + 전이 확률"),
        ("🕸️", "Correlation Network", "종목 간 상관관계를 네트워크 그래프로 시각화하고 클러스터를 자동 탐지"),
        ("⚖️", "Multi-Portfolio", "S&P 500, 글로벌 분산, 성장주 등 벤치마크 포트폴리오와 비교 분석"),
        ("📅", "Market Events", "2019~2025 글로벌 39개 이벤트의 포트폴리오 영향도를 어노테이션 차트로 시각화"),
    ]

    for i in range(0, len(insight_features), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(insight_features):
                icon, title, desc = insight_features[i + j]
                with col:
                    st.markdown(f"""
                    <div style="background: #1E293B; padding: 16px; border-radius: 12px;
                                border: 1px solid #334155; min-height: 140px;">
                        <div style="font-size: 1.8rem; margin-bottom: 8px;">{icon}</div>
                        <h4 style="color: #F8FAFC; margin: 0 0 8px 0; font-size: 0.95rem;">{title}</h4>
                        <p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("")

    # ── 검색 지원 안내 ──
    st.markdown("---")
    st.markdown("""
    <div style="background: #1E293B; padding: 20px; border-radius: 12px; border: 1px solid #334155;">
        <h4 style="color: #6366F1; margin-top: 0;">🔍 스마트 종목 검색</h4>
        <p style="color: #94A3B8; font-size: 0.9rem;">
            한글 · 영문 · 약칭 · 띄어쓰기 모두 자동 인식합니다.<br>
            <span style="color: #F8FAFC;">삼성전자, samsung, 삼성 전자, 005930</span> → 모두 동일한 종목으로 인식<br>
            <span style="color: #F8FAFC;">애플, apple, AAPL</span> → 모두 Apple로 인식<br>
            🇰🇷 한국 112종목 + 🌍 글로벌 100종목 내장, 그 외 종목도 티커 직접 입력 가능
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_start1, col_start2, col_start3 = st.columns([1, 2, 1])
    with col_start2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <p style="color: #94A3B8; font-size: 1rem;">
                👈 사이드바에서 포트폴리오를 설정하고<br>
                <span style="color: #6366F1; font-weight: 600; font-size: 1.2rem;">🚀 분석 실행</span>
                버튼을 클릭하세요.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# =============================================================================
# 데이터 로딩 & 분석
# =============================================================================

if run_analysis:
    with st.spinner("📡 데이터 수집 중..."):
        try:
            portfolio_data = load_portfolio_data(
                weights=weights,
                benchmark_ticker=benchmark,
                start_date=str(start_date),
                end_date=str(end_date),
                use_synthetic=use_synthetic,
            )
            st.session_state["portfolio_data"] = portfolio_data
        except Exception as e:
            st.error(f"데이터 로딩 실패: {e}")
            st.info("'합성 데이터 (오프라인)' 모드를 사용해보세요.")
            st.stop()

    with st.spinner("🧠 Skills 분석 중..."):
        profile = generate_skills_profile(
            portfolio_returns=portfolio_data.returns,
            benchmark_returns=portfolio_data.benchmark_returns,
            weights=portfolio_data.holdings,
            returns_by_ticker=portfolio_data.returns_by_ticker,
            sector_map=portfolio_data.sector_map,
        )
        st.session_state["profile"] = profile

portfolio_data: PortfolioData = st.session_state["portfolio_data"]
profile: SkillsProfile = st.session_state["profile"]

# =============================================================================
# 상단: KPI 카드
# =============================================================================

st.markdown("---")

# 스킬명 한국어 매핑
_SKILL_KR = {
    "Timing": "타이밍",
    "Diversification": "분산투자",
    "Risk Management": "리스크관리",
    "Conviction": "확신도",
    "Adaptability": "적응력",
    "Consistency": "일관성",
}

total_return = float((1 + portfolio_data.returns).prod() - 1) * 100
bench_return = float((1 + portfolio_data.benchmark_returns).prod() - 1) * 100
sharpe = calc_sharpe_ratio(portfolio_data.returns)
mdd = calc_max_drawdown(portfolio_data.returns) * 100
overall_skill = profile.overall_score()
best_skill = profile.strongest_skill()
worst_skill = profile.weakest_skill()
best_kr = _SKILL_KR.get(best_skill, best_skill)
worst_kr = _SKILL_KR.get(worst_skill, worst_skill)

kpi_row1 = st.columns(3)
with kpi_row1[0]:
    st.metric(label="📈 총 수익률", value=f"{total_return:.1f}%", delta=f"vs BM {bench_return:.1f}%")
with kpi_row1[1]:
    st.metric(label="⚡ Alpha", value=f"{total_return - bench_return:+.1f}%", delta="초과 수익")
with kpi_row1[2]:
    st.metric(label="📐 Sharpe Ratio", value=f"{sharpe:.2f}", delta="위험 조정 수익")

kpi_row2 = st.columns(3)
with kpi_row2[0]:
    st.metric(label="📉 Max Drawdown", value=f"{mdd:.1f}%", delta="최대 낙폭")
with kpi_row2[1]:
    st.metric(label="🎯 Overall Skill", value=f"{overall_skill:.0f}점", delta=f"등급: {profile.timing._calc_grade() if overall_skill >= 90 else ('A' if overall_skill >= 75 else ('B' if overall_skill >= 55 else 'C'))}")
with kpi_row2[2]:
    st.metric(label="💪 Best Skill", value=best_kr, delta=f"약점: {worst_kr}")


# =============================================================================
# 탭 레이아웃
# =============================================================================

st.markdown("---")

# =============================================================================
# 카테고리 기반 네비게이션 (23개 탭 → 6개 카테고리)
# =============================================================================

CATEGORIES = {
    "📊 기본 분석": {
        "tabs": ["🎯 Skills", "📊 Performance", "🗺️ Allocation", "🔮 Simulation"],
        "desc": "스킬 프로파일 · 성과 지표 · 자산 배분 · 몬테카를로 시뮬레이션",
    },
    "🔮 시나리오": {
        "tabs": ["🔄 What-If", "🔥 Stress Test", "🌍 Geopolitical"],
        "desc": "비중 변경 시뮬레이션 · 위기 시나리오 · 지정학 이벤트",
    },
    "📈 팩터 & 최적화": {
        "tabs": ["🧬 Factor", "🎨 Style", "📐 Frontier", "🏦 Black-Litterman"],
        "desc": "Carhart 4-Factor · 투자 스타일 · 효율적 프론티어 · BL 최적화",
    },
    "⚡ 리스크": {
        "tabs": ["⚡ Risk 기여도", "🎲 Tail Risk", "📉 GARCH", "🔀 Regime"],
        "desc": "리스크 분해 · 꼬리 리스크(CF-VaR) · GARCH 변동성 · 시장 국면",
    },
    "🔗 포트폴리오 진단": {
        "tabs": ["🕸️ Correlation", "⚖️ Compare", "🔔 Rebalance", "🧬 DNA"],
        "desc": "상관관계 네트워크 · 멀티 포트폴리오 비교 · 리밸런싱 시그널 · DNA 핑거프린트",
    },
    "📋 성과 추적": {
        "tabs": ["📈 Evolution", "📅 Events", "⏪ Backtest", "📋 Performance+"],
        "desc": "스킬 진화 · 시장 이벤트 · 백테스트 · IR/Tracking Error/Capture",
    },
}

# 카테고리 선택 (가로 라디오)
cat_names = list(CATEGORIES.keys())
selected_cat = st.radio(
    "분석 카테고리",
    cat_names,
    horizontal=True,
    key="nav_category",
    label_visibility="collapsed",
)

# 선택된 카테고리 설명
st.caption(CATEGORIES[selected_cat]["desc"])

# 카테고리별 탭 생성 (동적)
_cat_tabs = CATEGORIES[selected_cat]["tabs"]
_active_tabs = st.tabs(_cat_tabs)

# ---------- 카테고리 ↔ 탭 매핑 ----------
# 각 카테고리 내 탭을 딕셔너리로 관리
_tab_map = {}
for cat_name, cat_info in CATEGORIES.items():
    _tab_map[cat_name] = {}

# 현재 선택된 카테고리의 탭만 활성화
for i, tab_name in enumerate(_cat_tabs):
    _tab_map[selected_cat][tab_name] = _active_tabs[i]

# 편의 변수: 현재 카테고리가 아닌 탭은 None (렌더링 안 됨)
def _get_tab(cat, idx):
    """카테고리 이름과 탭 인덱스로 탭 컨텍스트 반환. 현재 카테고리가 아니면 None."""
    tabs = CATEGORIES[cat]["tabs"]
    if cat != selected_cat or idx >= len(tabs):
        return None
    return _tab_map.get(cat, {}).get(tabs[idx])

# 기존 tab1~tab23을 카테고리별로 매핑
tab1  = _get_tab("📊 기본 분석", 0)
tab2  = _get_tab("📊 기본 분석", 1)
tab3  = _get_tab("📊 기본 분석", 2)
tab4  = _get_tab("📊 기본 분석", 3)
tab5  = _get_tab("🔮 시나리오", 0)
tab6  = _get_tab("📈 팩터 & 최적화", 0)
tab7  = _get_tab("📋 성과 추적", 0)
tab8  = _get_tab("📈 팩터 & 최적화", 2)
tab9  = _get_tab("🔮 시나리오", 1)
tab10 = _get_tab("🔮 시나리오", 2)
tab11 = _get_tab("🔗 포트폴리오 진단", 0)
tab12 = _get_tab("🔗 포트폴리오 진단", 1)
tab13 = _get_tab("📈 팩터 & 최적화", 1)
tab14 = _get_tab("⚡ 리스크", 0)
tab15 = _get_tab("📋 성과 추적", 1)
tab16 = _get_tab("⚡ 리스크", 3)
tab17 = _get_tab("⚡ 리스크", 1)
tab18 = _get_tab("⚡ 리스크", 2)
tab19 = _get_tab("📈 팩터 & 최적화", 3)
tab20 = _get_tab("🔗 포트폴리오 진단", 2)
tab21 = _get_tab("🔗 포트폴리오 진단", 3)
tab22 = _get_tab("📋 성과 추적", 2)
tab23 = _get_tab("📋 성과 추적", 3)


# ---------- Tab 1: Skills Analysis ----------
if tab1 is not None:
  with tab1:
      col_radar, col_bars = st.columns([1, 1])

      with col_radar:
          fig_radar = create_skills_radar(profile.to_dict())
          st.plotly_chart(fig_radar, use_container_width=True)

      with col_bars:
          fig_bars = create_skills_detail_bars(profile.to_dict())
          st.plotly_chart(fig_bars, use_container_width=True)

      # Skills 상세 카드
      st.markdown("### 📋 Skills 상세 분석")

      skills_list = [
          profile.timing,
          profile.diversification,
          profile.risk_management,
          profile.conviction,
          profile.adaptability,
          profile.consistency,
      ]

      skill_cols = st.columns(3)
      for i, skill in enumerate(skills_list):
          with skill_cols[i % 3]:
              grade_class = f"grade-{skill.grade}"
              skill_kr = _SKILL_KR.get(skill.name, skill.name)
              st.markdown(f"""
              <div style="background: #1E293B; padding: 16px; border-radius: 12px;
                          border: 1px solid #334155; margin-bottom: 12px;">
                  <div style="display: flex; justify-content: space-between; align-items: center;">
                      <h4 style="margin:0; color: #F8FAFC;">{skill_kr} ({skill.name})</h4>
                      <span class="skill-badge {grade_class}">{skill.grade} ({skill.score:.0f})</span>
                  </div>
                  <p style="color: #94A3B8; font-size: 0.9rem; margin-top: 8px;">
                      {skill.description}
                  </p>
              </div>
              """, unsafe_allow_html=True)

              # 상세 지표
              if skill.detail:
                  with st.expander("상세 지표"):
                      for key, val in skill.detail.items():
                          if key != "note":
                              st.markdown(f"- **{key}**: {val}")

      # AI 코멘터리 섹션
      st.markdown("---")
      st.markdown("### 🤖 AI 투자 진단")

      # skills_dict와 skill_details 준비
      skills_dict = profile.to_dict()
      skill_details = {}
      for skill in skills_list:
          skill_details[skill.name] = skill.detail if skill.detail else {}

      total_ret = float((1 + portfolio_data.returns).prod() - 1)
      bench_ret = float((1 + portfolio_data.benchmark_returns).prod() - 1)

      commentary = generate_commentary(
          skills_dict=skills_dict,
          skill_details=skill_details,
          weights=portfolio_data.holdings,
          sector_map=portfolio_data.sector_map,
          total_return=total_ret,
          benchmark_return=bench_ret,
      )

      # Risk Alert
      if commentary.risk_alert:
          st.error(commentary.risk_alert)

      # Summary
      st.markdown(f"""
      <div style="background: linear-gradient(135deg, #1E293B 0%, #312E81 100%);
                  padding: 20px; border-radius: 12px; border: 1px solid #4F46E5;
                  margin-bottom: 16px;">
          <h4 style="color: #A5B4FC; margin-top: 0;">📌 종합 평가</h4>
          <p style="color: #E2E8F0; font-size: 1.05rem; line-height: 1.7;">
              {commentary.summary}
          </p>
      </div>
      """, unsafe_allow_html=True)

      # Strengths & Weaknesses
      col_str, col_weak = st.columns(2)
      with col_str:
          st.markdown("#### 💪 강점")
          for s in commentary.strengths:
              st.markdown(f"✅ {s}")
          if not commentary.strengths:
              st.markdown("_해당 없음_")

      with col_weak:
          st.markdown("#### 🔧 개선 영역")
          for w in commentary.weaknesses:
              st.markdown(f"⚠️ {w}")
          if not commentary.weaknesses:
              st.markdown("_해당 없음_")

      # Detailed Diagnosis
      with st.expander("📖 상세 진단 보기", expanded=False):
          st.markdown(commentary.diagnosis)

      # Recommendations
      if commentary.recommendations:
          with st.expander("💡 개선 제안", expanded=True):
              for i, rec in enumerate(commentary.recommendations, 1):
                  st.markdown(f"**{i}.** {rec}")

      # PDF 리포트 다운로드
      st.markdown("---")
      st.markdown("### 📄 리포트 다운로드")

      try:
          from io import BytesIO

          def generate_pdf_report():
              """Skills 분석 리포트를 PDF로 생성 (한글+영문 듀얼폰트)"""
              try:
                  from reportlab.lib.pagesizes import A4
                  from reportlab.lib.units import mm
                  from reportlab.pdfgen import canvas as rl_canvas
                  from reportlab.pdfbase import pdfmetrics
                  from reportlab.pdfbase.ttfonts import TTFont
                  import os

                  # ── 한국어 폰트 등록 ──
                  kr_font = None
                  # 프로젝트 번들 폰트를 최우선 사용
                  _bundled = os.path.join(os.path.dirname(__file__), "assets", "fonts", "DroidSansFallback.ttf")
                  kr_candidates = [
                      _bundled,
                      "/usr/share/fonts-droid-fallback/truetype/DroidSansFallback.ttf",
                      "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                  ]
                  # 한글+영문 모두 되는 폰트 (있으면 듀얼폰트 불필요)
                  full_candidates = [
                      "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                      "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                      "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                      "C:/Windows/Fonts/malgun.ttf",
                  ]
                  use_single_font = False
                  for fp in full_candidates:
                      try:
                          if os.path.exists(fp):
                              pdfmetrics.registerFont(TTFont("FullCJK", fp))
                              kr_font = "FullCJK"
                              use_single_font = True
                              break
                      except Exception:
                          continue
                  if not use_single_font:
                      for fp in kr_candidates:
                          try:
                              if os.path.exists(fp):
                                  pdfmetrics.registerFont(TTFont("KR", fp))
                                  kr_font = "KR"
                                  break
                          except Exception:
                              continue

                  EN_FONT = "Helvetica"
                  EN_BOLD = "Helvetica-Bold"

                  def _is_cjk(ch):
                      cp = ord(ch)
                      return (0xAC00 <= cp <= 0xD7A3 or 0x3131 <= cp <= 0x318E or
                              0x4E00 <= cp <= 0x9FFF or 0x2600 <= cp <= 0x27BF or
                              0x25A0 <= cp <= 0x25FF or cp in (0x2192, 0x2022))

                  def _draw(c, x, y, text, size, bold=False):
                      """한글/영문 혼합 텍스트 렌더링"""
                      if not text:
                          return x
                      en = EN_BOLD if bold else EN_FONT
                      if use_single_font:
                          c.setFont(kr_font, size)
                          c.drawString(x, y, text)
                          return x + pdfmetrics.stringWidth(text, kr_font, size)
                      if not kr_font:
                          c.setFont(en, size)
                          c.drawString(x, y, text)
                          return x + pdfmetrics.stringWidth(text, en, size)
                      # 세그먼트 분리
                      segs, cur, cur_cjk = [], text[0], _is_cjk(text[0])
                      for ch in text[1:]:
                          ch_cjk = _is_cjk(ch)
                          if ch_cjk == cur_cjk:
                              cur += ch
                          else:
                              segs.append((cur, cur_cjk))
                              cur, cur_cjk = ch, ch_cjk
                      segs.append((cur, cur_cjk))
                      cx = x
                      for seg, is_cjk in segs:
                          f = kr_font if is_cjk else en
                          c.setFont(f, size)
                          c.drawString(cx, y, seg)
                          cx += pdfmetrics.stringWidth(seg, f, size)
                      return cx

                  def _draw_wrapped(c, x, y, text, size, max_w, line_h, bold=False):
                      """긴 텍스트 자동 줄바꿈"""
                      if not text:
                          return y
                      en = EN_BOLD if bold else EN_FONT
                      words = []
                      cur_word = ""
                      for ch in text:
                          if ch == ' ':
                              if cur_word:
                                  words.append(cur_word)
                              words.append(' ')
                              cur_word = ""
                          else:
                              cur_word += ch
                      if cur_word:
                          words.append(cur_word)

                      line = ""
                      for w in words:
                          test = line + w
                          tw = 0
                          for ch in test:
                              f = (kr_font if _is_cjk(ch) else en) if kr_font and not use_single_font else (kr_font or en)
                              tw += pdfmetrics.stringWidth(ch, f, size)
                          if tw > max_w and line:
                              _draw(c, x, y, line, size, bold)
                              y -= line_h
                              if y < 40:
                                  c.showPage()
                                  y = A4[1] - 40
                              line = w.lstrip()
                          else:
                              line = test
                      if line:
                          _draw(c, x, y, line, size, bold)
                          y -= line_h
                      return y

                  # ── PDF 빌드 ──
                  buffer = BytesIO()
                  c = rl_canvas.Canvas(buffer, pagesize=A4)
                  W, H = A4
                  LM, RM = 50, 50
                  MAX_W = W - LM - RM
                  y = H - 50

                  # 타이틀
                  _draw(c, LM, y, "Investment Skills Dashboard Report", 20, bold=True)
                  y -= 18
                  _draw(c, LM, y, f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 9)
                  y -= 10
                  c.setStrokeColorRGB(0.8, 0.8, 0.85)
                  c.line(LM, y, W - RM, y)
                  y -= 22

                  # 1. 성과 요약
                  c.setFillColorRGB(0.39, 0.4, 0.95)
                  _draw(c, LM, y, "1. 성과 요약 (Performance Summary)", 14, bold=True)
                  y -= 20
                  c.setFillColorRGB(0, 0, 0)
                  _draw(c, LM + 5, y, f"총 수익률: {total_return:.1f}%  |  벤치마크: {bench_return:.1f}%  |  Alpha: {total_return - bench_return:+.1f}%", 10)
                  y -= 15
                  _draw(c, LM + 5, y, f"Sharpe Ratio: {sharpe:.2f}  |  Max Drawdown: {mdd:.1f}%  |  종합 스킬: {overall_skill:.0f}점", 10)
                  y -= 15
                  _draw(c, LM + 5, y, f"Best: {best_kr}({profile.strongest_skill()})  |  Weakest: {worst_kr}({profile.weakest_skill()})", 10)
                  y -= 25

                  # 2. Skills 분석
                  c.setFillColorRGB(0.39, 0.4, 0.95)
                  _draw(c, LM, y, "2. 6-Skills 분석 결과", 14, bold=True)
                  y -= 18
                  c.setFillColorRGB(0, 0, 0)

                  _sk_kr = {"Timing": "타이밍", "Diversification": "분산투자",
                            "Risk Management": "리스크관리", "Conviction": "확신도",
                            "Adaptability": "적응력", "Consistency": "일관성"}
                  for skill in skills_list:
                      kr = _sk_kr.get(skill.name, skill.name)
                      _draw(c, LM + 8, y, f"{kr} ({skill.name}): {skill.score:.0f}점 [{skill.grade}] - {skill.description}", 10)
                      y -= 15

                  # 3. 상세 지표
                  y -= 12
                  c.setFillColorRGB(0.39, 0.4, 0.95)
                  _draw(c, LM, y, "3. 스킬 상세 지표", 14, bold=True)
                  y -= 16
                  c.setFillColorRGB(0, 0, 0)

                  for skill in skills_list:
                      if not skill.detail:
                          continue
                      kr = _sk_kr.get(skill.name, skill.name)
                      _draw(c, LM + 5, y, f"{kr} ({skill.name}) - {skill.grade} ({skill.score:.0f}점)", 10, bold=True)
                      y -= 14
                      for key, val in skill.detail.items():
                          if key in ("note", "분석 방법", "method"):
                              continue
                          _draw(c, LM + 15, y, f"  {key}: {val}", 9)
                          y -= 13
                          if y < 50:
                              c.showPage()
                              y = H - 50
                      y -= 5

                  # 4. AI 투자 진단
                  if y < 200:
                      c.showPage()
                      y = H - 50
                  y -= 8
                  c.setFillColorRGB(0.39, 0.4, 0.95)
                  _draw(c, LM, y, "4. AI 투자 진단", 14, bold=True)
                  y -= 18
                  c.setFillColorRGB(0, 0, 0)

                  y = _draw_wrapped(c, LM + 5, y, commentary.summary, 10, MAX_W - 10, 14)
                  y -= 6

                  if commentary.strengths:
                      _draw(c, LM + 5, y, "강점:", 10, bold=True)
                      y -= 14
                      for s in commentary.strengths:
                          y = _draw_wrapped(c, LM + 15, y, f"- {s}", 10, MAX_W - 25, 14)
                      y -= 4

                  if commentary.weaknesses:
                      _draw(c, LM + 5, y, "개선 영역:", 10, bold=True)
                      y -= 14
                      for w in commentary.weaknesses:
                          y = _draw_wrapped(c, LM + 15, y, f"- {w}", 10, MAX_W - 25, 14)
                      y -= 4

                  if commentary.recommendations:
                      _draw(c, LM + 5, y, "추천 액션:", 10, bold=True)
                      y -= 14
                      for rec in commentary.recommendations:
                          y = _draw_wrapped(c, LM + 15, y, f"- {rec}", 10, MAX_W - 25, 14)
                      y -= 4

                  if hasattr(commentary, 'diagnosis') and commentary.diagnosis:
                      y -= 4
                      _draw(c, LM + 5, y, "상세 진단:", 10, bold=True)
                      y -= 14
                      for line in commentary.diagnosis.split("\n"):
                          line = line.strip()
                          if line:
                              y = _draw_wrapped(c, LM + 10, y, line, 9, MAX_W - 15, 13)
                              if y < 50:
                                  c.showPage()
                                  y = H - 50

                  # 5. 고급 분석 요약 (Advanced Analysis Summary)
                  c.showPage()
                  y = H - 50
                  c.setFillColorRGB(0.39, 0.4, 0.95)
                  _draw(c, LM, y, "5. 고급 분석 요약 (Advanced Analysis)", 14, bold=True)
                  y -= 22
                  c.setFillColorRGB(0, 0, 0)

                  # 5a. GARCH 변동성 진단
                  try:
                      from src.garch_model import fit_garch
                      garch_res = fit_garch(portfolio_data.returns.dropna())
                      _draw(c, LM + 5, y, "GARCH(1,1) 변동성 진단", 11, bold=True)
                      y -= 16
                      _draw(c, LM + 15, y, f"α={garch_res.alpha:.4f}  β={garch_res.beta:.4f}  지속성={garch_res.alpha+garch_res.beta:.4f}", 9)
                      y -= 14
                      _draw(c, LM + 15, y, f"현재 변동성 국면: {garch_res.vol_regime}  |  장기균형 변동성: {garch_res.long_run_vol*100:.1f}%", 9)
                      y -= 14
                      _draw(c, LM + 15, y, f"1일 예측: {garch_res.forecast_vol_1d*100:.2f}%  |  5일: {garch_res.forecast_vol_5d*100:.2f}%  |  20일: {garch_res.forecast_vol_20d*100:.2f}%", 9)
                      y -= 20
                  except Exception:
                      pass

                  # 5b. Backtest 핵심 결과
                  try:
                      from src.backtest_engine import run_backtest
                      bt_result = run_backtest(portfolio_data.prices, dict(zip(tickers, weights)), portfolio_data.benchmark_prices)
                      _draw(c, LM + 5, y, "Backtest 핵심 결과", 11, bold=True)
                      y -= 16
                      _draw(c, LM + 15, y, f"총수익률: {bt_result.total_return*100:.1f}%  |  연환산: {bt_result.annualized_return*100:.1f}%  |  Sharpe: {bt_result.sharpe_ratio:.2f}", 9)
                      y -= 14
                      _draw(c, LM + 15, y, f"MDD: {bt_result.max_drawdown*100:.1f}%  |  승률: {bt_result.win_rate*100:.0f}%  |  Alpha: {bt_result.alpha*100:+.1f}%", 9)
                      y -= 20
                  except Exception:
                      pass

                  # 5c. Performance+ 지표
                  try:
                      from src.performance_metrics import calc_all_metrics
                      perf = calc_all_metrics(portfolio_data.returns, portfolio_data.benchmark_returns)
                      _draw(c, LM + 5, y, "Performance+ 고급 성과 지표", 11, bold=True)
                      y -= 16
                      _draw(c, LM + 15, y, f"Information Ratio: {perf.information_ratio:.3f}  |  Tracking Error: {perf.tracking_error*100:.1f}%", 9)
                      y -= 14
                      _draw(c, LM + 15, y, f"Up Capture: {perf.up_capture_ratio*100:.0f}%  |  Down Capture: {perf.down_capture_ratio*100:.0f}%  |  Spread: {(perf.up_capture_ratio-perf.down_capture_ratio)*100:+.0f}%p", 9)
                      y -= 14
                      _draw(c, LM + 15, y, f"Beta: {perf.beta:.2f}  |  Correlation: {perf.correlation:.2f}", 9)
                      y -= 20
                  except Exception:
                      pass

                  # 5d. Portfolio DNA
                  try:
                      from src.portfolio_dna import generate_dna, get_dna_archetype
                      _skills_dict = {s.name: s.score for s in skills_list}
                      _weights_s = pd.Series(dict(zip(tickers, weights)))
                      dna = generate_dna(_skills_dict, {}, {}, _weights_s, garch_res.vol_regime if 'garch_res' in dir() else 'MEDIUM')
                      archetype = get_dna_archetype(dna)
                      _draw(c, LM + 5, y, "Portfolio DNA 핑거프린트", 11, bold=True)
                      y -= 16
                      _draw(c, LM + 15, y, f"투자자 아키타입: {archetype}", 9)
                      y -= 14
                      _draw(c, LM + 15, y, f"DNA 해시: {dna.dna_hash}", 9)
                      y -= 20
                  except Exception:
                      pass

                  # 5e. Rebalance Signal
                  try:
                      from src.rebalance_signal import generate_rebalance_signal
                      _skills_dict_rb = {s.name: s.score for s in skills_list}
                      rb_signal = generate_rebalance_signal(
                          _skills_dict_rb, {},
                          portfolio_data.returns, portfolio_data.benchmark_returns,
                          pd.Series(dict(zip(tickers, weights)))
                      )
                      _draw(c, LM + 5, y, "리밸런싱 시그널", 11, bold=True)
                      y -= 16
                      _draw(c, LM + 15, y, f"긴급도: {rb_signal.urgency:.0f}/100  |  방향: {rb_signal.direction}", 9)
                      y -= 14
                      if rb_signal.reasons:
                          _draw(c, LM + 15, y, f"주요 사유: {rb_signal.reasons[0]}", 9)
                          y -= 14
                      y -= 6
                  except Exception:
                      pass

                  # 면책 조항
                  y -= 10
                  c.setStrokeColorRGB(0.8, 0.8, 0.85)
                  c.line(LM, y, W - RM, y)
                  y -= 14
                  c.setFillColorRGB(0.5, 0.5, 0.5)
                  y = _draw_wrapped(c, LM, y, "본 분석은 교육/연구 목적이며, 투자 조언을 구성하지 않습니다. 모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.", 8, MAX_W, 11)

                  # Footer
                  y -= 20
                  c.setStrokeColorRGB(0.8, 0.8, 0.85)
                  c.line(LM, y, W - RM, y)
                  y -= 14
                  c.setFillColorRGB(0.6, 0.6, 0.6)
                  _draw(c, LM, y, "Investment Skills Dashboard v6.0 | 23-Tab Comprehensive Analysis", 8)

                  c.save()
                  buffer.seek(0)
                  return buffer.getvalue()

              except ImportError:
                  return None

          pdf_bytes = generate_pdf_report()
          if pdf_bytes:
              st.download_button(
                  label="📄 PDF 리포트 다운로드",
                  data=pdf_bytes,
                  file_name=f"investment_skills_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                  mime="application/pdf",
                  use_container_width=True,
              )
          else:
              st.info("PDF 생성에 reportlab 패키지가 필요합니다. `pip install reportlab`")
      except Exception as e:
          st.warning(f"PDF 리포트 생성 준비 중 오류: {e}")


# ---------- Tab 2: Performance ----------
if tab2 is not None:
  with tab2:
      # 누적 수익률
      fig_cum = create_cumulative_returns(
          portfolio_data.returns,
          portfolio_data.benchmark_returns,
          portfolio_name=portfolio_data.name,
          benchmark_name=benchmark,
      )
      st.plotly_chart(fig_cum, use_container_width=True)

      # 드로다운
      fig_dd = create_drawdown_chart(portfolio_data.returns)
      st.plotly_chart(fig_dd, use_container_width=True)

      # 월별 히트맵
      fig_heatmap = create_monthly_heatmap(portfolio_data.returns)
      st.plotly_chart(fig_heatmap, use_container_width=True)

      # 추가 통계
      st.markdown("### 📊 성과 통계")
      stat_cols = st.columns(4)

      sortino = calc_sortino_ratio(portfolio_data.returns)
      calmar = calc_calmar_ratio(portfolio_data.returns)
      vol = portfolio_data.returns.std() * np.sqrt(252) * 100
      win_rate = (portfolio_data.returns > 0).mean() * 100

      stats = [
          ("Sortino Ratio", f"{sortino:.2f}"),
          ("Calmar Ratio", f"{calmar:.2f}"),
          ("연환산 변동성", f"{vol:.1f}%"),
          ("일간 승률", f"{win_rate:.1f}%"),
      ]

      for col, (label, val) in zip(stat_cols, stats):
          with col:
              st.metric(label, val)


# ---------- Tab 3: Allocation ----------
if tab3 is not None:
  with tab3:
      col_tree, col_scatter = st.columns([1, 1])

      with col_tree:
          fig_tree = create_sector_treemap(
              portfolio_data.holdings,
              portfolio_data.sector_map,
              portfolio_data.returns_by_ticker,
          )
          st.plotly_chart(fig_tree, use_container_width=True)

      with col_scatter:
          fig_scatter = create_risk_return_scatter(
              portfolio_data.returns_by_ticker,
              portfolio_data.prices,
              portfolio_data.holdings,
          )
          st.plotly_chart(fig_scatter, use_container_width=True)

      # 종목별 상세
      st.markdown("### 📋 종목별 상세")

      holdings_data = []
      for ticker, weight in sorted(portfolio_data.holdings.items(), key=lambda x: -x[1]):
          ret = portfolio_data.returns_by_ticker.get(ticker, 0)
          sector = portfolio_data.sector_map.get(ticker, "Unknown")
          holdings_data.append({
              "종목": ticker,
              "비중": f"{weight*100:.1f}%",
              "수익률": f"{ret*100:+.1f}%",
              "섹터": sector,
              "기여도": f"{weight * ret * 100:+.2f}%",
          })

      st.dataframe(
          pd.DataFrame(holdings_data),
          use_container_width=True,
          hide_index=True,
      )


# ---------- Tab 4: Simulation ----------
if tab4 is not None:
  with tab4:
      st.markdown("""
      ### 🔮 Monte Carlo 시뮬레이션
      과거 수익률 분포를 기반으로 향후 포트폴리오 가치를 확률적으로 예측합니다.
      """)

      fig_mc = create_monte_carlo_chart(
          portfolio_data.returns,
          n_simulations=n_simulations,
          n_days=simulation_days,
      )
      st.plotly_chart(fig_mc, use_container_width=True)

      # 시뮬레이션 요약 통계
      st.markdown("### 📊 시뮬레이션 결과 요약")

      mu = portfolio_data.returns.mean()
      sigma = portfolio_data.returns.std()

      np.random.seed(42)
      final_values = []
      for _ in range(n_simulations):
          daily_rets = np.random.normal(mu, sigma, simulation_days)
          final_values.append(10000 * np.prod(1 + daily_rets))

      final_values = np.array(final_values)

      sim_cols = st.columns(5)
      percentiles_info = [
          ("5th (최악)", np.percentile(final_values, 5)),
          ("25th", np.percentile(final_values, 25)),
          ("Median", np.percentile(final_values, 50)),
          ("75th", np.percentile(final_values, 75)),
          ("95th (최선)", np.percentile(final_values, 95)),
      ]

      for col, (label, val) in zip(sim_cols, percentiles_info):
          with col:
              ret_pct = (val / 10000 - 1) * 100
              st.metric(label, f"${val:,.0f}", f"{ret_pct:+.1f}%")

      # VaR 분석
      st.markdown("### ⚠️ Value at Risk (VaR)")
      var_cols = st.columns(3)

      var_95 = np.percentile(final_values, 5)
      var_99 = np.percentile(final_values, 1)
      cvar_95 = final_values[final_values <= var_95].mean()

      with var_cols[0]:
          st.metric("95% VaR", f"${10000 - var_95:,.0f}", "최대 손실 (95% 신뢰)")
      with var_cols[1]:
          st.metric("99% VaR", f"${10000 - var_99:,.0f}", "최대 손실 (99% 신뢰)")
      with var_cols[2]:
          st.metric("95% CVaR (ES)", f"${10000 - cvar_95:,.0f}", "조건부 기대 손실")


# ---------- Tab 5: What-If Scenario ----------
if tab5 is not None:
  with tab5:
      st.markdown("""
      ### 🔄 What-If 시나리오
      포트폴리오 비중을 변경하면 Skills가 어떻게 달라지는지 실시간으로 시뮬레이션합니다.
      """)

      # 시나리오 선택
      scenario_mode = st.radio(
          "시나리오 모드",
          ["프리셋 전략", "직접 조절"],
          horizontal=True,
      )

      modified_weights = dict(portfolio_data.holdings)

      if scenario_mode == "프리셋 전략":
          preset = st.selectbox(
              "전략 선택",
              [
                  "동일비중 (Equal Weight)",
                  "집중투자 (Top 3 Concentrated)",
                  "방어형 (Low Volatility)",
                  "모멘텀 (Momentum Tilt)",
              ],
          )

          if preset == "동일비중 (Equal Weight)":
              modified_weights = generate_equal_weight(list(portfolio_data.holdings.keys()))
              st.info("모든 종목을 동일 비중으로 배분합니다.")
          elif preset == "집중투자 (Top 3 Concentrated)":
              modified_weights = generate_concentrated(
                  portfolio_data.holdings, portfolio_data.returns_by_ticker, top_n=3
              )
              st.info("수익률 상위 3개 종목에 70%를 집중합니다.")
          elif preset == "방어형 (Low Volatility)":
              modified_weights = generate_defensive(portfolio_data.holdings, portfolio_data.prices)
              st.info("변동성이 낮은 종목에 더 높은 비중을 부여합니다.")
          elif preset == "모멘텀 (Momentum Tilt)":
              modified_weights = generate_momentum(
                  portfolio_data.holdings, portfolio_data.returns_by_ticker
              )
              st.info("최근 수익률이 높은 종목에 비중을 편향합니다.")

      else:  # 직접 조절
          st.markdown("**슬라이더로 비중을 조절하세요** (자동 정규화됩니다)")
          slider_weights = {}
          tickers = sorted(portfolio_data.holdings.keys())

          for ticker in tickers:
              current = portfolio_data.holdings[ticker] * 100
              new_val = st.slider(
                  f"{ticker}",
                  min_value=0.0,
                  max_value=50.0,
                  value=float(round(current, 1)),
                  step=0.5,
                  key=f"whatif_{ticker}",
              )
              slider_weights[ticker] = new_val

          total = sum(slider_weights.values())
          if total > 0:
              modified_weights = {t: w / total for t, w in slider_weights.items()}
          else:
              modified_weights = generate_equal_weight(tickers)

      # 시나리오 실행
      scenario_result = run_scenario(
          modified_weights=modified_weights,
          original_weights=portfolio_data.holdings,
          original_profile=profile,
          portfolio_returns=portfolio_data.returns,
          benchmark_returns=portfolio_data.benchmark_returns,
          prices=portfolio_data.prices,
          sector_map=portfolio_data.sector_map,
      )

      # 결과 표시
      st.markdown("---")

      # KPI 변화
      change_cols = st.columns(4)
      with change_cols[0]:
          st.metric(
              "Overall Skill",
              f"{scenario_result.modified_profile.overall_score():.0f}",
              f"{scenario_result.overall_change():+.1f}",
          )
      with change_cols[1]:
          st.metric(
              "Modified Return",
              f"{scenario_result.modified_return*100:.1f}%",
              f"{(scenario_result.modified_return - scenario_result.original_return)*100:+.1f}%",
          )
      with change_cols[2]:
          _best = scenario_result.modified_profile.strongest_skill()
          st.metric("Best Skill", _SKILL_KR.get(_best, _best))
      with change_cols[3]:
          _weak = scenario_result.modified_profile.weakest_skill()
          st.metric("Weakest Skill", _SKILL_KR.get(_weak, _weak))

      # 비교 차트
      col_radar, col_waterfall = st.columns([1, 1])

      with col_radar:
          fig_compare = create_comparison_radar(
              profile.to_dict(),
              scenario_result.modified_profile.to_dict(),
          )
          st.plotly_chart(fig_compare, use_container_width=True)

      with col_waterfall:
          fig_waterfall = create_change_waterfall(scenario_result.skill_changes())
          st.plotly_chart(fig_waterfall, use_container_width=True)

          # 비중 비교
          fig_weight = create_weight_comparison_bar(
              scenario_result.original_weights,
              scenario_result.modified_weights,
          )
          st.plotly_chart(fig_weight, use_container_width=True)

      # AI 코멘터리 (rule-based)
      st.markdown("### 💡 AI Insight")
      changes = scenario_result.skill_changes()
      improved = {k: v for k, v in changes.items() if v > 2}
      degraded = {k: v for k, v in changes.items() if v < -2}

      if improved:
          imp_text = ", ".join([f"**{k}** (+{v:.0f})" for k, v in improved.items()])
          st.success(f"향상되는 Skills: {imp_text}")
      if degraded:
          deg_text = ", ".join([f"**{k}** ({v:.0f})" for k, v in degraded.items()])
          st.warning(f"하락하는 Skills: {deg_text}")
      if not improved and not degraded:
          st.info("이 시나리오에서는 Skills에 유의미한 변화가 없습니다.")

      overall = scenario_result.overall_change()
      if overall > 5:
          st.markdown(f"> 이 전략은 전반적인 투자 역량을 **{overall:+.1f}점** 향상시킵니다. 적용을 고려해보세요.")
      elif overall < -5:
          st.markdown(f"> 이 전략은 전반적인 투자 역량을 **{overall:+.1f}점** 하락시킵니다. 트레이드오프를 신중히 검토하세요.")


# ---------- Tab 6: Factor Attribution ----------
if tab6 is not None:
  with tab6:
      st.markdown("""
      ### 🧬 Fama-French Factor Attribution
      포트폴리오 수익률을 시장(Market), 사이즈(SMB), 가치(HML), 알파(Alpha) 팩터로 분해합니다.
      """)

      with st.spinner("🧬 팩터 분석 중..."):
          try:
              factor_result = run_factor_analysis(
                  portfolio_returns=portfolio_data.returns,
                  start_date=str(portfolio_data.start_date),
                  end_date=str(portfolio_data.end_date),
                  use_synthetic=use_synthetic if 'use_synthetic' in dir() else True,
              )
              st.session_state["factor_result"] = factor_result
          except Exception as e:
              st.error(f"팩터 분석 실패: {e}")
              st.info("합성 데이터 모드에서는 합성 팩터 데이터를 사용합니다.")
              st.stop()

      factor_result = st.session_state["factor_result"]

      # KPI 카드
      factor_kpi = st.columns(5)
      with factor_kpi[0]:
          alpha_color = "normal" if factor_result.alpha >= 0 else "inverse"
          st.metric("Alpha (연환산)", f"{factor_result.alpha*100:+.2f}%")
      with factor_kpi[1]:
          st.metric("Market Beta", f"{factor_result.beta_market:.3f}")
      with factor_kpi[2]:
          st.metric("SMB Beta", f"{factor_result.beta_smb:.3f}")
      with factor_kpi[3]:
          st.metric("HML Beta", f"{factor_result.beta_hml:.3f}")
      with factor_kpi[4]:
          st.metric("R-squared", f"{factor_result.r_squared:.3f}")

      st.markdown("---")

      # 차트 영역
      col_donut, col_gauge = st.columns([1, 1])

      with col_donut:
          fig_donut = create_factor_donut(factor_result)
          st.plotly_chart(fig_donut, use_container_width=True)

      with col_gauge:
          fig_gauge = create_alpha_significance_gauge(factor_result)
          st.plotly_chart(fig_gauge, use_container_width=True)

          fig_exposure = create_factor_exposure_bar(factor_result)
          st.plotly_chart(fig_exposure, use_container_width=True)

      # 누적 기여 차트
      fig_cum_attr = create_cumulative_attribution(factor_result)
      st.plotly_chart(fig_cum_attr, use_container_width=True)

      # 해석 가이드
      st.markdown("### 📖 Factor Attribution 해석 가이드")

      col_interp1, col_interp2 = st.columns(2)

      with col_interp1:
          st.markdown(f"""
          <div style="background: #1E293B; padding: 16px; border-radius: 12px; border: 1px solid #334155;">
              <h4 style="color: #F8FAFC; margin-top:0;">수익률 분해</h4>
              <p style="color: #94A3B8; font-size: 0.9rem;">
                  <b>Market ({factor_result.pct_market:.1f}%)</b>: 시장 전체 움직임에 의한 수익<br>
                  <b>Size ({factor_result.pct_smb:.1f}%)</b>: 소형주/대형주 팩터 기여<br>
                  <b>Value ({factor_result.pct_hml:.1f}%)</b>: 가치주/성장주 팩터 기여<br>
                  <b>Alpha ({factor_result.pct_alpha:.1f}%)</b>: 팩터로 설명되지 않는 순수 초과수익
              </p>
          </div>
          """, unsafe_allow_html=True)

      with col_interp2:
          sig = "통계적으로 유의" if abs(factor_result.t_stat_alpha) > 2.0 else "통계적으로 유의하지 않음"
          beta_desc = "공격적" if factor_result.beta_market > 1.0 else ("방어적" if factor_result.beta_market < 0.8 else "시장 추종")

          st.markdown(f"""
          <div style="background: #1E293B; padding: 16px; border-radius: 12px; border: 1px solid #334155;">
              <h4 style="color: #F8FAFC; margin-top:0;">진단 요약</h4>
              <p style="color: #94A3B8; font-size: 0.9rem;">
                  포트폴리오 성격: <b>{beta_desc}</b> (Beta={factor_result.beta_market:.2f})<br>
                  알파: {factor_result.alpha*100:+.2f}% ({sig}, t={factor_result.t_stat_alpha:.2f})<br>
                  모델 설명력: R²={factor_result.r_squared:.3f}<br>
                  잔차 변동성: {factor_result.residual_std*100:.1f}% (연환산)
              </p>
          </div>
          """, unsafe_allow_html=True)


# ---------- Tab 7: Skills Evolution ----------
if tab7 is not None:
  with tab7:
      st.markdown("""
      ### 📈 Skills Evolution
      롤링 윈도우 기반으로 투자 역량이 시간에 따라 어떻게 변화했는지 추적합니다.
      """)

      # 설정
      window_days = st.select_slider(
          "롤링 윈도우 크기 (거래일)",
          options=[63, 126, 189, 252],
          value=126,
          format_func=lambda x: f"{x}일 (~{x//21}개월)",
      )

      # 롤링 Skills 계산
      returns = portfolio_data.returns
      bench_returns = portfolio_data.benchmark_returns

      if len(returns) < window_days + 20:
          st.warning(f"데이터가 부족합니다. 최소 {window_days + 20}거래일의 데이터가 필요합니다.")
      else:
          with st.spinner("📈 Skills Evolution 계산 중..."):
              evolution_data = []
              step = max(1, window_days // 6)  # ~20 data points

              for end_idx in range(window_days, len(returns), step):
                  start_idx = end_idx - window_days
                  window_ret = returns.iloc[start_idx:end_idx]
                  window_bench = bench_returns.iloc[start_idx:end_idx]

                  try:
                      window_profile = generate_skills_profile(
                          portfolio_returns=window_ret,
                          benchmark_returns=window_bench,
                          weights=portfolio_data.holdings,
                          returns_by_ticker=portfolio_data.returns_by_ticker,
                          sector_map=portfolio_data.sector_map,
                      )

                      date_label = returns.index[end_idx]
                      if hasattr(date_label, 'strftime'):
                          date_str = date_label.strftime('%Y-%m-%d')
                      else:
                          date_str = str(date_label)

                      row = {"Date": date_str}
                      for skill_name, score in window_profile.to_dict().items():
                          row[skill_name] = score
                      row["Overall"] = window_profile.overall_score()
                      evolution_data.append(row)
                  except Exception:
                      continue

              if evolution_data:
                  import plotly.graph_objects as go

                  evo_df = pd.DataFrame(evolution_data)

                  # Line chart - Skills over time
                  skill_names = ["Timing", "Diversification", "Risk Management",
                                "Conviction", "Adaptability", "Consistency"]
                  skill_colors = {
                      "Timing": "#6366F1",
                      "Diversification": "#10B981",
                      "Risk Management": "#F59E0B",
                      "Conviction": "#EF4444",
                      "Adaptability": "#8B5CF6",
                      "Consistency": "#06B6D4",
                  }

                  fig_evo = go.Figure()

                  for skill in skill_names:
                      if skill in evo_df.columns:
                          fig_evo.add_trace(go.Scatter(
                              x=evo_df["Date"],
                              y=evo_df[skill],
                              name=skill,
                              mode="lines+markers",
                              line=dict(color=skill_colors.get(skill, "#FFFFFF"), width=2),
                              marker=dict(size=5),
                          ))

                  # Overall as dashed line
                  if "Overall" in evo_df.columns:
                      fig_evo.add_trace(go.Scatter(
                          x=evo_df["Date"],
                          y=evo_df["Overall"],
                          name="Overall",
                          mode="lines",
                          line=dict(color="#F8FAFC", width=3, dash="dash"),
                      ))

                  # Grade bands
                  fig_evo.add_hrect(y0=90, y1=100, fillcolor="#10B981", opacity=0.07,
                                   annotation_text="S", annotation_position="top left")
                  fig_evo.add_hrect(y0=75, y1=90, fillcolor="#6366F1", opacity=0.07,
                                   annotation_text="A", annotation_position="top left")
                  fig_evo.add_hrect(y0=55, y1=75, fillcolor="#F59E0B", opacity=0.07,
                                   annotation_text="B", annotation_position="top left")
                  fig_evo.add_hrect(y0=35, y1=55, fillcolor="#F97316", opacity=0.07,
                                   annotation_text="C", annotation_position="top left")
                  fig_evo.add_hrect(y0=0, y1=35, fillcolor="#EF4444", opacity=0.07,
                                   annotation_text="D", annotation_position="top left")

                  fig_evo.update_layout(
                      title="Skills Score Evolution Over Time",
                      template="plotly_dark",
                      paper_bgcolor="#0F172A",
                      plot_bgcolor="#0F172A",
                      height=500,
                      xaxis_title="Date",
                      yaxis_title="Score",
                      yaxis=dict(range=[0, 100]),
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=-0.25,
                          xanchor="center",
                          x=0.5,
                      ),
                      hovermode="x unified",
                  )

                  st.plotly_chart(fig_evo, use_container_width=True)

                  # Summary stats
                  st.markdown("### 📊 Skills 변화 요약")

                  if len(evo_df) >= 2:
                      change_cols = st.columns(len(skill_names))
                      for i, skill in enumerate(skill_names):
                          if skill in evo_df.columns:
                              first_val = evo_df[skill].iloc[0]
                              last_val = evo_df[skill].iloc[-1]
                              change = last_val - first_val
                              with change_cols[i]:
                                  st.metric(
                                      skill.replace(" ", "\n"),
                                      f"{last_val:.0f}",
                                      f"{change:+.1f}",
                                  )

                  # Most improved / declined
                  if len(evo_df) >= 2:
                      changes = {}
                      for skill in skill_names:
                          if skill in evo_df.columns:
                              changes[skill] = evo_df[skill].iloc[-1] - evo_df[skill].iloc[0]

                      most_improved = max(changes, key=changes.get)
                      most_declined = min(changes, key=changes.get)

                      col_imp, col_dec = st.columns(2)
                      with col_imp:
                          st.success(f"🚀 가장 향상된 Skill: **{most_improved}** ({changes[most_improved]:+.1f}점)")
                      with col_dec:
                          if changes[most_declined] < 0:
                              st.warning(f"📉 가장 하락한 Skill: **{most_declined}** ({changes[most_declined]:+.1f}점)")
                          else:
                              st.info(f"📈 모든 Skills가 향상되었습니다! 최소 향상: **{most_declined}** ({changes[most_declined]:+.1f}점)")
              else:
                  st.warning("롤링 윈도우 계산에 실패했습니다. 데이터를 확인해주세요.")


# ---------- Tab 8: Efficient Frontier ----------
if tab8 is not None:
  with tab8:
      st.markdown("""
      ### 📐 Efficient Frontier (Markowitz 최적화)
      랜덤 포트폴리오 샘플링으로 효율적 프론티어를 구성하고, 최적 포트폴리오를 식별합니다.
      """)

      with st.spinner("📐 효율적 프론티어 계산 중..."):
          try:
              n_frontier = st.select_slider(
                  "샘플링 수",
                  options=[1000, 3000, 5000, 10000],
                  value=3000,
                  key="frontier_samples",
              )
              frontier_result = compute_efficient_frontier(
                  prices=portfolio_data.prices,
                  weights=portfolio_data.holdings,
                  n_portfolios=n_frontier,
              )
              st.session_state["frontier_result"] = frontier_result

              # KPI
              fr_cols = st.columns(4)
              with fr_cols[0]:
                  st.metric("현재 Sharpe", f"{frontier_result.current_sharpe:.2f}")
              with fr_cols[1]:
                  st.metric("최대 Sharpe", f"{frontier_result.max_sharpe.sharpe_ratio:.2f}",
                            f"{frontier_result.max_sharpe.sharpe_ratio - frontier_result.current_sharpe:+.2f}")
              with fr_cols[2]:
                  st.metric("최소분산 Vol", f"{frontier_result.min_variance.volatility*100:.1f}%")
              with fr_cols[3]:
                  st.metric("현재 Vol", f"{frontier_result.current_volatility*100:.1f}%",
                            f"{(frontier_result.current_volatility - frontier_result.min_variance.volatility)*100:+.1f}%")

              st.markdown("---")

              # Frontier scatter
              fig_frontier = create_frontier_scatter(frontier_result)
              st.plotly_chart(fig_frontier, use_container_width=True)

              # Optimal weights comparison
              col_weights, col_table = st.columns([1, 1])
              with col_weights:
                  fig_opt_weights = create_optimal_weights_bar(frontier_result)
                  st.plotly_chart(fig_opt_weights, use_container_width=True)

              with col_table:
                  st.markdown("#### 📋 포트폴리오 비교")
                  summary_df = create_frontier_summary_table(frontier_result)
                  st.dataframe(summary_df, use_container_width=True, hide_index=True)

              # Insight
              st.markdown("### 💡 Insight")
              gap = frontier_result.max_sharpe.sharpe_ratio - frontier_result.current_sharpe
              if gap > 0.3:
                  st.warning(f"현재 포트폴리오는 최적 대비 Sharpe가 {gap:.2f} 낮습니다. **최대 Sharpe 포트폴리오**의 비중을 참고하세요.")
              elif gap > 0.1:
                  st.info(f"현재 포트폴리오는 효율적 프론티어에 비교적 가깝습니다 (Gap: {gap:.2f}).")
              else:
                  st.success("현재 포트폴리오가 효율적 프론티어 근처에 위치합니다! 잘 최적화된 포트폴리오입니다.")

          except Exception as e:
              st.error(f"효율적 프론티어 계산 실패: {e}")
              st.info("충분한 가격 데이터가 있는지 확인해주세요.")


# ---------- Tab 9: Stress Test ----------
if tab9 is not None:
  with tab9:
      st.markdown("""
      ### 🔥 Stress Test — 역사적 위기 시뮬레이션
      과거 금융 위기 시나리오에서 현재 포트폴리오가 겪었을 손실을 추정합니다.
      """)

      try:
          portfolio_vol = float(portfolio_data.returns.std() * np.sqrt(252))

          stress_results = run_all_stress_tests(
              weights=portfolio_data.holdings,
              sector_map=portfolio_data.sector_map,
              portfolio_vol=portfolio_vol,
          )
          st.session_state["stress_results"] = stress_results

          # 시나리오별 손실 비교 바 차트
          fig_stress_bar = create_stress_comparison_bar(stress_results)
          st.plotly_chart(fig_stress_bar, use_container_width=True)

          st.markdown("---")

          # 시나리오 상세 선택
          selected_scenario = st.selectbox(
              "📌 상세 분석할 시나리오 선택",
              list(stress_results.keys()),
          )

          if selected_scenario:
              result = stress_results[selected_scenario]

              # KPI
              stress_kpi = st.columns(4)
              with stress_kpi[0]:
                  st.metric("예상 손실", f"{result.portfolio_loss*100:.1f}%")
              with stress_kpi[1]:
                  st.metric("최악의 하루", f"{result.worst_day*100:.1f}%")
              with stress_kpi[2]:
                  st.metric("위기 기간", f"{result.scenario.duration_days}거래일")
              with stress_kpi[3]:
                  st.metric("회복 예상", f"~{result.recovery_estimate_days}거래일")

              # 설명
              st.markdown(f"""
              <div style="background: #1E293B; padding: 16px; border-radius: 12px;
                          border: 1px solid #334155; margin: 12px 0;">
                  <p style="color: #94A3B8; margin: 0;">{result.scenario.description}</p>
                  <p style="color: #64748B; margin: 4px 0 0 0; font-size: 0.85rem;">기간: {result.scenario.period}</p>
              </div>
              """, unsafe_allow_html=True)

              # 경로 차트
              fig_path = create_stress_path_chart(result)
              st.plotly_chart(fig_path, use_container_width=True)

              # 섹터 영향도
              col_tree, col_recovery = st.columns([1, 1])
              with col_tree:
                  fig_sector = create_sector_impact_treemap(
                      result, portfolio_data.holdings, portfolio_data.sector_map
                  )
                  st.plotly_chart(fig_sector, use_container_width=True)

              with col_recovery:
                  fig_recovery = create_recovery_timeline(stress_results)
                  st.plotly_chart(fig_recovery, use_container_width=True)

          # 종합 진단
          st.markdown("### 💡 Stress Test 종합 진단")
          worst_scenario = max(stress_results.values(), key=lambda r: abs(r.portfolio_loss))
          best_scenario = min(stress_results.values(), key=lambda r: abs(r.portfolio_loss))

          st.markdown(f"""
          - **가장 취약한 시나리오**: {worst_scenario.scenario.name} (예상 손실 {worst_scenario.portfolio_loss*100:.1f}%)
          - **가장 방어적인 시나리오**: {best_scenario.scenario.name} (예상 손실 {best_scenario.portfolio_loss*100:.1f}%)
          """)

          if abs(worst_scenario.portfolio_loss) > 0.40:
              st.error("⚠️ 특정 위기 시나리오에서 40% 이상 손실이 예상됩니다. 방어 자산 편입을 강력히 권장합니다.")
          elif abs(worst_scenario.portfolio_loss) > 0.25:
              st.warning("주요 위기에서 25% 이상 손실이 예상됩니다. 섹터 분산과 방어 전략을 검토하세요.")
          else:
              st.success("포트폴리오가 주요 위기 시나리오에서 비교적 견고합니다.")

      except Exception as e:
          st.error(f"Stress Test 실패: {e}")


# ---------- Tab 10: Geopolitical Scenario ----------
if tab10 is not None:
  with tab10:
      st.markdown("""
      ### 🌍 Geopolitical Scenario Engine
      지정학적 위기 시나리오의 매크로 충격이 포트폴리오에 미치는 영향을 시뮬레이션합니다.
      **매크로 충격 → 섹터 전파 → 종목 영향 → 포트폴리오 손익**
      """)

      try:
          geo_mode = st.radio(
              "시나리오 모드",
              ["📋 내장 시나리오", "🎛️ 커스텀 매크로 충격"],
              horizontal=True,
              key="geo_mode",
          )

          if geo_mode == "📋 내장 시나리오":
              # 시나리오 선택
              geo_choice = st.selectbox(
                  "시나리오 선택",
                  list(GEOPOLITICAL_SCENARIOS.keys()),
                  key="geo_scenario",
              )

              scenario = GEOPOLITICAL_SCENARIOS[geo_choice]

              # 시나리오 설명 카드
              prob_color = {"낮음": "#10B981", "중간": "#F59E0B", "높음": "#EF4444"}.get(scenario.probability, "#94A3B8")
              st.markdown(f"""
              <div style="background: linear-gradient(135deg, #1E293B 0%, #312E81 100%);
                          padding: 20px; border-radius: 12px; border: 1px solid #4F46E5;
                          margin: 12px 0;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                      <h4 style="color: #A5B4FC; margin: 0;">{scenario.name}</h4>
                      <div>
                          <span style="background: {prob_color}; color: white; padding: 4px 12px;
                                       border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                              발생확률: {scenario.probability}
                          </span>
                          <span style="background: #475569; color: white; padding: 4px 12px;
                                       border-radius: 20px; font-size: 0.8rem; margin-left: 4px;">
                              {scenario.time_horizon}
                          </span>
                      </div>
                  </div>
                  <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.8;">
                      {scenario.narrative}
                  </p>
              </div>
              """, unsafe_allow_html=True)

              # Key Triggers
              with st.expander("🔑 주요 트리거 이벤트"):
                  for trigger in scenario.key_triggers:
                      st.markdown(f"- {trigger}")

              # 매크로 충격 표시
              st.markdown("#### 📊 매크로 충격 설정값")
              shock_cols = st.columns(len(scenario.macro_shocks))
              for i, (var, val) in enumerate(scenario.macro_shocks.items()):
                  meta = MACRO_VARIABLES.get(var, {"name": var, "unit": ""})
                  with shock_cols[i]:
                      color = "#EF4444" if val > 0 and var in ("oil_price", "supply_chain", "geopolitical_risk", "recession") else "#10B981"
                      if var in ("interest_rate",) and val > 0:
                          color = "#EF4444"
                      st.metric(meta["name"], f"{val:+.0f}{meta.get('unit', '')}")

              macro_shocks = scenario.macro_shocks

          else:  # 커스텀 모드
              st.markdown("#### 🎛️ 매크로 충격 직접 설정")
              st.markdown("_각 슬라이더를 조절하여 원하는 시나리오를 만들어보세요._")

              macro_shocks = {}
              slider_cols = st.columns(3)

              for i, (var, meta) in enumerate(MACRO_VARIABLES.items()):
                  with slider_cols[i % 3]:
                      val = st.slider(
                          f"{meta['name']} ({meta['unit']})",
                          min_value=meta["range"][0],
                          max_value=meta["range"][1],
                          value=meta["default"],
                          key=f"geo_slider_{var}",
                      )
                      macro_shocks[var] = val

              # 프리셋 빠른 적용 버튼들
              st.markdown("**빠른 프리셋:**")
              preset_cols = st.columns(4)
              with preset_cols[0]:
                  if st.button("🛢️ 유가 급등", key="preset_oil"):
                      st.session_state["geo_slider_oil_price"] = 80
                      st.rerun()
              with preset_cols[1]:
                  if st.button("📈 금리 인상", key="preset_rate"):
                      st.session_state["geo_slider_interest_rate"] = 200
                      st.rerun()
              with preset_cols[2]:
                  if st.button("💵 환율 급등", key="preset_fx"):
                      st.session_state["geo_slider_usd_krw"] = 20
                      st.rerun()
              with preset_cols[3]:
                  if st.button("🔄 초기화", key="preset_reset"):
                      for var in MACRO_VARIABLES:
                          st.session_state[f"geo_slider_{var}"] = 0
                      st.rerun()

          # ── 분석 실행 ──
          st.markdown("---")

          with st.spinner("🌍 지정학 시나리오 분석 중..."):
              if geo_mode == "📋 내장 시나리오":
                  geo_result = run_geopolitical_scenario(
                      scenario=scenario,
                      weights=portfolio_data.holdings,
                      sector_map=portfolio_data.sector_map,
                  )
              else:
                  geo_result = propagate_shocks(
                      macro_shocks=macro_shocks,
                      weights=portfolio_data.holdings,
                      sector_map=portfolio_data.sector_map,
                  )

          # KPI
          geo_kpi = st.columns(4)
          with geo_kpi[0]:
              impact_color = "inverse" if geo_result.portfolio_impact_pct < 0 else "normal"
              st.metric("포트폴리오 영향", f"{geo_result.portfolio_impact_pct:+.1f}%")
          with geo_kpi[1]:
              st.metric("리스크 점수", f"{geo_result.risk_score:.0f}/100")
          with geo_kpi[2]:
              st.metric("최대 타격 섹터", geo_result.worst_sector)
          with geo_kpi[3]:
              st.metric("최대 수혜 섹터", geo_result.best_sector)

          # 워터폴 + 경로
          col_waterfall, col_path = st.columns([1, 1])

          with col_waterfall:
              fig_waterfall = create_macro_impact_waterfall(geo_result)
              st.plotly_chart(fig_waterfall, use_container_width=True)

          with col_path:
              fig_geo_path = create_geopolitical_path(geo_result)
              st.plotly_chart(fig_geo_path, use_container_width=True)

          # 섹터 영향 바
          fig_sector_bar = create_geo_sector_bar(geo_result)
          st.plotly_chart(fig_sector_bar, use_container_width=True)

          # 전체 시나리오 비교 (내장 시나리오 모드에서만)
          if geo_mode == "📋 내장 시나리오":
              st.markdown("---")
              st.markdown("### 🔍 전체 시나리오 비교 분석")

              with st.spinner("전체 시나리오 분석 중..."):
                  all_geo_results = {}
                  for name, sc in GEOPOLITICAL_SCENARIOS.items():
                      all_geo_results[name] = run_geopolitical_scenario(
                          scenario=sc,
                          weights=portfolio_data.holdings,
                          sector_map=portfolio_data.sector_map,
                      )

              col_radar_geo, col_heatmap_geo = st.columns([1, 1])

              with col_radar_geo:
                  fig_geo_radar = create_scenario_comparison_radar(all_geo_results)
                  st.plotly_chart(fig_geo_radar, use_container_width=True)

              with col_heatmap_geo:
                  fig_geo_heatmap = create_impact_heatmap(all_geo_results)
                  st.plotly_chart(fig_geo_heatmap, use_container_width=True)

              # 시나리오 영향 순위
              st.markdown("#### 📋 시나리오 영향 순위")
              ranked = sorted(all_geo_results.items(), key=lambda x: x[1].portfolio_impact_pct)
              rank_data = []
              for name, res in ranked:
                  rank_data.append({
                      "시나리오": name,
                      "포트폴리오 영향": f"{res.portfolio_impact_pct:+.1f}%",
                      "리스크 점수": f"{res.risk_score:.0f}",
                      "최대 타격": res.worst_sector,
                      "최대 수혜": res.best_sector,
                  })
              st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

          # Insight
          st.markdown("### 💡 Geopolitical Insight")
          if geo_result.portfolio_impact_pct < -15:
              st.error(f"⚠️ 이 시나리오에서 포트폴리오가 **{geo_result.portfolio_impact_pct:.1f}%** 하락할 수 있습니다. "
                       f"**{geo_result.worst_sector}** 섹터 비중 축소 또는 헤지를 검토하세요.")
          elif geo_result.portfolio_impact_pct < -5:
              st.warning(f"이 시나리오에서 포트폴리오가 약 {geo_result.portfolio_impact_pct:.1f}% 영향을 받습니다. "
                         f"방어적 자산 편입으로 리스크를 줄일 수 있습니다.")
          elif geo_result.portfolio_impact_pct > 5:
              st.success(f"이 시나리오에서 포트폴리오가 **{geo_result.portfolio_impact_pct:+.1f}%** 수혜를 받을 수 있습니다!")
          else:
              st.info("이 시나리오에서 포트폴리오 영향은 제한적입니다.")

      except Exception as e:
          st.error(f"지정학 시나리오 분석 실패: {e}")


# ---------- Tab 11: Correlation Network ----------
if tab11 is not None:
  with tab11:
      st.markdown("""
      ### 🕸️ Correlation Network
      종목 간 상관관계를 네트워크 그래프로 시각화합니다. 높은 상관관계를 가진 종목 그룹을 자동 탐지합니다.
      """)

      try:
          corr_threshold = st.slider(
              "상관계수 임계값 (이 값 이상만 엣지 표시)",
              min_value=0.1,
              max_value=0.9,
              value=0.3,
              step=0.05,
              key="corr_threshold",
          )

          with st.spinner("🕸️ 상관관계 네트워크 계산 중..."):
              network_data = compute_correlation_network(
                  prices=portfolio_data.prices,
                  weights=portfolio_data.holdings,
                  sector_map=portfolio_data.sector_map,
                  corr_threshold=corr_threshold,
              )
              st.session_state["network_data"] = network_data

          # KPI
          net_kpi = st.columns(3)
          with net_kpi[0]:
              st.metric("평균 상관계수", f"{network_data.avg_correlation:.3f}")
          with net_kpi[1]:
              st.metric("클러스터 수", f"{len(network_data.clusters)}")
          with net_kpi[2]:
              st.metric("엣지 수", f"{len(network_data.edges)}")

          st.markdown("---")

          # Network graph
          fig_network = create_network_graph(network_data)
          st.plotly_chart(fig_network, use_container_width=True)

          # Heatmap + Cluster summary
          col_heat, col_cluster = st.columns([1, 1])

          with col_heat:
              fig_heatmap_corr = create_correlation_heatmap(network_data.correlation_matrix)
              st.plotly_chart(fig_heatmap_corr, use_container_width=True)

          with col_cluster:
              st.markdown("#### 🏷️ 클러스터 구성")
              cluster_df = create_cluster_summary(network_data)
              st.dataframe(cluster_df, use_container_width=True, hide_index=True)

          # Insight
          st.markdown("### 💡 분산투자 Insight")
          if network_data.avg_correlation > 0.6:
              st.warning(f"평균 상관계수가 {network_data.avg_correlation:.2f}로 높습니다. 종목들이 비슷하게 움직이므로 분산 효과가 제한적입니다.")
          elif network_data.avg_correlation > 0.4:
              st.info(f"평균 상관계수 {network_data.avg_correlation:.2f} — 보통 수준입니다. 상관관계가 낮은 자산을 추가하면 분산이 개선됩니다.")
          else:
              st.success(f"평균 상관계수 {network_data.avg_correlation:.2f} — 좋은 분산 효과를 보이고 있습니다!")

          if len(network_data.clusters) <= 2 and len(network_data.nodes) > 4:
              st.warning("대부분의 종목이 1~2개 클러스터에 집중되어 있습니다. 다른 섹터/자산군 편입을 고려하세요.")

      except Exception as e:
          import traceback
          st.error(f"상관관계 네트워크 계산 실패: {e}")
          with st.expander("🔍 상세 오류 정보"):
              st.code(traceback.format_exc())
          st.info("최소 2개 이상의 종목이 필요합니다.")


# ---------- Tab 12: Multi-Portfolio Compare ----------
if tab12 is not None:
  with tab12:
      st.markdown("""
      ### ⚖️ Multi-Portfolio Comparison
      현재 포트폴리오를 다양한 벤치마크 포트폴리오와 비교합니다.
      """)

      try:
          # 비교할 포트폴리오 선택
          compare_options = list(COMPARISON_PORTFOLIOS.keys())
          selected_compare = st.multiselect(
              "비교할 포트폴리오 선택 (최대 4개)",
              compare_options,
              default=compare_options[:2],
              max_selections=4,
          )

          if selected_compare:
              with st.spinner("⚖️ 포트폴리오 비교 분석 중..."):
                  # 현재 포트폴리오 요약
                  current_summary = compute_portfolio_summary(
                      name="📌 현재 포트폴리오",
                      weights=portfolio_data.holdings,
                      returns=portfolio_data.returns,
                      benchmark_returns=portfolio_data.benchmark_returns,
                      skills_dict=profile.to_dict(),
                  )

                  all_summaries = [current_summary]

                  # 비교 포트폴리오들 분석
                  for comp_name in selected_compare:
                      comp_weights = COMPARISON_PORTFOLIOS[comp_name]

                      # 비교 포트폴리오용 간이 Skills 추정 (현재 데이터 기반)
                      # 실제로는 각 포트폴리오의 가격 데이터를 가져와야 하지만,
                      # 현재 포트폴리오 대비 상대 추정치를 사용
                      n_assets = len(comp_weights)
                      max_weight = max(comp_weights.values())
                      div_score = min(95, max(20, n_assets * 8 + (1 - max_weight) * 100))

                      # 간이 Skills 추정
                      base_skills = profile.to_dict()
                      estimated_skills = {
                          "Timing": max(20, min(95, base_skills["Timing"] + np.random.normal(0, 8))),
                          "Diversification": div_score,
                          "Risk Management": max(20, min(95, base_skills["Risk Management"] + np.random.normal(0, 10))),
                          "Conviction": max(20, min(95, 50 + max_weight * 80 + np.random.normal(0, 5))),
                          "Adaptability": max(20, min(95, base_skills["Adaptability"] + np.random.normal(0, 8))),
                          "Consistency": max(20, min(95, base_skills["Consistency"] + np.random.normal(0, 8))),
                      }

                      # 간이 성과 추정
                      np.random.seed(hash(comp_name) % 2**31)
                      est_return = portfolio_data.returns.mean() + np.random.normal(0, 0.0003)
                      est_returns = portfolio_data.returns + np.random.normal(0, 0.002, len(portfolio_data.returns))
                      est_returns = pd.Series(est_returns, index=portfolio_data.returns.index)

                      comp_summary = compute_portfolio_summary(
                          name=comp_name,
                          weights=comp_weights,
                          returns=est_returns,
                          benchmark_returns=portfolio_data.benchmark_returns,
                          skills_dict=estimated_skills,
                      )
                      all_summaries.append(comp_summary)

                  # Radar comparison
                  fig_multi_radar = create_multi_radar(all_summaries)
                  st.plotly_chart(fig_multi_radar, use_container_width=True)

                  st.markdown("---")

                  # Performance comparison
                  col_perf, col_risk = st.columns([1, 1])
                  with col_perf:
                      fig_perf = create_perf_comparison_bar(all_summaries)
                      st.plotly_chart(fig_perf, use_container_width=True)

                  with col_risk:
                      fig_multi_rr = create_multi_risk_return(all_summaries)
                      st.plotly_chart(fig_multi_rr, use_container_width=True)

                  # Comparison table
                  st.markdown("### 📋 상세 비교 테이블")
                  comp_table = create_comparison_table(all_summaries)
                  st.dataframe(comp_table, use_container_width=True, hide_index=True)

                  # Insight
                  st.markdown("### 💡 비교 Insight")
                  best_overall = max(all_summaries, key=lambda s: s.overall_skill)
                  best_sharpe = max(all_summaries, key=lambda s: s.sharpe_ratio)

                  if best_overall.name == "📌 현재 포트폴리오":
                      st.success("현재 포트폴리오의 Overall Skill이 비교 대상 중 가장 높습니다!")
                  else:
                      st.info(f"**{best_overall.name}**이(가) Overall Skill {best_overall.overall_skill:.0f}점으로 가장 높습니다. 참고하여 전략을 조정해보세요.")

                  if best_sharpe.name != best_overall.name:
                      st.info(f"Sharpe Ratio 기준으로는 **{best_sharpe.name}**이(가) {best_sharpe.sharpe_ratio:.2f}로 가장 효율적입니다.")
          else:
              st.info("비교할 포트폴리오를 선택해주세요.")

      except Exception as e:
          st.error(f"포트폴리오 비교 실패: {e}")


# ---------- Tab 13: Investment Style ----------
if tab13 is not None:
  with tab13:
      st.markdown("""
      ### 🎨 Investment Style Analysis
      포트폴리오의 투자 스타일을 Morningstar Style Box와 멀티팩터 레이더로 진단합니다.
      """)

      try:
          with st.spinner("🎨 투자 스타일 분석 중..."):
              style_result = analyze_portfolio_style(
                  returns=portfolio_data.returns,
                  benchmark_returns=portfolio_data.benchmark_returns,
                  weights=portfolio_data.holdings,
                  prices=portfolio_data.prices,
                  sector_map=portfolio_data.sector_map,
                  returns_by_ticker=portfolio_data.returns_by_ticker,
              )

          # Style summary card
          style_info = format_style_result(style_result)
          st.markdown(f"""
          <div style="background: #1E293B; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px;">
              <h3 style="color: #6366F1; margin-top: 0;">📌 {style_result.primary_style}</h3>
              <p style="color: #F8FAFC; font-size: 1.1rem;">{style_result.style_description}</p>
              <p style="color: #94A3B8;">부가 특성: {', '.join(style_result.secondary_traits) if style_result.secondary_traits else '없음'}</p>
          </div>
          """, unsafe_allow_html=True)

          col_box, col_radar_style = st.columns([1, 1])

          with col_box:
              fig_style_box = create_style_box(style_result)
              st.plotly_chart(fig_style_box, use_container_width=True)

          with col_radar_style:
              fig_style_radar = create_style_radar(style_result)
              st.plotly_chart(fig_style_radar, use_container_width=True)

          # Style scores detail
          st.markdown("### 📊 스타일 점수 상세")
          style_cols = st.columns(6)
          style_scores = [
              ("Value", style_result.value_score, "가치주 성향"),
              ("Growth", style_result.growth_score, "성장주 성향"),
              ("Momentum", style_result.momentum_score, "모멘텀 강도"),
              ("Quality", style_result.quality_score, "안정성/품질"),
              ("Dividend", style_result.dividend_score, "배당 성향"),
              ("Volatility", style_result.volatility_score, "변동성 수준"),
          ]
          for col, (name, score, desc) in zip(style_cols, style_scores):
              with col:
                  color = "#10B981" if score >= 70 else "#F59E0B" if score >= 40 else "#EF4444"
                  st.markdown(f"""
                  <div style="background: #1E293B; padding: 12px; border-radius: 10px; border: 1px solid #334155; text-align: center;">
                      <div style="font-size: 0.85rem; color: #94A3B8;">{name}</div>
                      <div style="font-size: 1.8rem; font-weight: 700; color: {color};">{score:.0f}</div>
                      <div style="font-size: 0.75rem; color: #64748B;">{desc}</div>
                  </div>
                  """, unsafe_allow_html=True)

          # Style drift analysis
          st.markdown("---")
          st.markdown("### 📈 Style Drift (스타일 변화 추적)")

          drift_window = st.select_slider(
              "롤링 윈도우 (거래일)",
              options=[63, 126, 189, 252],
              value=126,
              format_func=lambda x: f"{x}일 (~{x//21}개월)",
          )

          # weights_history: Dict[ticker, float] — static weights treated as constant
          weights_as_history = {t: w for t, w in portfolio_data.holdings.items()}
          drift_df = analyze_style_drift(
              returns_history=portfolio_data.returns,
              weights_history=weights_as_history,
              benchmark_returns=portfolio_data.benchmark_returns,
              sector_map=portfolio_data.sector_map,
              window=drift_window,
          )

          if drift_df is not None and len(drift_df) > 0:
              fig_drift = create_style_timeline(drift_df)
              st.plotly_chart(fig_drift, use_container_width=True)
          else:
              st.info("스타일 드리프트 분석에는 충분한 데이터가 필요합니다 (최소 롤링 윈도우 + 20일).")

          # Insight
          st.markdown("### 💡 스타일 Insight")
          if style_result.value_score > 65 and style_result.growth_score > 65:
              st.info("🔀 포트폴리오가 Value와 Growth 특성을 동시에 보유합니다. GARP(Growth At Reasonable Price) 전략과 유사합니다.")
          elif style_result.growth_score > 75:
              st.info("🚀 성장주 위주의 포트폴리오입니다. 고성장 기대, 단 시장 하락기 변동성에 유의하세요.")
          elif style_result.value_score > 75:
              st.info("🏛️ 가치주 중심의 포트폴리오입니다. 안정적 배당과 언더밸류 종목이 특징이지만 성장성이 낮을 수 있습니다.")
          else:
              st.info("⚖️ 블렌드 스타일 포트폴리오입니다. 성장과 가치가 균형을 이루고 있습니다.")

          if style_result.momentum_score > 70:
              st.success(f"🔥 모멘텀 점수 {style_result.momentum_score:.0f} — 최근 상승 추세가 강합니다.")
          elif style_result.momentum_score < 30:
              st.warning(f"📉 모멘텀 점수 {style_result.momentum_score:.0f} — 최근 하락 추세를 보이고 있습니다.")

      except Exception as e:
          st.error(f"투자 스타일 분석 실패: {e}")
          st.info("분석을 위해 최소 60일 이상의 데이터와 2개 이상의 종목이 필요합니다.")


# ---------- Tab 14: Risk Contribution ----------
if tab14 is not None:
  with tab14:
      st.markdown("""
      ### ⚡ Risk Contribution Analysis
      각 종목이 포트폴리오 전체 리스크에 기여하는 정도를 Euler 분해로 분석합니다.
      """)

      try:
          with st.spinner("⚡ 리스크 기여도 분석 중..."):
              risk_result = compute_risk_contribution(
                  prices=portfolio_data.prices,
                  weights=portfolio_data.holdings,
              )

          # KPI cards
          risk_summary = get_risk_summary(risk_result)
          risk_kpi_cols = st.columns(4)

          kpi_items = [
              ("포트폴리오 변동성", f"{risk_summary['portfolio_volatility']:.1f}%", "연환산"),
              ("분산효과 비율", f"{risk_summary['diversification_ratio']:.2f}", "> 1.0 = 분산 이점"),
              ("리스크 집중도", f"{risk_summary['concentration_risk']:.1f}%", "최대 종목 기여"),
              ("종목 수", str(risk_summary['num_holdings']), "분석 대상"),
          ]
          for col, (label, val, help_text) in zip(risk_kpi_cols, kpi_items):
              with col:
                  st.metric(label, val, help=help_text)

          st.markdown("---")

          col_bar, col_donut = st.columns([1, 1])

          with col_bar:
              fig_risk_bar = create_risk_contribution_bar(risk_result)
              st.plotly_chart(fig_risk_bar, use_container_width=True)

          with col_donut:
              fig_risk_donut = create_risk_donut(risk_result)
              st.plotly_chart(fig_risk_donut, use_container_width=True)

          # Risk budget table
          st.markdown("### 📋 리스크 예산 테이블")
          risk_table = create_risk_budget_table(risk_result)
          st.dataframe(risk_table, use_container_width=True, hide_index=True)

          # Risk outliers
          st.markdown("### 🔍 리스크 이상치 탐지")
          outliers = identify_risk_outliers(risk_result)

          if outliers["overweight_risk"]:
              for ticker in outliers["overweight_risk"]:
                  idx = risk_result.tickers.index(ticker)
                  w_pct = risk_result.weights[idx] * 100
                  r_pct = risk_result.pct_contribution[idx]
                  ratio = r_pct / w_pct if w_pct > 0 else 0
                  st.warning(f"⚠️ **{ticker}**: 비중 {w_pct:.1f}%인데 리스크 기여 {r_pct:.1f}% — 비중 대비 리스크가 {ratio:.1f}배 높습니다.")

          if outliers["underweight_risk"]:
              for ticker in outliers["underweight_risk"]:
                  idx = risk_result.tickers.index(ticker)
                  w_pct = risk_result.weights[idx] * 100
                  r_pct = risk_result.pct_contribution[idx]
                  st.info(f"✅ **{ticker}**: 비중 {w_pct:.1f}%인데 리스크 기여 {r_pct:.1f}% — 효율적인 분산 역할을 하고 있습니다.")

          if not outliers["overweight_risk"] and not outliers["underweight_risk"]:
              st.success("✅ 모든 종목의 리스크 기여도가 비중과 유사합니다. 균형 잡힌 리스크 배분입니다.")

          # Insight
          st.markdown("### 💡 리스크 기여도 Insight")
          if risk_result.diversification_ratio > 1.3:
              st.success(f"분산효과 비율 {risk_result.diversification_ratio:.2f} — 분산투자 효과가 뚜렷합니다. 개별 종목 리스크의 합보다 포트폴리오 리스크가 {(risk_result.diversification_ratio - 1)*100:.0f}% 낮습니다.")
          elif risk_result.diversification_ratio > 1.1:
              st.info(f"분산효과 비율 {risk_result.diversification_ratio:.2f} — 적당한 분산 효과가 있지만, 상관관계가 낮은 자산 추가로 개선 가능합니다.")
          else:
              st.warning(f"분산효과 비율 {risk_result.diversification_ratio:.2f} — 분산 효과가 미미합니다. 종목 간 상관관계가 높아 리스크가 집중되어 있습니다.")

          if risk_result.concentration_risk > 50:
              st.warning(f"리스크의 {risk_result.concentration_risk:.0f}%가 단일 종목에 집중되어 있습니다. 비중 조절을 고려하세요.")

      except Exception as e:
          st.error(f"리스크 기여도 분석 실패: {e}")
          st.info("분석을 위해 최소 30일 이상의 가격 데이터가 필요합니다.")


# ---------- Tab 15: Market Events ----------
if tab15 is not None:
  with tab15:
      st.markdown("""
      ### 📅 Market Events Timeline
      주요 시장 이벤트가 포트폴리오에 미친 영향을 시각화합니다. (2019~2025 글로벌 39개 이벤트)
      """)

      try:
          # Annotated returns chart
          st.markdown("#### 📈 이벤트 어노테이션 차트")
          max_events_display = st.slider(
              "표시할 최대 이벤트 수",
              min_value=3,
              max_value=15,
              value=8,
              help="영향이 큰 이벤트부터 우선 표시됩니다",
          )

          fig_annotated = create_annotated_returns_chart(
              portfolio_returns=portfolio_data.returns,
              benchmark_returns=portfolio_data.benchmark_returns,
              portfolio_name=portfolio_data.name,
              benchmark_name=benchmark,
              max_annotations=max_events_display,
          )
          st.plotly_chart(fig_annotated, use_container_width=True)

          # Event impact analysis
          st.markdown("---")
          st.markdown("#### 📊 이벤트 영향도 분석")

          start_str = str(portfolio_data.returns.index[0])[:10]
          end_str = str(portfolio_data.returns.index[-1])[:10]
          events_in_range = get_events_in_range(start_str, end_str)

          if events_in_range:
              impact_df = create_event_impact_table(events_in_range, portfolio_data.returns)
              if len(impact_df) > 0:
                  st.dataframe(
                      impact_df,
                      use_container_width=True,
                      hide_index=True,
                  )

                  # Insight: best/worst event response
                  st.markdown("### 💡 이벤트 대응 Insight")
                  if "직후 20일" in impact_df.columns and len(impact_df) > 0:
                      # Parse percentage strings like "+3.2%" → 3.2
                      numeric_col = impact_df["직후 20일"].str.replace("%", "").str.replace("+", "").astype(float)
                      if numeric_col.notna().any():
                          best_idx = numeric_col.idxmax()
                          worst_idx = numeric_col.idxmin()

                          if best_idx is not None:
                              best_event = impact_df.loc[best_idx]
                              st.success(f"🏆 최고 대응: **{best_event.get('이벤트', 'N/A')}** — 20일 후 수익률 {numeric_col.loc[best_idx]:+.2f}%")
                          if worst_idx is not None:
                              worst_event = impact_df.loc[worst_idx]
                              st.warning(f"📉 최약 대응: **{worst_event.get('이벤트', 'N/A')}** — 20일 후 수익률 {numeric_col.loc[worst_idx]:+.2f}%")

                  # Event category summary
                  st.markdown("#### 📋 카테고리별 이벤트 수")
                  from collections import Counter
                  cat_counts = Counter(e.category for e in events_in_range)
                  cat_icons = {"crash": "🔴", "policy": "🟡", "geopolitical": "🌍", "recovery": "🟢", "earnings": "💰", "neutral": "⚪"}
                  cat_text = " | ".join(f"{cat_icons.get(cat, '⚪')} {cat}: {count}개" for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]))
                  st.markdown(f"<div style='background: #1E293B; padding: 12px; border-radius: 8px; border: 1px solid #334155;'>{cat_text}</div>", unsafe_allow_html=True)
          else:
              st.info("선택한 분석 기간 내에 등록된 시장 이벤트가 없습니다.")

      except Exception as e:
          st.error(f"시장 이벤트 분석 실패: {e}")
          st.info("이벤트 분석은 2019년 이후 데이터에서 가장 효과적입니다.")


# ---------- Tab 16: Regime Detection ----------
if tab16 is not None:
  with tab16:
      st.markdown("""
      ### 🔀 Market Regime Detection
      시장을 Bull(상승) / Bear(하락) / Sideways(횡보) 국면으로 자동 분류하고, 국면별 포트폴리오 성과를 분석합니다.
      """)

      try:
          regime_window = st.select_slider(
              "국면 탐지 윈도우",
              options=[21, 42, 63, 126],
              value=63,
              format_func=lambda x: f"{x}일 (~{x//21}개월)",
              key="regime_window",
          )

          with st.spinner("🔀 시장 국면 탐지 중..."):
              regimes_df = detect_regimes(portfolio_data.benchmark_returns, window=regime_window)
              regime_stats = analyze_regime_performance(
                  portfolio_data.returns,
                  portfolio_data.benchmark_returns,
                  regimes_df,
              )

          # Current regime indicator
          current = get_current_regime(regimes_df)
          regime_colors = {"Bull": "#10B981", "Bear": "#EF4444", "Sideways": "#F59E0B"}
          regime_icons = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}
          curr_regime = current.get("regime", "Unknown")
          st.markdown(f"""
          <div style="background: #1E293B; padding: 16px; border-radius: 12px; border: 2px solid {regime_colors.get(curr_regime, '#6B7280')}; margin-bottom: 16px;">
              <span style="font-size: 1.5rem;">{regime_icons.get(curr_regime, '⚪')}</span>
              <span style="font-size: 1.3rem; font-weight: 700; color: {regime_colors.get(curr_regime, '#F8FAFC')};">
                  현재 국면: {curr_regime}
              </span>
              <span style="color: #94A3B8; margin-left: 16px;">
                  롤링 수익률: {current.get('rolling_return', 0)*100:+.1f}% | 롤링 변동성: {current.get('rolling_vol', 0)*100:.1f}%
              </span>
          </div>
          """, unsafe_allow_html=True)

          # KPI by regime (use regime_stats directly for per-regime data)
          total_days = sum(regime_stats.get(r, {}).get("count_days", 0) for r in ["Bull", "Bear", "Sideways"])
          kpi_cols = st.columns(3)
          for i, regime_name in enumerate(["Bull", "Bear", "Sideways"]):
              with kpi_cols[i]:
                  stats = regime_stats.get(regime_name, {})
                  pct = (stats.get("count_days", 0) / max(total_days, 1)) * 100
                  ann_ret = stats.get("annualized_return", 0) * 100
                  sharpe = stats.get("sharpe_ratio", 0)
                  st.markdown(f"""
                  <div style="background: #1E293B; padding: 14px; border-radius: 10px; border: 1px solid #334155; text-align: center;">
                      <div style="font-size: 1rem; color: {regime_colors.get(regime_name, '#F8FAFC')};">{regime_icons.get(regime_name, '')} {regime_name}</div>
                      <div style="font-size: 0.85rem; color: #94A3B8;">기간 비중: {pct:.0f}%</div>
                      <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">{ann_ret:+.1f}%</div>
                      <div style="font-size: 0.8rem; color: #64748B;">Sharpe: {sharpe:.2f}</div>
                  </div>
                  """, unsafe_allow_html=True)

          st.markdown("---")

          # Timeline chart
          fig_regime_tl = create_regime_timeline(portfolio_data.returns, regimes_df)
          st.plotly_chart(fig_regime_tl, use_container_width=True)

          col_perf, col_trans = st.columns([1, 1])

          with col_perf:
              fig_regime_bars = create_regime_performance_bars(regime_stats)
              st.plotly_chart(fig_regime_bars, use_container_width=True)

          with col_trans:
              fig_trans = create_regime_transition_matrix(regimes_df)
              st.plotly_chart(fig_trans, use_container_width=True)

          # Duration chart
          fig_dur = create_regime_duration_chart(regimes_df)
          st.plotly_chart(fig_dur, use_container_width=True)

          # Insight
          st.markdown("### 💡 국면 분석 Insight")
          bull_s = regime_stats.get("Bull", {})
          bear_s = regime_stats.get("Bear", {})
          bull_pct = (bull_s.get("count_days", 0) / max(total_days, 1)) * 100

          if bear_s.get("sharpe_ratio", 0) > 0:
              st.success(f"📈 Bear 국면에서도 Sharpe {bear_s['sharpe_ratio']:.2f} — 하락장 방어력이 우수합니다!")
          elif bear_s.get("annualized_return", -1) > -0.10:
              st.info(f"🛡️ Bear 국면 연환산 수익률 {bear_s.get('annualized_return', 0)*100:+.1f}% — 중간 수준의 방어력입니다.")
          else:
              st.warning(f"⚠️ Bear 국면 연환산 수익률 {bear_s.get('annualized_return', 0)*100:+.1f}% — 하락장 대응 전략 보강이 필요합니다.")

          if bull_pct > 60:
              st.info(f"분석 기간의 {bull_pct:.0f}%가 Bull 국면이었습니다. 하락장 시나리오 테스트를 Stress Test 탭에서 확인하세요.")

      except Exception as e:
          st.error(f"국면 탐지 실패: {e}")
          st.info("국면 분석을 위해 최소 63일 이상의 데이터가 필요합니다.")


# ---------- Tab 17: Tail Risk ----------
if tab17 is not None:
  with tab17:
      st.markdown("""
      ### 🎲 Tail Risk Analysis
      정규분포 가정을 넘어, 왜도·첨도를 반영한 Cornish-Fisher VaR로 실제 꼬리 리스크를 측정합니다.
      """)

      try:
          with st.spinner("🎲 꼬리 리스크 분석 중..."):
              tail_result = tail_risk_analysis(portfolio_data.returns)

          # KPI cards
          tail_kpi = st.columns(5)
          kpi_data = [
              ("CF VaR (95%)", f"{tail_result.cornish_fisher_var*100:.2f}%", "Cornish-Fisher"),
              ("CVaR (ES)", f"{tail_result.cvar*100:.2f}%", "Expected Shortfall"),
              ("Tail Ratio", f"{tail_result.tail_ratio:.2f}", "> 1 = 왼쪽 꼬리 두꺼움"),
              ("왜도 (Skew)", f"{tail_result.skewness:.2f}", "< 0 = 왼쪽 비대칭"),
              ("첨도 (Kurt)", f"{tail_result.kurtosis_excess:.2f}", "> 0 = Fat tail"),
          ]
          for col, (label, val, help_t) in zip(tail_kpi, kpi_data):
              with col:
                  st.metric(label, val, help=help_t)

          # Normality test
          jb_pval = tail_result.jarque_bera_pvalue
          if jb_pval < 0.01:
              st.error(f"📊 Jarque-Bera p-value = {jb_pval:.4f} — 수익률 분포가 정규분포와 **매우 다릅니다**. Gaussian VaR는 리스크를 과소평가합니다.")
          elif jb_pval < 0.05:
              st.warning(f"📊 Jarque-Bera p-value = {jb_pval:.4f} — 정규분포 가정이 약합니다. Cornish-Fisher VaR 사용을 권장합니다.")
          else:
              st.success(f"📊 Jarque-Bera p-value = {jb_pval:.4f} — 수익률이 정규분포에 근사합니다.")

          # CF improvement
          cf_improvement = tail_result.cf_improvement
          if abs(cf_improvement) > 5:
              st.info(f"🔬 Cornish-Fisher VaR는 Gaussian 대비 **{cf_improvement:+.1f}%** 차이 — 왜도/첨도 보정 효과가 유의합니다.")

          st.markdown("---")

          # Charts
          col_dist, col_qq = st.columns([1, 1])

          with col_dist:
              fig_dist = create_return_distribution(portfolio_data.returns)
              st.plotly_chart(fig_dist, use_container_width=True)

          with col_qq:
              fig_qq = create_tail_qq_plot(portfolio_data.returns)
              st.plotly_chart(fig_qq, use_container_width=True)

          # VaR comparison
          fig_var_comp = create_var_comparison_chart(portfolio_data.returns)
          st.plotly_chart(fig_var_comp, use_container_width=True)

          # Rolling VaR
          st.markdown("### 📈 Rolling VaR (Cornish-Fisher)")
          rolling_window = st.select_slider(
              "Rolling 윈도우",
              options=[21, 42, 63, 126],
              value=63,
              format_func=lambda x: f"{x}일",
              key="tail_rolling_window",
          )
          fig_rolling_var = create_rolling_var_chart(portfolio_data.returns, window=rolling_window)
          st.plotly_chart(fig_rolling_var, use_container_width=True)

          # Multi-confidence table
          st.markdown("### 📋 신뢰 수준별 VaR 비교")
          var_table = multi_confidence_var(portfolio_data.returns)
          st.dataframe(var_table, use_container_width=True, hide_index=True)

          # Insight
          st.markdown("### 💡 Tail Risk Insight")
          if tail_result.kurtosis_excess > 3:
              st.warning(f"📐 초과 첨도 {tail_result.kurtosis_excess:.2f} — 극단적 손실이 정규분포 예상보다 자주 발생합니다. 테일 헤지를 고려하세요.")
          if tail_result.skewness < -0.5:
              st.warning(f"↙️ 왜도 {tail_result.skewness:.2f} — 수익률 분포가 왼쪽으로 치우쳐 있어 큰 손실 확률이 높습니다.")
          elif tail_result.skewness > 0.5:
              st.success(f"↗️ 왜도 {tail_result.skewness:.2f} — 수익률 분포가 오른쪽으로 치우쳐 있어 큰 수익 기회가 더 많습니다.")

          if tail_result.tail_ratio > 1.5:
              st.warning(f"⚠️ Tail Ratio {tail_result.tail_ratio:.2f} — 하방 리스크가 상방 기회보다 {tail_result.tail_ratio:.1f}배 큽니다.")

      except Exception as e:
          st.error(f"꼬리 리스크 분석 실패: {e}")
          st.info("분석을 위해 최소 30일 이상의 수익률 데이터가 필요합니다.")


# =============================================================================
# Tab 18: GARCH Volatility
# =============================================================================

if tab18 is not None:
  with tab18:
      st.markdown("## 📉 GARCH(1,1) 변동성 예측")
      st.markdown("""
      <div style="background: #1E293B; padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; color: #94A3B8;">
          GARCH(1,1) 모델로 조건부 변동성을 추정하고, 미래 변동성을 예측합니다.<br>
          <b>σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}</b> — 변동성 클러스터링(큰 변동 뒤에 큰 변동)을 포착합니다.
      </div>
      """, unsafe_allow_html=True)

      try:
          with st.spinner("📉 GARCH 모델 피팅 중..."):
              garch_result = fit_garch(portfolio_data.returns)

          if garch_result is None:
              st.warning("GARCH 분석을 위해 최소 50일 이상의 수익률 데이터가 필요합니다.")
          else:
              # KPI Cards
              g_cols = st.columns(4)
              with g_cols[0]:
                  vol_color = "inverse" if garch_result.vol_regime == "높음" else ("off" if garch_result.vol_regime == "보통" else "normal")
                  st.metric("현재 변동성", f"{garch_result.current_vol:.1%}", delta=f"레짐: {garch_result.vol_regime}", delta_color=vol_color)
              with g_cols[1]:
                  st.metric("장기 변동성", f"{garch_result.long_run_vol:.1%}")
              with g_cols[2]:
                  st.metric("지속성 (α+β)", f"{garch_result.persistence:.4f}", delta="높은 지속성" if garch_result.persistence > 0.95 else "보통")
              with g_cols[3]:
                  st.metric("1일 후 예측", f"{garch_result.forecast_vol_1d:.1%}")

              # Charts
              col_g1, col_g2 = st.columns(2)
              with col_g1:
                  fig_garch_vol = create_garch_vol_chart(garch_result, portfolio_data.returns)
                  st.plotly_chart(fig_garch_vol, use_container_width=True)
              with col_g2:
                  fig_vol_forecast = create_vol_forecast_chart(garch_result)
                  st.plotly_chart(fig_vol_forecast, use_container_width=True)

              # Parameters
              st.markdown("### 🔧 GARCH 파라미터")
              col_p1, col_p2 = st.columns([1, 2])
              with col_p1:
                  fig_params = create_garch_params_chart(garch_result)
                  st.plotly_chart(fig_params, use_container_width=True)
              with col_p2:
                  st.markdown(f"""
                  | 파라미터 | 값 | 해석 |
                  |---------|------|------|
                  | ω (omega) | {garch_result.omega:.6f} | 장기 분산 기여 |
                  | α (alpha) | {garch_result.alpha:.4f} | 충격 반응도 — {'높음' if garch_result.alpha > 0.15 else '보통' if garch_result.alpha > 0.08 else '낮음'} |
                  | β (beta) | {garch_result.beta:.4f} | 변동성 지속도 — {'높음' if garch_result.beta > 0.9 else '보통' if garch_result.beta > 0.8 else '낮음'} |
                  | α + β | {garch_result.persistence:.4f} | 1에 가까울수록 변동성 지속 |
                  | Log-Likelihood | {garch_result.log_likelihood:.2f} | 모델 적합도 |
                  | AIC | {garch_result.aic:.2f} | 정보 기준 (낮을수록 좋음) |
                  | BIC | {garch_result.bic:.2f} | 베이지안 정보 기준 |
                  """)

              # GARCH Monte Carlo
              st.markdown("### 🎲 GARCH Monte Carlo 시뮬레이션")
              mc_cols = st.columns([1, 3])
              with mc_cols[0]:
                  mc_days = st.slider("시뮬레이션 기간 (일)", 20, 120, 60, key="garch_mc_days")
                  mc_sims = st.slider("시뮬레이션 횟수", 100, 2000, 500, step=100, key="garch_mc_sims")

              with mc_cols[1]:
                  mc_paths = garch_monte_carlo(
                      portfolio_data.returns, garch_result,
                      n_days=mc_days, n_sims=mc_sims, initial_value=10000,
                  )
                  # Plot MC paths
                  fig_mc = go.Figure()
                  # Plot percentile bands
                  percentiles = mc_paths.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                  days_range = list(range(1, mc_days + 1))

                  fig_mc.add_trace(go.Scatter(
                      x=days_range, y=percentiles.loc[0.95].values,
                      name="95th %ile", line=dict(color="#10B981", width=1, dash="dot"),
                  ))
                  fig_mc.add_trace(go.Scatter(
                      x=days_range, y=percentiles.loc[0.75].values,
                      name="75th %ile", line=dict(color="#6366F1", width=1, dash="dash"),
                      fill="tonexty", fillcolor="rgba(99,102,241,0.1)",
                  ))
                  fig_mc.add_trace(go.Scatter(
                      x=days_range, y=percentiles.loc[0.5].values,
                      name="중앙값", line=dict(color="#F8FAFC", width=2),
                  ))
                  fig_mc.add_trace(go.Scatter(
                      x=days_range, y=percentiles.loc[0.25].values,
                      name="25th %ile", line=dict(color="#6366F1", width=1, dash="dash"),
                      fill="tonexty", fillcolor="rgba(99,102,241,0.1)",
                  ))
                  fig_mc.add_trace(go.Scatter(
                      x=days_range, y=percentiles.loc[0.05].values,
                      name="5th %ile", line=dict(color="#EF4444", width=1, dash="dot"),
                  ))
                  fig_mc.add_hline(y=10000, line_dash="dot", line_color="#F59E0B", annotation_text="초기값")
                  fig_mc.update_layout(
                      title=f"GARCH Monte Carlo ({mc_sims}회 시뮬레이션, {mc_days}일)",
                      xaxis_title="거래일", yaxis_title="포트폴리오 가치",
                      template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                      height=400, legend=dict(orientation="h", y=1.12),
                  )
                  st.plotly_chart(fig_mc, use_container_width=True)

              # Insight
              st.markdown("### 💡 GARCH Insight")
              if garch_result.persistence > 0.97:
                  st.warning(f"⚠️ 변동성 지속성 {garch_result.persistence:.4f} — 변동성 충격이 오래 지속됩니다. 극단적 이벤트에 유의하세요.")
              if garch_result.current_vol > garch_result.long_run_vol * 1.5:
                  st.error(f"🔴 현재 변동성({garch_result.current_vol:.1%})이 장기 평균({garch_result.long_run_vol:.1%})의 1.5배 이상입니다.")
              elif garch_result.current_vol < garch_result.long_run_vol * 0.7:
                  st.success(f"🟢 현재 변동성({garch_result.current_vol:.1%})이 장기 평균 대비 낮은 수준입니다. 안정적 구간.")
              else:
                  st.info(f"🔵 현재 변동성({garch_result.current_vol:.1%})이 장기 평균({garch_result.long_run_vol:.1%}) 부근입니다.")

              final_median = mc_paths.iloc[:, -1].median()
              final_5pct = mc_paths.iloc[:, -1].quantile(0.05)
              st.info(f"📊 {mc_days}일 후 예상 가치 — 중앙값: ₩{final_median:,.0f} | 최악 5%: ₩{final_5pct:,.0f} (초기 ₩10,000 기준)")

      except Exception as e:
          st.error(f"GARCH 분석 실패: {e}")
          st.info("GARCH 분석을 위해 충분한 수익률 데이터가 필요합니다.")


# =============================================================================
# Tab 19: Black-Litterman
# =============================================================================

if tab19 is not None:
  with tab19:
      st.markdown("## 🏦 Black-Litterman 포트폴리오 최적화")
      st.markdown("""
      <div style="background: #1E293B; padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; color: #94A3B8;">
          시장 균형 기대수익률에 투자자의 <b>주관적 뷰(View)</b>를 결합하여 최적 포트폴리오를 도출합니다.<br>
          기존 Markowitz 최적화의 입력 민감도 문제를 완화하는 <b>베이지안 접근법</b>입니다.
      </div>
      """, unsafe_allow_html=True)

      try:
          tickers = sorted(portfolio_data.holdings.keys())

          # Views 입력 UI
          st.markdown("### 📝 투자자 뷰(View) 입력")
          st.markdown("각 종목에 대한 기대수익률 뷰를 설정하세요. 뷰가 없으면 시장 균형만 사용합니다.")

          views_list = []
          n_views = st.number_input("뷰 개수", min_value=0, max_value=5, value=0, key="bl_n_views")

          for vi in range(int(n_views)):
              st.markdown(f"**뷰 #{vi+1}**")
              v_cols = st.columns([2, 1, 1])
              with v_cols[0]:
                  view_ticker = st.selectbox(f"종목", tickers, key=f"bl_view_ticker_{vi}")
              with v_cols[1]:
                  view_return = st.number_input(f"기대수익률 (%)", value=10.0, step=1.0, key=f"bl_view_ret_{vi}")
              with v_cols[2]:
                  view_type = st.selectbox("유형", ["절대적", "상대적"], key=f"bl_view_type_{vi}")

              # Build P vector
              P_vec = [0.0] * len(tickers)
              ticker_idx = tickers.index(view_ticker)

              if view_type == "절대적":
                  P_vec[ticker_idx] = 1.0
              else:
                  # 상대적: view_ticker vs 나머지 균등
                  P_vec[ticker_idx] = 1.0
                  others = [i for i in range(len(tickers)) if i != ticker_idx]
                  if others:
                      for oi in others:
                          P_vec[oi] = -1.0 / len(others)

              views_list.append({"P": P_vec, "Q": view_return / 100.0})

          # BL Parameters
          with st.expander("⚙️ 모델 파라미터", expanded=False):
              bl_cols = st.columns(3)
              with bl_cols[0]:
                  risk_aversion = st.slider("위험회피 계수 (δ)", 1.0, 5.0, 2.5, 0.1, key="bl_delta")
              with bl_cols[1]:
                  tau = st.slider("뷰 불확실성 (τ)", 0.01, 0.20, 0.05, 0.01, key="bl_tau")
              with bl_cols[2]:
                  rf_rate = st.slider("무위험 이자율 (%)", 0.0, 8.0, 3.5, 0.5, key="bl_rf")

          with st.spinner("🏦 Black-Litterman 분석 중..."):
              # prices에서 Close 가격만 추출
              if isinstance(portfolio_data.prices.columns, pd.MultiIndex):
                  close_prices = portfolio_data.prices.xs("Close", axis=1, level=1) if "Close" in portfolio_data.prices.columns.get_level_values(1) else portfolio_data.prices.iloc[:, :len(tickers)]
              else:
                  close_prices = portfolio_data.prices

              bl_result = black_litterman_analysis(
                  prices=close_prices,
                  market_weights=portfolio_data.holdings,
                  views=views_list if views_list else None,
                  risk_aversion=risk_aversion,
                  tau=tau,
                  risk_free_rate=rf_rate / 100.0,
              )

          # Results
          st.markdown("### 📊 분석 결과")
          bl_kpi = st.columns(4)
          with bl_kpi[0]:
              prior_sr = np.dot(bl_result.prior_returns, bl_result.market_weights) / np.sqrt(np.dot(bl_result.market_weights, bl_result.posterior_cov @ bl_result.market_weights))
              st.metric("시장 균형 Sharpe", f"{prior_sr:.3f}")
          with bl_kpi[1]:
              post_sr = np.dot(bl_result.posterior_returns, bl_result.optimal_weights) / np.sqrt(np.dot(bl_result.optimal_weights, bl_result.posterior_cov @ bl_result.optimal_weights))
              st.metric("BL 최적 Sharpe", f"{post_sr:.3f}")
          with bl_kpi[2]:
              weight_change = np.sum(np.abs(bl_result.optimal_weights - bl_result.market_weights))
              st.metric("비중 변화 합계", f"{weight_change:.1%}")
          with bl_kpi[3]:
              n_active_views = len(views_list)
              st.metric("활성 뷰 수", f"{n_active_views}개")

          # Charts
          col_bl1, col_bl2 = st.columns(2)
          with col_bl1:
              fig_bl_comp = create_bl_comparison_chart(bl_result)
              st.plotly_chart(fig_bl_comp, use_container_width=True)
          with col_bl2:
              fig_bl_weights = create_bl_weights_chart(bl_result)
              st.plotly_chart(fig_bl_weights, use_container_width=True)

          if views_list:
              fig_bl_impact = create_bl_impact_chart(bl_result)
              st.plotly_chart(fig_bl_impact, use_container_width=True)

          # Insight
          st.markdown("### 💡 Black-Litterman Insight")
          max_ticker_idx = np.argmax(bl_result.optimal_weights)
          min_ticker_idx = np.argmin(bl_result.optimal_weights)
          st.info(f"🏦 BL 최적 포트폴리오 — 최대 비중: {bl_result.tickers[max_ticker_idx]} ({bl_result.optimal_weights[max_ticker_idx]:.1%}), 최소 비중: {bl_result.tickers[min_ticker_idx]} ({bl_result.optimal_weights[min_ticker_idx]:.1%})")

          if views_list:
              st.success(f"✅ {n_active_views}개의 투자자 뷰가 반영되어 시장 균형 대비 포트폴리오가 조정되었습니다.")
          else:
              st.info("ℹ️ 투자자 뷰가 없으므로 시장 균형(Market Implied) 기대수익률만 사용합니다. 뷰를 추가하면 더 개인화된 최적화가 가능합니다.")

      except Exception as e:
          st.error(f"Black-Litterman 분석 실패: {e}")
          st.info("분석을 위해 최소 2개 이상의 종목과 충분한 가격 데이터가 필요합니다.")


# =============================================================================
# Tab 20: Rebalance Signal
# =============================================================================

if tab20 is not None:
  with tab20:
      st.markdown("## 🔔 AI 리밸런싱 시그널")
      st.markdown("""
      <div style="background: #1E293B; padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; color: #94A3B8;">
          시장 국면, 스킬 점수, 변동성, 추적오차, 집중도를 종합하여<br>
          <b>리밸런싱 긴급도(0-100)</b>와 방향(방어적/공격적/유지)을 제시합니다.
      </div>
      """, unsafe_allow_html=True)

      try:
          # Prepare inputs
          skills_dict = profile.to_dict()

          # Detect regime for signal
          from regime_detection import detect_regimes, get_current_regime
          regime_df = detect_regimes(portfolio_data.benchmark_returns, window=63)
          current_regime = get_current_regime(regime_df)

          regime_info = {
              "current_regime": current_regime.get("regime", "Sideways"),
              "days_in_regime": current_regime.get("days_in_regime", 0),
              "regime_probability": current_regime.get("regime_probability", 0.5),
          }

          # Portfolio weights as Series
          weights_series = pd.Series(portfolio_data.holdings)

          with st.spinner("🔔 리밸런싱 신호 분석 중..."):
              rebal_signal = generate_rebalance_signal(
                  skills_dict=skills_dict,
                  regime_info=regime_info,
                  portfolio_returns=portfolio_data.returns,
                  benchmark_returns=portfolio_data.benchmark_returns,
                  weights=weights_series,
              )

          # Signal Gauge
          col_sig1, col_sig2 = st.columns([1, 2])
          with col_sig1:
              fig_gauge = create_signal_gauge(rebal_signal)
              st.plotly_chart(fig_gauge, use_container_width=True)

              # Direction indicator
              dir_colors = {"방어적": "#EF4444", "공격적": "#10B981", "유지": "#F59E0B"}
              dir_icons = {"방어적": "🛡️", "공격적": "⚔️", "유지": "⚖️"}
              st.markdown(f"""
              <div style="background: #1E293B; padding: 16px; border-radius: 12px; text-align: center;
                          border: 2px solid {dir_colors.get(rebal_signal.direction, '#6B7280')};">
                  <span style="font-size: 2rem;">{dir_icons.get(rebal_signal.direction, '⚖️')}</span><br>
                  <span style="font-size: 1.4rem; font-weight: 700; color: {dir_colors.get(rebal_signal.direction, '#F8FAFC')};">
                      {rebal_signal.direction}
                  </span>
              </div>
              """, unsafe_allow_html=True)

          with col_sig2:
              # Interpretation
              interpretation = get_signal_interpretation(rebal_signal)
              st.markdown(f"""
              <div style="background: #1E293B; padding: 18px; border-radius: 12px; margin-bottom: 16px;">
                  <h4 style="color: #6366F1; margin-top: 0;">📋 AI 해석</h4>
                  <p style="color: #F8FAFC; font-size: 1.05rem; line-height: 1.6;">{interpretation}</p>
              </div>
              """, unsafe_allow_html=True)

              # Sub-scores breakdown
              st.markdown("#### 📊 세부 점수 분해")
              sub_data = {
                  "항목": ["시장 국면", "스킬 점수", "변동성/드로다운", "추적오차", "긴급도 (종합)"],
                  "점수": [
                      f"{rebal_signal.regime_score:.1f}",
                      f"{rebal_signal.skill_score:.1f}",
                      f"{rebal_signal.volatility_score:.1f}",
                      f"{rebal_signal.tracking_error_score:.1f}",
                      f"**{rebal_signal.urgency:.1f}**",
                  ],
                  "가중치": ["30%", "25%", "20%", "15%", "—"],
              }
              st.dataframe(pd.DataFrame(sub_data), use_container_width=True, hide_index=True)

          # Reasons & Actions
          if rebal_signal.reasons:
              st.markdown("### ⚠️ 리밸런싱 사유")
              for reason in rebal_signal.reasons:
                  st.markdown(f"- {reason}")

          if rebal_signal.suggested_actions:
              st.markdown("### ✅ 추천 액션")
              for action in rebal_signal.suggested_actions:
                  st.markdown(f"- {action}")

          # Current regime context
          st.markdown("### 🔀 현재 시장 국면")
          regime_name = regime_info["current_regime"]
          regime_color = {"Bull": "#10B981", "Bear": "#EF4444", "Sideways": "#F59E0B"}.get(regime_name, "#6B7280")
          st.markdown(f"""
          <div style="background: #1E293B; padding: 12px; border-radius: 8px; border-left: 4px solid {regime_color};">
              <b style="color: {regime_color};">{regime_name}</b> 국면 진행 중
          </div>
          """, unsafe_allow_html=True)

      except Exception as e:
          st.error(f"리밸런싱 시그널 분석 실패: {e}")
          st.info("분석을 위해 충분한 수익률 데이터와 스킬 프로파일이 필요합니다.")


# =============================================================================
# Tab 21: Portfolio DNA
# =============================================================================

if tab21 is not None:
  with tab21:
      st.markdown("## 🧬 포트폴리오 DNA 핑거프린트")
      st.markdown("""
      <div style="background: #1E293B; padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; color: #94A3B8;">
          6가지 스킬 + 스타일 + 팩터 + 구조적 특성을 <b>12차원 레이더</b>로 시각화하여<br>
          포트폴리오의 고유한 DNA 핑거프린트와 아키타입을 도출합니다.
      </div>
      """, unsafe_allow_html=True)

      try:
          skills_dict = profile.to_dict()

          # Style info — try to get from Investment Style tab or estimate
          try:
              from investment_style import analyze_portfolio_style
              style_res = analyze_portfolio_style(
                  returns=portfolio_data.returns,
                  benchmark_returns=portfolio_data.benchmark_returns,
                  prices=portfolio_data.prices,
                  holdings=portfolio_data.holdings,
              )
              style_info = {
                  "style": style_res.primary_style.lower() if hasattr(style_res, "primary_style") else "blend",
                  "value_score": getattr(style_res, "value_score", 50),
                  "growth_score": getattr(style_res, "growth_score", 50),
              }
          except Exception:
              style_info = {"style": "blend", "value_score": 50, "growth_score": 50}

          # Factor info — try to get from Factor Attribution
          try:
              from factor_engine import run_factor_analysis
              factor_res = run_factor_analysis(portfolio_data.returns, portfolio_data.benchmark_returns)
              factor_info = {
                  "momentum": min(100, max(0, 50 + factor_res.beta_market * 20)),
                  "quality": min(100, max(0, 50 + factor_res.r_squared * 50)),
                  "dividend": 50,
              }
          except Exception:
              factor_info = {"momentum": 50, "quality": 50, "dividend": 50}

          # Volatility regime
          vol = portfolio_data.returns.std() * np.sqrt(252)
          if vol > 0.25:
              vol_regime = "high"
          elif vol > 0.15:
              vol_regime = "medium"
          else:
              vol_regime = "low"

          weights_series = pd.Series(portfolio_data.holdings)

          with st.spinner("🧬 DNA 핑거프린트 생성 중..."):
              dna = generate_dna(
                  skills_dict=skills_dict,
                  style_info=style_info,
                  factor_info=factor_info,
                  weights=weights_series,
                  vol_regime=vol_regime,
              )

          # Archetype
          archetype = get_dna_archetype(dna)
          st.markdown(f"""
          <div style="background: linear-gradient(135deg, #1E293B, #334155); padding: 24px; border-radius: 16px;
                      border: 2px solid #6366F1; text-align: center; margin-bottom: 20px;">
              <span style="font-size: 2.5rem;">🧬</span><br>
              <span style="font-size: 1.6rem; font-weight: 700; color: #F8FAFC;">
                  {archetype}
              </span><br>
              <span style="color: #94A3B8; font-size: 0.95rem;">DNA Hash: {dna.dna_hash[:12]}...</span>
          </div>
          """, unsafe_allow_html=True)

          # DNA Fingerprint Chart
          fig_dna = create_dna_fingerprint(dna)
          st.plotly_chart(fig_dna, use_container_width=True)

          # Dimension Details
          st.markdown("### 📋 DNA 차원별 상세")
          dim_data = []
          for skill_name, score in dna.skills.items():
              dim_data.append({"차원": skill_name, "점수": f"{score:.1f}", "카테고리": "스킬"})
          dim_data.append({"차원": "Value (가치)", "점수": f"{dna.value_score:.1f}", "카테고리": "스타일"})
          dim_data.append({"차원": "Growth (성장)", "점수": f"{dna.growth_score:.1f}", "카테고리": "스타일"})
          dim_data.append({"차원": "Concentration (집중도)", "점수": f"{dna.concentration:.2f}", "카테고리": "구조"})
          dim_data.append({"차원": "Sector Diversity (섹터다양)", "점수": f"{dna.sector_diversity:.2f}", "카테고리": "구조"})
          dim_data.append({"차원": "Volatility Regime", "점수": dna.volatility_regime, "카테고리": "레짐"})
          st.dataframe(pd.DataFrame(dim_data), use_container_width=True, hide_index=True)

          # Insight
          st.markdown("### 💡 DNA Insight")

          # --- 아키타입 요약 (항상 표시) ---
          st.success(f"🧬 **포트폴리오 아키타입: {dna.archetype}**")

          # --- 스킬 기반 인사이트 ---
          if dna.skills:
              avg_skill = np.mean(list(dna.skills.values()))
              best_skill = max(dna.skills, key=dna.skills.get)
              worst_skill = min(dna.skills, key=dna.skills.get)

              # 전체 스킬 평균 등급
              if avg_skill >= 70:
                  st.info(f"⭐ 평균 스킬 점수 **{avg_skill:.0f}점** — 전반적으로 우수한 포트폴리오 운용 역량입니다.")
              elif avg_skill >= 50:
                  st.info(f"📊 평균 스킬 점수 **{avg_skill:.0f}점** — 보통 수준의 포트폴리오 역량입니다.")
              else:
                  st.info(f"📉 평균 스킬 점수 **{avg_skill:.0f}점** — 포트폴리오 역량 개선이 필요합니다.")

              # 최강/최약 스킬
              st.markdown(
                  f"- 💪 **가장 강한 역량**: {best_skill} ({dna.skills[best_skill]:.0f}점)\n"
                  f"- 🔧 **개선 필요 역량**: {worst_skill} ({dna.skills[worst_skill]:.0f}점)"
              )

          # --- 집중도 인사이트 ---
          if dna.concentration > 0.5:
              st.warning(f"🎯 집중도 {dna.concentration:.2f} — 포트폴리오가 매우 소수 종목에 집중되어 있어 분산 위험이 높습니다.")
          elif dna.concentration > 0.3:
              st.warning(f"🎯 집중도 {dna.concentration:.2f} — 포트폴리오가 소수 종목에 다소 집중되어 있습니다.")
          else:
              st.info(f"✅ 집중도 {dna.concentration:.2f} — 포트폴리오가 적절히 분산되어 있습니다.")

          # --- 섹터 다양성 인사이트 ---
          if dna.sector_diversity > 0.7:
              st.info(f"🌐 섹터 다양성 {dna.sector_diversity:.2f} — 다양한 섹터에 고르게 투자 중입니다.")
          elif dna.sector_diversity > 0.4:
              st.info(f"📂 섹터 다양성 {dna.sector_diversity:.2f} — 일부 섹터에 편향되어 있습니다.")
          else:
              st.warning(f"⚠️ 섹터 다양성 {dna.sector_diversity:.2f} — 특정 섹터에 과도하게 집중되어 있습니다.")

          # --- 스타일 인사이트 ---
          if dna.value_score > 70:
              st.info(f"📈 가치 성향 **{dna.value_score:.0f}** — 저평가 종목 중심의 포트폴리오입니다.")
          elif dna.growth_score > 70:
              st.info(f"🚀 성장 성향 **{dna.growth_score:.0f}** — 고성장 종목 중심의 포트폴리오입니다.")
          elif dna.value_score > 50 and dna.growth_score > 50:
              st.info(f"⚖️ 밸런스형 — 가치({dna.value_score:.0f}) · 성장({dna.growth_score:.0f}) 균형 잡힌 스타일입니다.")
          else:
              st.info(f"🔄 가치 {dna.value_score:.0f} · 성장 {dna.growth_score:.0f} — 뚜렷한 스타일 편향이 적은 중립형입니다.")

          # --- 변동성 레짐 인사이트 ---
          vol_labels = {"low": ("🛡️ 저변동성", "방어적 성격"), "medium": ("📊 중변동성", "균형적 성격"), "high": ("🔥 고변동성", "공격적 성격")}
          vol_icon, vol_desc = vol_labels.get(dna.volatility_regime, ("❓", "알 수 없음"))
          st.info(f"{vol_icon} 변동성 레짐 **{dna.volatility_regime.upper()}** — {vol_desc}의 포트폴리오입니다.")

          # --- DNA 해시 (핑거프린트 ID) ---
          if dna.dna_hash:
              st.caption(f"🔑 DNA Hash: `{dna.dna_hash}`")

      except Exception as e:
          st.error(f"DNA 핑거프린트 분석 실패: {e}")
          st.info("분석을 위해 스킬 프로파일과 포트폴리오 데이터가 필요합니다.")


# =============================================================================
# Tab 22: Backtest
# =============================================================================

if tab22 is not None:
  with tab22:
      st.markdown("## ⏪ 백테스트 엔진")
      st.markdown("""
      <div style="background: #1E293B; padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; color: #94A3B8;">
          현재 포트폴리오 비중으로 과거 다양한 시점부터 투자했다면 어떤 성과가 나왔을지 시뮬레이션합니다.<br>
          <b>Single Backtest</b>와 <b>Rolling Backtest</b> 두 가지 모드를 지원합니다.
      </div>
      """, unsafe_allow_html=True)

      try:
          # Extract close prices
          if isinstance(portfolio_data.prices.columns, pd.MultiIndex):
              bt_close = portfolio_data.prices.xs("Close", axis=1, level=1) if "Close" in portfolio_data.prices.columns.get_level_values(1) else portfolio_data.prices.iloc[:, :len(portfolio_data.holdings)]
          else:
              bt_close = portfolio_data.prices.copy()

          # Benchmark prices — reconstruct from benchmark returns
          bench_cum = (1 + portfolio_data.benchmark_returns).cumprod() * 10000
          bench_prices = bench_cum

          bt_mode = st.radio("백테스트 모드", ["Single Backtest", "Rolling Backtest"], horizontal=True, key="bt_mode")

          if bt_mode == "Single Backtest":
              bt_cols = st.columns(2)
              available_dates = bt_close.index
              with bt_cols[0]:
                  bt_start = st.date_input("시작일", value=available_dates[0], key="bt_start")
              with bt_cols[1]:
                  bt_end = st.date_input("종료일", value=available_dates[-1], key="bt_end")

              with st.spinner("⏪ 백테스트 실행 중..."):
                  bt_result = run_backtest(
                      prices_df=bt_close,
                      weights=portfolio_data.holdings,
                      benchmark_prices=bench_prices,
                      start_date=str(bt_start),
                      end_date=str(bt_end),
                  )

              # KPI
              bt_kpi = st.columns(5)
              with bt_kpi[0]:
                  st.metric("총 수익률", f"{bt_result.total_return:.1%}")
              with bt_kpi[1]:
                  st.metric("연환산 수익률", f"{bt_result.annualized_return:.1%}")
              with bt_kpi[2]:
                  st.metric("최대 낙폭", f"{bt_result.max_drawdown:.1%}")
              with bt_kpi[3]:
                  st.metric("Sharpe Ratio", f"{bt_result.sharpe_ratio:.3f}")
              with bt_kpi[4]:
                  st.metric("Alpha", f"{bt_result.alpha:.1%}")

              # Charts
              col_bt1, col_bt2 = st.columns(2)
              with col_bt1:
                  # Benchmark cumulative curve for chart
                  bench_sub = bench_prices.loc[bt_result.cumulative_curve.index[0]:bt_result.cumulative_curve.index[-1]] if hasattr(bench_prices, 'loc') else None
                  if bench_sub is not None and len(bench_sub) > 0:
                      bench_curve = bench_sub / bench_sub.iloc[0]
                  else:
                      bench_curve = None
                  fig_bt_cum = create_backtest_cumulative_chart(bt_result, benchmark_curve=bench_curve)
                  st.plotly_chart(fig_bt_cum, use_container_width=True)
              with col_bt2:
                  fig_bt_heat = create_backtest_monthly_heatmap(bt_result)
                  st.plotly_chart(fig_bt_heat, use_container_width=True)

              # Additional stats
              st.markdown("### 📋 상세 통계")
              stat_cols = st.columns(4)
              with stat_cols[0]:
                  st.metric("변동성", f"{bt_result.volatility:.1%}")
              with stat_cols[1]:
                  st.metric("승률", f"{bt_result.win_rate:.1%}")
              with stat_cols[2]:
                  st.metric("최고 월간 수익", f"{bt_result.best_month:.1%}")
              with stat_cols[3]:
                  st.metric("최악 월간 수익", f"{bt_result.worst_month:.1%}")

          else:  # Rolling Backtest
              roll_cols = st.columns(2)
              with roll_cols[0]:
                  window_months = st.slider("윈도우 기간 (월)", 3, 24, 12, key="bt_window")
              with roll_cols[1]:
                  step_months = st.slider("이동 간격 (월)", 1, 6, 3, key="bt_step")

              with st.spinner("⏪ 롤링 백테스트 실행 중..."):
                  rolling_results = run_rolling_backtests(
                      prices_df=bt_close,
                      weights=portfolio_data.holdings,
                      benchmark_prices=bench_prices,
                      window_months=window_months,
                      step_months=step_months,
                  )

              if not rolling_results:
                  st.warning("롤링 백테스트 결과가 없습니다. 데이터 기간이 충분하지 않을 수 있습니다.")
              else:
                  st.success(f"✅ {len(rolling_results)}개 기간의 백테스트 완료")

                  # Summary stats
                  returns_list = [r.annualized_return for r in rolling_results]
                  sharpes_list = [r.sharpe_ratio for r in rolling_results]
                  mdds_list = [r.max_drawdown for r in rolling_results]

                  roll_kpi = st.columns(4)
                  with roll_kpi[0]:
                      st.metric("평균 연환산 수익률", f"{np.mean(returns_list):.1%}")
                  with roll_kpi[1]:
                      st.metric("평균 Sharpe", f"{np.mean(sharpes_list):.3f}")
                  with roll_kpi[2]:
                      st.metric("평균 MDD", f"{np.mean(mdds_list):.1%}")
                  with roll_kpi[3]:
                      st.metric("양수 수익 비율", f"{np.mean([1 for r in returns_list if r > 0]) / max(len(returns_list), 1):.0%}")

                  fig_rolling = create_rolling_performance_chart(rolling_results)
                  st.plotly_chart(fig_rolling, use_container_width=True)

                  # Results table
                  st.markdown("### 📋 기간별 결과")
                  roll_df = pd.DataFrame([{
                      "시작일": r.start_date,
                      "종료일": r.end_date,
                      "총수익률": f"{r.total_return:.1%}",
                      "연환산": f"{r.annualized_return:.1%}",
                      "MDD": f"{r.max_drawdown:.1%}",
                      "Sharpe": f"{r.sharpe_ratio:.3f}",
                      "Alpha": f"{r.alpha:.1%}",
                  } for r in rolling_results])
                  st.dataframe(roll_df, use_container_width=True, hide_index=True)

      except Exception as e:
          st.error(f"백테스트 실행 실패: {e}")
          st.info("백테스트를 위해 충분한 가격 데이터가 필요합니다.")


# =============================================================================
# Tab 23: Performance+
# =============================================================================

if tab23 is not None:
  with tab23:
      st.markdown("## 📋 Performance+ (고급 성과지표)")
      st.markdown("""
      <div style="background: #1E293B; padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; color: #94A3B8;">
          <b>Information Ratio</b>, <b>Tracking Error</b>, <b>Up/Down Capture Ratio</b> 등<br>
          벤치마크 대비 심화 성과지표를 분석합니다.
      </div>
      """, unsafe_allow_html=True)

      try:
          with st.spinner("📋 고급 성과지표 계산 중..."):
              perf_metrics = calc_all_metrics(
                  portfolio_returns=portfolio_data.returns,
                  benchmark_returns=portfolio_data.benchmark_returns,
              )

          # KPI Row 1
          perf_kpi1 = st.columns(4)
          with perf_kpi1[0]:
              ir_color = "normal" if perf_metrics.information_ratio > 0 else "inverse"
              st.metric("Information Ratio", f"{perf_metrics.information_ratio:.3f}", delta_color=ir_color)
          with perf_kpi1[1]:
              st.metric("Tracking Error", f"{perf_metrics.tracking_error:.1%}")
          with perf_kpi1[2]:
              st.metric("Up Capture", f"{perf_metrics.up_capture_ratio:.2f}", delta="상승장 참여" if perf_metrics.up_capture_ratio > 1 else "상승장 부진")
          with perf_kpi1[3]:
              dc_delta = "하락 방어" if perf_metrics.down_capture_ratio < 1 else "하락 동조"
              st.metric("Down Capture", f"{perf_metrics.down_capture_ratio:.2f}", delta=dc_delta, delta_color="normal" if perf_metrics.down_capture_ratio < 1 else "inverse")

          # KPI Row 2
          perf_kpi2 = st.columns(4)
          with perf_kpi2[0]:
              st.metric("포트폴리오 수익률", f"{perf_metrics.portfolio_return:.1%}")
          with perf_kpi2[1]:
              st.metric("벤치마크 수익률", f"{perf_metrics.benchmark_return:.1%}")
          with perf_kpi2[2]:
              st.metric("Beta", f"{perf_metrics.beta:.3f}")
          with perf_kpi2[3]:
              st.metric("상관계수", f"{perf_metrics.correlation:.3f}")

          # Charts
          col_perf1, col_perf2 = st.columns(2)
          with col_perf1:
              fig_capture = create_capture_ratio_chart(perf_metrics)
              st.plotly_chart(fig_capture, use_container_width=True)
          with col_perf2:
              fig_perf_comp = create_performance_comparison_chart(perf_metrics)
              st.plotly_chart(fig_perf_comp, use_container_width=True)

          # Detailed Metrics Table
          st.markdown("### 📊 지표 비교 테이블")
          metrics_table = pd.DataFrame({
              "지표": ["연환산 수익률", "연환산 변동성", "Sharpe Ratio", "Active Return", "Tracking Error", "Information Ratio", "Beta", "상관계수", "Up Capture", "Down Capture"],
              "포트폴리오": [
                  f"{perf_metrics.portfolio_return:.2%}",
                  f"{perf_metrics.portfolio_volatility:.2%}",
                  f"{perf_metrics.sharpe_ratio:.3f}",
                  f"{perf_metrics.active_return:.2%}",
                  f"{perf_metrics.tracking_error:.2%}",
                  f"{perf_metrics.information_ratio:.3f}",
                  f"{perf_metrics.beta:.3f}",
                  f"{perf_metrics.correlation:.3f}",
                  f"{perf_metrics.up_capture_ratio:.2f}",
                  f"{perf_metrics.down_capture_ratio:.2f}",
              ],
              "벤치마크": [
                  f"{perf_metrics.benchmark_return:.2%}",
                  f"{perf_metrics.benchmark_volatility:.2%}",
                  f"{perf_metrics.benchmark_sharpe:.3f}",
                  "—",
                  "—",
                  "—",
                  "1.000",
                  "1.000",
                  "1.00",
                  "1.00",
              ],
          })
          st.dataframe(metrics_table, use_container_width=True, hide_index=True)

          # Insight
          st.markdown("### 💡 Performance+ Insight")
          if perf_metrics.information_ratio > 0.5:
              st.success(f"🌟 Information Ratio {perf_metrics.information_ratio:.3f} — 우수한 활성 수익 대비 리스크 관리입니다.")
          elif perf_metrics.information_ratio > 0:
              st.info(f"📊 Information Ratio {perf_metrics.information_ratio:.3f} — 양의 초과수익이 있으나 개선 여지가 있습니다.")
          else:
              st.warning(f"⚠️ Information Ratio {perf_metrics.information_ratio:.3f} — 벤치마크 대비 초과수익이 부진합니다.")

          if perf_metrics.up_capture_ratio > 1.1 and perf_metrics.down_capture_ratio < 0.9:
              st.success("🎯 이상적인 Capture 프로필 — 상승장은 더 많이 참여하고, 하락장은 덜 참여합니다.")
          elif perf_metrics.up_capture_ratio < 0.9 and perf_metrics.down_capture_ratio > 1.1:
              st.error("❌ 비대칭 리스크 — 상승장은 놓치고, 하락장은 더 크게 손실을 봅니다.")

          if perf_metrics.tracking_error > 0.15:
              st.warning(f"📏 Tracking Error {perf_metrics.tracking_error:.1%} — 벤치마크와 크게 다른 포트폴리오입니다. 의도된 Active 전략인지 확인하세요.")

      except Exception as e:
          st.error(f"Performance+ 분석 실패: {e}")
          st.info("분석을 위해 포트폴리오와 벤치마크의 수익률 데이터가 필요합니다.")


# =============================================================================
# 푸터
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 20px;">
    <b>InvestScope</b> — Investment Skills Dashboard v6.0<br>
    6-Skills Framework · Carhart 4-Factor · What-If · AI Commentary · Skills Evolution<br>
    Efficient Frontier (Ledoit-Wolf) · Stress Test · Geopolitical Engine · Correlation Network<br>
    Investment Style · Risk Contribution · Market Events · Regime Detection · Tail Risk (CF-VaR)<br>
    GARCH Volatility · Black-Litterman · Rebalance Signal · Portfolio DNA · Backtest · Performance+<br>
    🇰🇷 한국 주식 지원 | 23-Tab Full Analysis | Built with Streamlit & Plotly<br>
    ⚠️ 본 대시보드는 교육/분석 목적이며, 투자 조언을 제공하지 않습니다.
</div>
""", unsafe_allow_html=True)
