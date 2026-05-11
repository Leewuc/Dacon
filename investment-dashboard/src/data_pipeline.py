"""
Data Pipeline: 투자 데이터 수집, 전처리, 포트폴리오 구성

지원 데이터 소스:
    - yfinance: 글로벌 주식/ETF
    - CSV 업로드: 사용자 포트폴리오
    - 샘플 데이터: 데모용 내장 데이터
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# =============================================================================
# 데이터 모델
# =============================================================================

@dataclass
class PortfolioData:
    """포트폴리오 전체 데이터를 담는 컨테이너"""
    name: str
    holdings: Dict[str, float]         # {ticker: weight}
    prices: pd.DataFrame                # OHLCV 멀티인덱스
    returns: pd.Series                  # 일간 포트폴리오 수익률
    benchmark_returns: pd.Series        # 벤치마크 수익률
    returns_by_ticker: Dict[str, float] # 종목별 누적 수익률
    sector_map: Dict[str, str]          # 섹터 매핑
    start_date: pd.Timestamp
    end_date: pd.Timestamp


# =============================================================================
# 데이터 수집
# =============================================================================

def fetch_price_data(
    tickers: List[str],
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    yfinance로 복수 종목의 가격 데이터 수집

    Parameters:
        tickers: 종목 코드 리스트
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (None이면 오늘)

    Returns:
        MultiIndex DataFrame (Date x (OHLCV, Ticker))
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance가 설치되지 않았습니다. pip install yfinance")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"데이터 수집 실패: {tickers}")

    # 결측치 처리 (주말/공휴일)
    data = data.ffill().bfill()

    return data


def fetch_sector_info(tickers: List[str]) -> Dict[str, str]:
    """yfinance에서 종목별 섹터 정보 수집"""
    sector_map = {}
    if not HAS_YFINANCE:
        return sector_map

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector_map[ticker] = info.get("sector", "Unknown")
        except Exception:
            sector_map[ticker] = "Unknown"

    return sector_map


# =============================================================================
# 포트폴리오 수익률 계산
# =============================================================================

def calc_portfolio_returns(
    prices: pd.DataFrame,
    weights: Dict[str, float],
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    포트폴리오 가중 일간 수익률 계산

    Returns:
        (portfolio_daily_returns, {ticker: cumulative_return})
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()  # 정규화

    # Close 가격 추출
    if isinstance(prices.columns, pd.MultiIndex):
        close_prices = prices["Close"][tickers]
    else:
        close_prices = prices[["Close"]].copy()
        close_prices.columns = tickers

    # 일간 수익률
    daily_returns = close_prices.pct_change().dropna()

    # 가중 포트폴리오 수익률
    portfolio_returns = daily_returns.dot(w)
    portfolio_returns.name = "portfolio"

    # 종목별 누적 수익률
    cumulative = (1 + daily_returns).prod() - 1
    returns_by_ticker = {t: float(cumulative[t]) for t in tickers}

    return portfolio_returns, returns_by_ticker


def calc_benchmark_returns(
    benchmark_ticker: str = "SPY",
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
) -> pd.Series:
    """벤치마크 일간 수익률 계산"""
    data = fetch_price_data([benchmark_ticker], start_date, end_date)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][benchmark_ticker]
    else:
        close = data["Close"]

    returns = close.pct_change().dropna()
    returns.name = "benchmark"
    return returns


# =============================================================================
# CSV 파싱
# =============================================================================

def parse_portfolio_csv(uploaded_file) -> Dict[str, float]:
    """
    사용자 업로드 CSV에서 포트폴리오 비중 파싱

    지원 형식:
        ticker,weight          (예: AAPL,0.3)
        ticker,shares,price    (예: AAPL,100,150.0)
        ticker,amount           (예: AAPL,15000)
    """
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    if "weight" in df.columns and "ticker" in df.columns:
        weights = dict(zip(df["ticker"], df["weight"]))
    elif "shares" in df.columns and "price" in df.columns and "ticker" in df.columns:
        df["value"] = df["shares"] * df["price"]
        total = df["value"].sum()
        weights = dict(zip(df["ticker"], df["value"] / total))
    elif "amount" in df.columns and "ticker" in df.columns:
        total = df["amount"].sum()
        weights = dict(zip(df["ticker"], df["amount"] / total))
    else:
        raise ValueError(
            "CSV 형식 오류. 지원 형식:\n"
            "1) ticker, weight\n"
            "2) ticker, shares, price\n"
            "3) ticker, amount"
        )

    # 정규화
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    return weights


# =============================================================================
# 샘플 데이터 생성
# =============================================================================

def generate_sample_portfolio() -> Dict[str, float]:
    """데모용 샘플 포트폴리오"""
    return {
        "AAPL": 0.20,
        "MSFT": 0.15,
        "GOOGL": 0.12,
        "AMZN": 0.10,
        "NVDA": 0.10,
        "JPM": 0.08,
        "JNJ": 0.07,
        "V": 0.06,
        "PG": 0.05,
        "XOM": 0.04,
        "GLD": 0.03,
    }


def generate_sample_sector_map() -> Dict[str, str]:
    """샘플 섹터 매핑"""
    return {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Communication Services",
        "AMZN": "Consumer Cyclical",
        "NVDA": "Technology",
        "JPM": "Financial Services",
        "JNJ": "Healthcare",
        "V": "Financial Services",
        "PG": "Consumer Defensive",
        "XOM": "Energy",
        "GLD": "Commodities",
    }


def generate_synthetic_data(
    n_days: int = 504,
    n_assets: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    yfinance 없이 사용 가능한 합성 데이터 생성
    실제 시장과 유사한 통계적 특성을 가진 가격 데이터
    """
    np.random.seed(seed)

    tickers = [
        "STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D", "STOCK_E",
        "STOCK_F", "STOCK_G", "STOCK_H", "STOCK_I", "STOCK_J",
    ][:n_assets]

    dates = pd.bdate_range(end=datetime.now(), periods=n_days)

    # 상관관계가 있는 수익률 생성
    mean_returns = np.random.uniform(0.0002, 0.0008, n_assets)
    vols = np.random.uniform(0.01, 0.03, n_assets)

    # 상관관계 행렬 생성 (Cholesky decomposition)
    random_corr = np.random.uniform(0.1, 0.6, (n_assets, n_assets))
    random_corr = (random_corr + random_corr.T) / 2
    np.fill_diagonal(random_corr, 1.0)

    # Positive definite 보장
    eigenvalues, eigenvectors = np.linalg.eigh(random_corr)
    eigenvalues = np.maximum(eigenvalues, 0.01)
    random_corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    D = np.diag(1 / np.sqrt(np.diag(random_corr)))
    random_corr = D @ random_corr @ D

    L = np.linalg.cholesky(random_corr)
    uncorr_returns = np.random.normal(0, 1, (n_days, n_assets))
    corr_returns = uncorr_returns @ L.T * vols + mean_returns

    # 가끔 큰 하락 (fat tails)
    crash_days = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)
    corr_returns[crash_days] -= np.random.uniform(0.02, 0.05, (len(crash_days), n_assets))

    # 가격으로 변환
    prices_data = np.exp(np.cumsum(corr_returns, axis=0)) * 100

    # OHLCV 생성
    records = []
    for i, ticker in enumerate(tickers):
        for j, date in enumerate(dates):
            close = prices_data[j, i]
            high = close * (1 + abs(np.random.normal(0, 0.005)))
            low = close * (1 - abs(np.random.normal(0, 0.005)))
            open_p = low + (high - low) * np.random.random()
            volume = int(np.random.lognormal(15, 1))
            records.append({
                "Date": date,
                "Ticker": ticker,
                "Open": open_p,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            })

    df = pd.DataFrame(records)
    df = df.pivot_table(index="Date", columns="Ticker", values=["Open", "High", "Low", "Close", "Volume"])

    # 균등 비중
    weights = {t: 1.0 / n_assets for t in tickers}

    return df, weights


# =============================================================================
# 통합 데이터 로더
# =============================================================================

def load_portfolio_data(
    weights: Optional[Dict[str, float]] = None,
    benchmark_ticker: str = "SPY",
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    use_synthetic: bool = False,
) -> PortfolioData:
    """
    모든 데이터 소스를 통합하여 PortfolioData 객체 생성

    Parameters:
        weights: 포트폴리오 비중 (None이면 샘플)
        benchmark_ticker: 벤치마크 티커
        start_date: 분석 시작일
        end_date: 분석 종료일
        use_synthetic: True면 합성 데이터 사용 (오프라인)
    """
    if use_synthetic:
        prices, synth_weights = generate_synthetic_data()
        weights = weights or synth_weights
        tickers = list(weights.keys())

        port_returns, returns_by_ticker = calc_portfolio_returns(prices, weights)

        # 벤치마크 = 균등 가중 포트폴리오
        bench_w = {t: 1.0 / len(tickers) for t in tickers}
        bench_returns, _ = calc_portfolio_returns(prices, bench_w)

        sector_map = {t: f"Sector_{i % 5}" for i, t in enumerate(tickers)}

        return PortfolioData(
            name="Synthetic Portfolio",
            holdings=weights,
            prices=prices,
            returns=port_returns,
            benchmark_returns=bench_returns,
            returns_by_ticker=returns_by_ticker,
            sector_map=sector_map,
            start_date=prices.index[0],
            end_date=prices.index[-1],
        )

    # 실제 데이터
    if weights is None:
        weights = generate_sample_portfolio()

    tickers = list(weights.keys())

    # 가격 데이터 수집
    prices = fetch_price_data(tickers, start_date, end_date)

    # 포트폴리오 수익률
    port_returns, returns_by_ticker = calc_portfolio_returns(prices, weights)

    # 벤치마크 수익률
    bench_returns = calc_benchmark_returns(benchmark_ticker, start_date, end_date)

    # 인덱스 정렬
    common_idx = port_returns.index.intersection(bench_returns.index)
    port_returns = port_returns.reindex(common_idx)
    bench_returns = bench_returns.reindex(common_idx)

    # 섹터 정보
    sector_map = fetch_sector_info(tickers)

    return PortfolioData(
        name="My Portfolio",
        holdings=weights,
        prices=prices,
        returns=port_returns,
        benchmark_returns=bench_returns,
        returns_by_ticker=returns_by_ticker,
        sector_map=sector_map,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date or datetime.now().strftime("%Y-%m-%d")),
    )
