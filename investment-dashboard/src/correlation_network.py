"""
Correlation Network: 종목 간 상관관계 네트워크 시각화
- 노드: 종목 (크기=비중)
- 엣지: 상관계수 (두께=|상관계수|, 색상=양/음의 상관)
- Force-directed layout 근사 (spring layout)
- 클러스터 탐지 (threshold-based)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NetworkData:
    """네트워크 그래프 데이터"""
    nodes: List[Dict]          # [{ticker, x, y, size, sector, cluster}]
    edges: List[Dict]          # [{source, target, correlation, color, width}]
    correlation_matrix: pd.DataFrame
    clusters: Dict[int, List[str]]  # cluster_id -> [tickers]
    avg_correlation: float


def compute_correlation_network(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    sector_map: Dict[str, str],
    corr_threshold: float = 0.3,
) -> NetworkData:
    """
    상관관계 네트워크 계산

    1. 일간 수익률에서 상관행렬 계산
    2. |correlation| > threshold인 쌍만 엣지로 생성
    3. Spring layout 알고리즘으로 노드 위치 결정
    4. Simple threshold-based 클러스터링

    Args:
        prices: DataFrame with tickers as columns, dates as index
        weights: Dict[ticker] -> weight
        sector_map: Dict[ticker] -> sector
        corr_threshold: Minimum correlation magnitude to include edge

    Returns:
        NetworkData with nodes, edges, clusters, and positions
    """
    # MultiIndex columns 처리 (OHLCV → Close만 추출)
    close_prices = None
    if isinstance(prices.columns, pd.MultiIndex):
        level0_vals = prices.columns.get_level_values(0).unique().tolist()
        level1_vals = prices.columns.get_level_values(1).unique().tolist()

        if "Close" in level0_vals:
            close_prices = prices["Close"]
        elif "Close" in level1_vals:
            close_prices = prices.xs("Close", axis=1, level=1)
        else:
            # 첫 번째 레벨이 티커일 수도 있음 (yfinance 최신 버전)
            test_tickers = [t for t in weights.keys() if t in level0_vals]
            if test_tickers:
                # level 0 = Ticker, level 1 = Price
                try:
                    close_prices = prices.xs("Close", axis=1, level=1)
                except KeyError:
                    close_prices = prices[test_tickers[0]].iloc[:, [0]]
            if close_prices is None:
                close_prices = prices.iloc[:, :len(weights)]
    else:
        close_prices = prices.copy()

    # close_prices columns가 아직 MultiIndex인 경우 flatten
    if isinstance(close_prices.columns, pd.MultiIndex):
        close_prices.columns = close_prices.columns.get_level_values(-1)

    # weights에 있는 티커만 필터
    available = [t for t in weights.keys() if t in close_prices.columns]
    if len(available) < 2:
        # 컬럼 이름이 다를 수 있으므로 부분 매칭 시도
        col_list = close_prices.columns.tolist()
        raise ValueError(
            f"상관관계 분석을 위해 최소 2개 종목이 필요합니다. "
            f"(매칭: {len(available)}개 / weights: {list(weights.keys())[:5]} / "
            f"columns: {col_list[:5]})"
        )
    close_prices = close_prices[available]

    # NaN이 많은 경우 처리
    close_prices = close_prices.dropna(how="all").ffill().bfill()

    # 수익률 계산 및 상관행렬
    returns = close_prices.pct_change().dropna()
    if len(returns) < 5:
        raise ValueError(f"수익률 데이터가 부족합니다. (현재 {len(returns)}일)")
    corr_matrix = returns.corr()
    tickers = list(corr_matrix.index)
    n_nodes = len(tickers)

    # 엣지 생성: |correlation| > threshold
    edges_list = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > corr_threshold:
                edges_list.append((i, j, corr_value))

    # Spring layout으로 노드 위치 계산
    positions = _spring_layout(n_nodes, edges_list, iterations=150, k=2.0)

    # 클러스터 탐지
    corr_array = corr_matrix.values
    clusters = _find_clusters(corr_array, threshold=0.5)
    ticker_to_cluster = {}
    for cluster_id, node_indices in clusters.items():
        for node_idx in node_indices:
            ticker_to_cluster[tickers[node_idx]] = cluster_id

    # 클러스터 색상 팔레트
    cluster_colors = [
        "#6366F1", "#10B981", "#F59E0B", "#EF4444",
        "#8B5CF6", "#06B6D4", "#EC4899", "#14B8A6",
        "#F97316", "#6B7280"
    ]

    # 노드 데이터 구성
    nodes = []
    weight_values = [weights.get(ticker, 0.01) for ticker in tickers]
    min_weight = min(weight_values) if weight_values else 0.01
    max_weight = max(weight_values) if weight_values else 1.0

    for idx, ticker in enumerate(tickers):
        weight = weights.get(ticker, 0.01)
        # 노드 크기: 가중치에 비례 (15~50)
        node_size = 15 + 35 * (weight - min_weight) / (max_weight - min_weight + 1e-6)

        cluster_id = ticker_to_cluster.get(ticker, 0)
        cluster_color = cluster_colors[cluster_id % len(cluster_colors)]

        nodes.append({
            "ticker": ticker,
            "x": positions[idx, 0],
            "y": positions[idx, 1],
            "size": node_size,
            "sector": sector_map.get(ticker, "Unknown"),
            "cluster": cluster_id,
            "weight": weight,
            "color": cluster_color,
        })

    # 엣지 데이터 구성
    edges = []
    for i, j, corr_value in edges_list:
        ticker_i = tickers[i]
        ticker_j = tickers[j]

        # 상관계수 기반 색상: 양수=초록, 음수=빨강
        if corr_value > 0:
            color = "rgba(16, 185, 129, 0.4)"  # 초록
        else:
            color = "rgba(239, 68, 68, 0.4)"   # 빨강

        # 엣지 두께: |correlation| 비례 (0.5~4)
        edge_width = 0.5 + 3.5 * abs(corr_value)

        edges.append({
            "source": ticker_i,
            "target": ticker_j,
            "correlation": corr_value,
            "color": color,
            "width": edge_width,
        })

    # 평균 상관계수 (절댓값)
    avg_corr = np.mean(np.abs([e["correlation"] for e in edges])) if edges else 0.0

    # NetworkData 구성
    return NetworkData(
        nodes=nodes,
        edges=edges,
        correlation_matrix=corr_matrix,
        clusters=clusters,
        avg_correlation=avg_corr,
    )


def _spring_layout(
    n_nodes: int,
    edges: List[Tuple[int, int, float]],
    iterations: int = 100,
    k: float = 1.0,
) -> np.ndarray:
    """
    Force-directed spring layout (Fruchterman-Reingold approximation)

    Returns: positions array of shape (n_nodes, 2)

    Algorithm:
    - Repulsive force between all node pairs: F_rep = k^2 / d
    - Attractive force along edges: F_att = d^2 / k * |correlation|
    - Temperature cooling schedule

    Args:
        n_nodes: Number of nodes
        edges: List of (i, j, correlation) tuples
        iterations: Number of iterations
        k: Optimal distance parameter

    Returns:
        positions: (n_nodes, 2) array of x, y coordinates
    """
    if n_nodes <= 1:
        return np.array([[0.0, 0.0]]) if n_nodes == 1 else np.empty((0, 2))

    # 초기 위치: 단위 원 위에 랜덤 배치
    np.random.seed(42)
    positions = np.random.randn(n_nodes, 2)
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    positions = positions / norms

    # 에지를 딕셔너리로 변환 (빠른 조회)
    edge_dict = {}
    for i, j, corr in edges:
        edge_dict[(min(i, j), max(i, j))] = abs(corr)

    # 온도 스케줄
    max_temp = np.sqrt(n_nodes)

    # Force-directed 반복
    for iteration in range(iterations):
        # 온도 감소 (선형)
        temperature = max_temp * (1 - iteration / iterations)

        # 힘 계산
        forces = np.zeros_like(positions)

        # 1. 척력 (모든 쌍)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                delta = positions[i] - positions[j]
                dist = np.linalg.norm(delta) + 1e-6

                # F_rep = k^2 / d
                force_mag = k * k / dist
                force_dir = delta / dist

                forces[i] += force_dir * force_mag
                forces[j] -= force_dir * force_mag

        # 2. 인력 (엣지만)
        for (i, j), corr_abs in edge_dict.items():
            delta = positions[i] - positions[j]
            dist = np.linalg.norm(delta) + 1e-6

            # F_att = d^2 / k * correlation
            force_mag = dist * dist / k * corr_abs
            force_dir = delta / dist

            forces[i] -= force_dir * force_mag
            forces[j] += force_dir * force_mag

        # 3. 위치 업데이트 (온도로 제한)
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mags = np.clip(force_mags, 0, temperature)

        for i in range(n_nodes):
            if np.linalg.norm(forces[i]) > 1e-6:
                direction = forces[i] / np.linalg.norm(forces[i])
                positions[i] += direction * force_mags[i]

    # 정규화: 약 1.5 범위로
    positions = positions / (np.max(np.abs(positions)) + 1e-6) * 1.5

    return positions


def _find_clusters(
    corr_matrix: np.ndarray,
    threshold: float = 0.5,
) -> Dict[int, List[int]]:
    """
    Simple correlation-based clustering
    - Nodes with avg mutual correlation > threshold belong to same cluster
    - Uses greedy agglomerative approach

    Args:
        corr_matrix: Correlation matrix (n_nodes, n_nodes)
        threshold: Minimum average correlation to form cluster

    Returns:
        Dict mapping cluster_id to list of node indices
    """
    n_nodes = corr_matrix.shape[0]

    # 각 노드의 평균 상관계수 (절댓값)
    avg_corrs = np.mean(np.abs(corr_matrix), axis=1)

    # 초기: 각 노드가 자신의 클러스터
    clusters = {i: [i] for i in range(n_nodes)}
    cluster_assignment = list(range(n_nodes))

    # Greedy 병합: 높은 상관계수를 가진 노드끼리 병합
    merged = True
    while merged:
        merged = False
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if cluster_assignment[i] == cluster_assignment[j]:
                    continue

                # 두 클러스터 사이의 평균 상관계수
                cluster_i_nodes = [idx for idx in range(n_nodes)
                                   if cluster_assignment[idx] == cluster_assignment[i]]
                cluster_j_nodes = [idx for idx in range(n_nodes)
                                   if cluster_assignment[idx] == cluster_assignment[j]]

                avg_corr_between = np.mean(
                    np.abs(corr_matrix[np.ix_(cluster_i_nodes, cluster_j_nodes)])
                )

                if avg_corr_between > threshold:
                    # 병합
                    old_cluster = cluster_assignment[j]
                    new_cluster = cluster_assignment[i]
                    for idx in range(n_nodes):
                        if cluster_assignment[idx] == old_cluster:
                            cluster_assignment[idx] = new_cluster
                    merged = True
                    break
            if merged:
                break

    # 클러스터 재구성
    result = {}
    for node_idx, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in result:
            result[cluster_id] = []
        result[cluster_id].append(node_idx)

    # 클러스터 ID 재정렬 (0부터 시작)
    final_result = {}
    for new_id, (old_id, nodes) in enumerate(sorted(result.items())):
        final_result[new_id] = nodes

    return final_result


# ========== Plotly Visualizations ==========

def create_network_graph(data: NetworkData) -> go.Figure:
    """
    인터랙티브 상관관계 네트워크 그래프

    - 노드: 원형 마커, 크기=비중 비례, 색상=섹터/클러스터
    - 엣지: 선, 두께=|상관계수| 비례
      - 양의 상관: 초록색 (#10B981)
      - 음의 상관: 빨간색 (#EF4444)
    - Hover: 종목명, 섹터, 비중, 가장 상관 높은 종목
    - 다크 테마

    Args:
        data: NetworkData instance

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # 엣지 그리기
    for edge in data.edges:
        source_node = next(n for n in data.nodes if n["ticker"] == edge["source"])
        target_node = next(n for n in data.nodes if n["ticker"] == edge["target"])

        fig.add_trace(go.Scatter(
            x=[source_node["x"], target_node["x"], None],
            y=[source_node["y"], target_node["y"], None],
            mode="lines",
            line=dict(
                width=edge["width"],
                color=edge["color"],
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    # 노드 그리기 (마커)
    node_x = [n["x"] for n in data.nodes]
    node_y = [n["y"] for n in data.nodes]
    node_text = [n["ticker"] for n in data.nodes]
    node_size = [n["size"] for n in data.nodes]
    node_color = [n["color"] for n in data.nodes]

    # Hover text 구성: 각 노드에 대해 가장 상관이 높은 종목들
    hover_texts = []
    for node in data.nodes:
        ticker = node["ticker"]
        sector = node["sector"]
        weight = node["weight"]

        # 이 종목과의 상관계수
        corr_row = data.correlation_matrix.loc[ticker]
        # 절댓값으로 정렬하여 상위 3개 선택 (자신 제외)
        top_corr = corr_row.drop(ticker).abs().nlargest(3)

        hover_text = f"<b>{ticker}</b><br>"
        hover_text += f"Sector: {sector}<br>"
        hover_text += f"Weight: {weight:.2%}<br>"
        hover_text += f"Cluster: {node['cluster']}<br><br>"
        hover_text += "Top Correlations:<br>"
        for top_ticker, corr_val in top_corr.items():
            actual_corr = data.correlation_matrix.loc[ticker, top_ticker]
            hover_text += f"{top_ticker}: {actual_corr:+.3f}<br>"

        hover_texts.append(hover_text)

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color="rgba(255, 255, 255, 0.8)"),
            opacity=0.9,
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color="white", family="monospace"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    ))

    # 레이아웃 설정
    fig.update_layout(
        title={
            "text": f"<b>Correlation Network</b><br><sub>Avg Correlation: {data.avg_correlation:.3f}</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "white"}
        },
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="white", family="Arial"),
        height=700,
    )

    return fig


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    상관관계 히트맵 (보조 차트)
    - Diverging colorscale (RdBu)
    - Annotation with correlation values
    - 다크 테마

    Args:
        corr_matrix: Correlation matrix DataFrame

    Returns:
        Plotly Figure
    """
    # 상관계수 값을 텍스트로 변환
    text_values = [[f"{val:.2f}" for val in row] for row in corr_matrix.values]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 9, "color": "white"},
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color="white")),
            thickness=15,
            len=0.8,
            tickfont=dict(color="white"),
        ),
    ))

    fig.update_layout(
        title={
            "text": "<b>Correlation Matrix Heatmap</b>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "white"}
        },
        xaxis=dict(
            tickfont=dict(size=10, color="white"),
            showgrid=False,
            side="bottom",
        ),
        yaxis=dict(
            tickfont=dict(size=10, color="white"),
            showgrid=False,
        ),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="white"),
        height=600,
        width=800,
        margin=dict(l=100, r=100, t=80, b=100),
    )

    return fig


def create_cluster_summary(data: NetworkData) -> pd.DataFrame:
    """
    클러스터 요약 테이블
    - 클러스터 ID, 종목 목록, 평균 상관계수, 종목 수

    Args:
        data: NetworkData instance

    Returns:
        Summary DataFrame
    """
    summary_rows = []

    for cluster_id, node_indices in sorted(data.clusters.items()):
        # 클러스터 내 종목들
        tickers_in_cluster = [data.nodes[idx]["ticker"] for idx in node_indices]

        # 클러스터 내 평균 상관계수
        if len(node_indices) > 1:
            corr_values = []
            for i, idx_i in enumerate(node_indices):
                for idx_j in node_indices[i+1:]:
                    ticker_i = data.nodes[idx_i]["ticker"]
                    ticker_j = data.nodes[idx_j]["ticker"]
                    corr_val = abs(data.correlation_matrix.loc[ticker_i, ticker_j])
                    corr_values.append(corr_val)
            avg_intra_corr = np.mean(corr_values) if corr_values else 0.0
        else:
            avg_intra_corr = 0.0

        # 클러스터 내 총 가중치
        total_weight = sum(data.nodes[idx]["weight"] for idx in node_indices)

        summary_rows.append({
            "Cluster": cluster_id,
            "Count": len(node_indices),
            "Tickers": ", ".join(sorted(tickers_in_cluster)),
            "Avg Correlation": f"{avg_intra_corr:.3f}",
            "Total Weight": f"{total_weight:.2%}",
        })

    return pd.DataFrame(summary_rows)
