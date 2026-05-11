# Model XAI Comparison

## Top Gain Features
### LightGBM
- `congestion_score_exp_mean`: `1619713.8235`
- `pack_utilization_exp_mean`: `281528.2776`
- `max_zone_density_roll3_mean`: `111471.3882`
- `avg_trip_distance`: `65659.6526`
- `max_zone_density`: `59488.7033`
- `layout_compactness`: `52133.3561`
- `battery_mean_exp_std`: `50833.7967`
- `low_battery_ratio_exp_std`: `46171.1951`
- `congestion_score`: `44929.4896`
- `pack_utilization`: `44645.7211`

### CatBoost
- `pack_utilization_exp_mean`: `5.1058`
- `layout_robot_pack_ratio`: `4.9360`
- `pack_utilization_exp_std`: `4.6223`
- `pack_station_count`: `4.4458`
- `congestion_score_exp_mean`: `3.6713`
- `sku_concentration_roll3_mean`: `3.5075`
- `loading_dock_util_exp_mean`: `3.3830`
- `order_inflow_15m_exp_std`: `2.8360`
- `order_inflow_15m_exp_mean`: `2.8168`
- `layout_compactness`: `2.5821`

## Top Permutation Features
### LightGBM
- `congestion_score_exp_mean`: `2.644213`
- `low_battery_ratio_exp_std`: `0.528648`
- `pack_utilization_exp_mean`: `0.477472`
- `max_zone_density`: `0.176459`
- `pack_utilization`: `0.113452`
- `max_zone_density_roll3_mean`: `0.109011`
- `charging_ratio_layout`: `0.096934`
- `congestion_score`: `0.092314`
- `battery_mean_exp_std`: `0.082387`
- `avg_trip_distance`: `0.067162`

### CatBoost
- `congestion_score_exp_mean`: `1.672165`
- `pack_utilization_exp_std`: `0.302232`
- `pack_utilization_exp_mean`: `0.297437`
- `low_battery_ratio_exp_std`: `0.164158`
- `battery_mean_exp_mean`: `0.154496`
- `pack_station_count`: `0.088290`
- `order_inflow_15m_exp_std`: `0.079260`
- `layout_robot_pack_ratio`: `0.073500`
- `congestion_score_exp_std`: `0.060093`
- `avg_trip_distance`: `0.049951`

## Common High-Value Features
- `avg_trip_distance`
- `congestion_score_exp_mean`
- `congestion_score_exp_std`
- `layout_compactness`
- `layout_robot_pack_ratio`
- `low_battery_ratio_exp_std`
- `order_inflow_15m_exp_mean`
- `pack_station_count`
- `pack_utilization_exp_mean`

## How To Use Common vs Model-Specific Features

공통 중요 피처와 모델별 차별 피처는 같은 방식으로 다루지 않는 편이 좋다.

- 공통 중요 피처는 두 모델이 모두 신호로 본 항목이므로 `shared feature backbone`으로 유지한다.
- 모델별 차별 피처는 버릴 대상이 아니라 각 모델의 전문 영역으로 본다.
- 최종 판단 기준은 importance 자체가 아니라 `CV 개선 여부`다.

### Shared Backbone

다음 피처들은 두 모델이 공통으로 높게 본 축이라서 우선적으로 강화할 가치가 높다.

- `avg_trip_distance`
- `congestion_score_exp_mean`
- `congestion_score_exp_std`
- `layout_compactness`
- `layout_robot_pack_ratio`
- `low_battery_ratio_exp_std`
- `order_inflow_15m_exp_mean`
- `pack_station_count`
- `pack_utilization_exp_mean`

이 축은 다음과 같이 더 파는 게 맞다.

- 혼잡 축: `congestion_score`, `max_zone_density`, `aisle_traffic_score`, `intersection_wait_time_avg`
- 패킹 축: `pack_utilization`, `pack_station_count`, `label_print_queue`, `pick_list_length_avg`
- 배터리 축: `low_battery_ratio`, `battery_mean`, `charge_queue_length`, `charging_ratio_layout`
- 구조 축: `layout_compactness`, `layout_robot_pack_ratio`, `pack_station_count`, `aisle_width_avg`

### LightGBM-Specific Signals

LightGBM이 상대적으로 더 강하게 보는 축은 혼잡 누적, 배터리 변동성, 현재 시점의 밀도/혼잡 계열이다.

- `congestion_score_exp_mean`
- `max_zone_density`
- `charging_ratio_layout`
- `battery_mean_exp_std`
- `low_battery_ratio_exp_std`

이 축은 `volatility`, `slope`, `recent jump` 같은 피처를 더 붙이는 쪽이 유효하다.

### CatBoost-Specific Signals

CatBoost가 상대적으로 더 강하게 보는 축은 패킹 구조, 레이아웃-용량 균형, 주문 부하 변동성이다.

- `pack_utilization_exp_std`
- `layout_robot_pack_ratio`
- `pack_station_count`
- `order_inflow_15m_exp_std`
- `loading_dock_util_exp_mean`

이 축은 구조 비율, 병목 조합, 레이아웃-운영 interaction을 더 만드는 쪽이 유효하다.

### Decision Rule

피처를 유지하거나 확장할 때는 아래 기준으로 판단하는 게 좋다.

- 공통 중요 + CV 개선: 최우선 유지 및 확장
- 단일 모델 중요 + 해당 모델 CV 개선: 유지
- 단일 모델 중요 + CV 개선 없음: 후보군 보류
- 두 모델 모두 낮음: 확장 우선순위 낮춤

### Practical Direction

실전적으로는 이렇게 가져가면 된다.

- `shared` 피처군은 두 모델 공통 베이스로 강화
- `lightgbm_only` 피처군은 혼잡/변동성 중심으로 확장
- `catboost_only` 피처군은 구조/병목 interaction 중심으로 확장
- 최종 제출은 단일 모델보다 `blend` 성능으로 판단

즉 목표는 두 모델의 중요도를 억지로 같게 만드는 것이 아니라, 공통 신호는 안정적으로 잡고 차별 신호는 `blend diversity`로 활용하는 것이다.
