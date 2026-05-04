# XAI Insight Summary

## Validation
- split: `layout_holdout`
- mae: `7.911877`
- rmse: `19.060673`
- n_train: `198750`
- n_valid: `51250`

## Top Gain Features
- `congestion_score_exp_mean`: gain `1619713.82`
- `pack_utilization_exp_mean`: gain `281528.28`
- `max_zone_density_roll3_mean`: gain `111471.39`
- `avg_trip_distance`: gain `65659.65`
- `max_zone_density`: gain `59488.70`
- `layout_compactness`: gain `52133.36`
- `battery_mean_exp_std`: gain `50833.80`
- `low_battery_ratio_exp_std`: gain `46171.20`
- `congestion_score`: gain `44929.49`
- `pack_utilization`: gain `44645.72`
- `pack_utilization_roll3_mean`: gain `38822.96`
- `sku_concentration_roll3_mean`: gain `36787.65`
- `zone_dispersion`: gain `34646.42`
- `order_inflow_15m_exp_mean`: gain `30962.05`
- `layout_robot_pack_ratio`: gain `30793.70`

## Top Permutation Features
- `congestion_score_exp_mean`: mean drop `2.644213` +/- `0.027911`
- `low_battery_ratio_exp_std`: mean drop `0.528648` +/- `0.016517`
- `pack_utilization_exp_mean`: mean drop `0.477472` +/- `0.012836`
- `max_zone_density`: mean drop `0.176459` +/- `0.010449`
- `pack_utilization`: mean drop `0.113452` +/- `0.017886`
- `max_zone_density_roll3_mean`: mean drop `0.109011` +/- `0.003582`
- `charging_ratio_layout`: mean drop `0.096934` +/- `0.004478`
- `congestion_score`: mean drop `0.092314` +/- `0.005741`
- `battery_mean_exp_std`: mean drop `0.082387` +/- `0.006280`
- `avg_trip_distance`: mean drop `0.067162` +/- `0.005778`
- `pack_station_count`: mean drop `0.060803` +/- `0.006471`
- `layout_robot_pack_ratio`: mean drop `0.037299` +/- `0.010172`
- `order_inflow_15m_exp_mean`: mean drop `0.032178` +/- `0.002094`
- `layout_compactness`: mean drop `0.028007` +/- `0.002714`
- `congestion_score_exp_std`: mean drop `0.024698` +/- `0.002667`

## Family-Level Importance
- `traffic_and_congestion`: total gain `1750461.86`
- `sequence_history`: total gain `490545.84`
- `other`: total gain `327394.76`
- `order_and_pick_workload`: total gain `253265.40`
- `battery_and_charging`: total gain `175401.57`
- `layout_structure`: total gain `160383.32`
- `robot_operations`: total gain `100515.45`
- `dock_and_material_flow`: total gain `46906.20`
- `systems_and_it`: total gain `32902.84`
- `environment`: total gain `29557.60`

## Modeling Direction
- `battery_and_charging`, `traffic_and_congestion`, `order_and_pick_workload`, `sequence_history` 패밀리가 상위면 lag/rolling 강화가 계속 유효하다는 의미다.
- `layout_structure` 패밀리 비중이 높으면 unseen layout 일반화가 중요하므로 layout cluster, layout target encoding, ratio-based normalization을 더 강화할 가치가 있다.
- `systems_and_it`나 `missingness` 비중이 높으면 단순 값 자체보다 운영 이상 신호를 모델이 읽고 있다는 뜻이므로 missing indicator와 interaction을 유지하는 편이 낫다.
- permutation importance에서 top feature가 gain importance와 다르면, split을 많이 탄 feature보다 실제 성능 기여 feature를 우선해서 정제해야 한다.