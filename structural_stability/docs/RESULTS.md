# Results Summary

## Best Submission

- File: `submission_anchor99_clean1.csv`
- Score: `0.0452095804`

## What Worked

- `submission_stack_convnext_geo.csv`
  - strong anchor submission
- `train_clean_convnext.py`
  - clean ConvNeXt base branch
- `ensemble_submissions.py`
  - anchor + clean convnext `99:1` blend improved slightly

## What Did Not Generalize

- `physics_base` rerun
- `Track B` video distillation
- `front ROI + top ROI` clean convnext
- `geometry_first_v1_toproi`
- `block_graph` / `support_graph`
- `top-only` structural baseline
- geometry physics theory variants
  - `support_margin`
  - `hybrid`
  - `paper_hybrid`

## Interpretation

- 내부 OOF와 실제 leaderboard 사이 차이가 컸음
- 구조적/물리적 아이디어 자체보다도, 현재 pseudo parsing 품질이 부족해서 실전 generalization이 깨지는 경우가 많았음
- 최종적으로는 기존 strong anchor를 유지하고, clean ConvNeXt를 아주 소량 섞는 방식이 가장 안정적이었음

## Recommended Baseline To Resume

1. `submission_stack_convnext_geo.csv`를 anchor로 유지
2. `train_clean_convnext.py` 기반 branch를 보조 축으로 사용
3. geometry branch는 원래 `geometry_first_v1`를 유지
