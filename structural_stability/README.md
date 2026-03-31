# Physics Baseline Export

이 폴더는 `/data/AskFake/Image/physics/baseline` 실험물 중에서 GitHub 업로드용으로 정리한 export 폴더입니다.

## Challenge Overview

이 코드는 DACON의 **구조물 안정성 물리 추론 AI 경진대회**를 기준으로 정리되었습니다.

- 대회 링크: https://dacon.io/competitions/official/236686/overview/description
- 주제: 시각 기반 구조물 안정성 예측 AI 모델 개발
- 입력: 각 샘플당 `front`, `top` 2개 시점 이미지
- 목표: 시뮬레이션 시작 후 10초 동안 구조물이
  - `unstable` 상태로 전환될 확률
  - `stable` 상태를 유지할 확률
  를 예측

대회 설명에 따르면 라벨은 물리 시뮬레이션 결과를 기반으로 정의됩니다.

- `stable`: 10초 동안 의미 있는 이동이나 변형이 없는 경우
- `unstable`: 10초 이내 누적 이동 거리가 1.5cm 이상이거나 구조적 붕괴가 나타난 경우

데이터는 단순 분류 문제처럼 보이지만, 실제로는 **무게중심 편차, 층별 하중 분포, 구조적 배치 패턴** 같은 물리적 요소를 함께 고려해야 하는 시각 기반 물리 추론 문제에 가깝습니다.

또한 대회 구성은 train/dev/test의 환경 차이를 의도적으로 둡니다.

- `train`: 1,000 샘플, 고정된 광원/카메라의 실험실 환경
- `dev`: 100 샘플, 실제 평가 환경과 같은 무작위 광원/카메라
- `test`: 1,000 샘플, dev와 동일한 무작위 환경

즉 이 저장소의 모든 baseline은 단순한 이미지 분류 성능보다, **환경 변화에 강건한 물리적 일반화**를 얼마나 만들 수 있는지에 초점을 맞추고 있습니다.

## 구조

- `scripts/`
  - 실험에 사용한 주요 학습/예측 스크립트
- `docs/`
  - 결과 요약과 실험 메모
- `.gitignore`
  - 업로드 시 제외할 항목

## 포함한 주요 스크립트

- `train_clean_convnext.py`
  - clean ConvNeXt dual-stream CV baseline
- `train_geometry_first.py`
  - handcrafted geometry / physics feature baseline
- `ensemble_submissions.py`
  - submission weighted average
- `stack_meta_ensemble.py`
  - OOF 기반 meta stacking
- `train_block_graph_baseline.py`
  - block graph baseline
- `train_support_graph_baseline.py`
  - support graph baseline
- `train_top_structure_baseline.py`
  - top-only structural baseline
- `debug_full_roi_crop.py`
  - ROI crop 디버그
- `debug_structural_parser.py`
  - structural parser 디버그

## 현재 best

현재 best 제출은 원본 baseline 폴더의 아래 파일입니다.

- `submission_anchor99_clean1.csv`
- leaderboard score: `0.0452095804`

이 export 폴더에는 대용량 `runs/`와 제출 CSV는 포함하지 않았습니다.

## 권장 업로드 방식

1. 이 `github_ready` 폴더를 새 repo 루트로 사용
2. 필요하면 `scripts/`를 루트로 옮기고 경로를 정리
3. 이후 `requirements.txt`, `LICENSE`, 예시 manifest를 추가

## 빠른 시작 예시

ConvNeXt clean baseline:

```bash
python scripts/train_clean_convnext.py train \
  --data-root /data/AskFake/Image/physics \
  --out-dir runs/clean_convnext_base \
  --model-name convnext_small.fb_in22k_ft_in1k_384 \
  --image-size 384 \
  --batch-size 8 \
  --accum-steps 4 \
  --epochs 30 \
  --n-folds 5 \
  --num-workers 4 \
  --backbone-lr 2e-5 \
  --head-lr 2e-4 \
  --seed 42 \
  --front-mode original \
  --top-mode original
```

Geometry baseline:

```bash
python scripts/train_geometry_first.py train \
  --data-root /data/AskFake/Image/physics \
  --out-dir runs/geometry_first_v1 \
  --n-folds 5 \
  --n-clusters 16 \
  --seed 42 \
  --use-top-normalize
```
