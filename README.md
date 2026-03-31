# Dacon 프로젝트 모음

이 레포지토리는 **데이콘(Dacon)** 대회별 실험 코드를 폴더 단위로 정리한 저장소입니다.
현재는 전력 사용량 예측, 수출입 품목 공이동(comovement) 분석, 유전체 언어모델(MLM) 학습, 구조물 안정성 물리 추론 프로젝트가 포함되어 있습니다.

## 폴더 구성

| 폴더 | 설명 | 주요 파일 |
|------|------|-----------|
| `Power_usage_pred` | 건물별 전력 사용량 예측 대회 실험 코드 | `catboost_train.py`, `lightbgm_train.py`, `Optuna.py` |
| `comovement` | 시계열 기반 품목 간 선후행 관계(후보 pair) 탐색/학습 파이프라인 | `prepare_data.py`, `make_pairs_corr.py`, `build_pairs_ml.py`, `make_final_pairs.py` |
| `mai` | 유전체 서열(FASTA) 기반 마스킹 언어모델(MLM) 학습/추론 코드 | `train_nt.py`, `models_nt.py`, `inference.py`, `inference_nt2p5b.py` |
| `structural_stability` | 구조물 안정성 물리 추론 대회용 비전 baseline 및 앙상블 코드 | `scripts/train_clean_convnext.py`, `scripts/train_geometry_first.py`, `scripts/stack_meta_ensemble.py` |

---

## 폴더별 실행 예시

> 아래 커맨드는 **예시**이며, 실제 실행 전 각 스크립트의 데이터 경로/출력 경로를 본인 환경에 맞게 수정해야 합니다.

### 1) 전력 사용량 예측 (`Power_usage_pred`)

```bash
python Power_usage_pred/catboost_train.py
python Power_usage_pred/lightbgm_train.py
python Power_usage_pred/Optuna.py
```

### 2) 공이동 후보쌍 생성 (`comovement`)

```bash
python comovement/prepare_data.py
python comovement/make_pairs_corr.py
python comovement/build_pairs_ml.py
python comovement/make_final_pairs.py \
  --pairs_corr pairs_corr.csv \
  --pseudo_stats pair_pseudo_stats.csv \
  --out_pairs candidate_pairs_final.csv
```

### 3) 유전체 MLM 학습/추론 (`mai`)

```bash
python mai/train_nt.py
python mai/inference.py
```

### 4) 구조물 안정성 물리 추론 (`structural_stability`)

```bash
python structural_stability/scripts/train_clean_convnext.py train \
  --data-root /path/to/data \
  --out-dir runs/clean_convnext_base
```

세부 실험 설정은 `structural_stability/README.md`를 참고하세요.

---

## 공통 환경 준비

Python 3.8+ 환경을 권장합니다.

```bash
pip install pandas numpy scikit-learn lightgbm catboost xgboost optuna torch transformers tqdm
```

필요 패키지는 프로젝트별로 다를 수 있으니, 실행 스크립트의 import를 기준으로 추가 설치해 주세요.

## 참고

- 대회 데이터/규정은 각 데이콘 대회 페이지의 약관과 규칙을 따릅니다.
- 저장소 내 일부 스크립트는 경로를 비워 둔 템플릿 형태이므로 실행 전 경로 설정이 필요합니다.
