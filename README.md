# Dacon 프로젝트 모음

이 레포지토리는 **데이콘(Dacon)** AI 대회에서 진행한 코드들을 정리한 공간입니다. 각 폴더는 대회별 코드를 포함하며, 향후 참여한 대회들이 추가될 예정입니다.

## 폴더 구성

| 폴더 | 설명 |
|------|------|
| `Power_usage_pred` | 건물별 전력 사용량 예측 대회 코드
|

`Power_usage_pred` 폴더 안에는 CatBoost, LightGBM 등을 활용한 학습 스크립트와 Optuna 튜닝 예시가 포함되어 있습니다. 데이터 경로는 비워 두었으니 필요에 맞게 수정 후 실행하면 됩니다.

## 실행 방법

1. Python 환경을 준비합니다. (예: Python 3.8 이상)
2. 필요 라이브러리를 설치합니다.
   ```bash
   pip install pandas numpy scikit-learn lightgbm catboost xgboost optuna
   ```
3. 각 스크립트 상단의 데이터 경로를 본인 환경에 맞게 지정합니다.
4. 원하는 스크립트를 실행하여 모델을 학습합니다.

```bash
python Power_usage_pred/catboost_train.py
```

## 라이선스

본 레포지토리의 코드는 자유롭게 활용할 수 있지만, 데이터 사용 규정은 각 대회의 규칙을 따릅니다.

