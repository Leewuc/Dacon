import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
import warnings
import optuna
warnings.filterwarnings("ignore")

# 1. 데이터 로딩
train = pd.read_csv(r"")
test = pd.read_csv(r"")
building_info = pd.read_csv(r"")

# 2. merge

train = train.merge(building_info, on='건물번호', how='left')
test = test.merge(building_info, on='건물번호', how='left')

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))

# 3. 일시 파싱 및 시간 변수 추가
for df in [train, test]:
    df['일시'] = pd.to_datetime(df['일시'], errors='coerce')  # datetime 형식으로 변환
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['quarter'] = df['일시'].dt.quarter
    df['hour'] = df['일시'].dt.hour  # is_night 계산을 위해 먼저 생성
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)

# 4. 불필요한 열 제거
drop_cols = ['일시', 'num_date_time', '일조(hr)', '일사(MJ/m2)']
target = '전력소비량(kWh)'
features = [col for col in train.columns if col not in drop_cols + [target]]

# 5. object → category 변환
for col in train.columns:
    if train[col].dtype == 'object':
        try:
            train[col] = pd.to_numeric(train[col])
            test[col] = pd.to_numeric(test[col])
        except:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')

# 6. 이상치 제거
q1, q3 = train[target].quantile([0.25, 0.75])
iqr = q3 - q1
train = train[(train[target] >= q1 - 1.5 * iqr) & (train[target] <= q3 + 1.5 * iqr)]


# 7. 로그 변환
train[target] = np.log1p(train[target])

# 8. 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(train[features], train[target], test_size=0.2, random_state=42)
X_test = test[features]
y_val_exp = np.expm1(y_val)

### 6. Optuna Objective Functions

# LightGBM
def objective_lgb(trial):
    params = {
        "n_estimators": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_gain_to_split" : trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "random_state": 42
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, 
              y_train, 
              eval_set=[(X_val, y_val)], 
              callbacks=[
                  early_stopping(50), 
                  log_evaluation(period = 100)
            ]
    )
    pred = np.expm1(model.predict(X_val))
    smape = 100 * np.mean(2 * np.abs(pred - y_val_exp) / (np.abs(pred) + np.abs(y_val_exp)))
    return smape

# XGBoost
def objective_xgb(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "verbosity" : 0,
        "eval_metric" : "mae",
        "enable_categorical": True
    }
    model = XGBRegressor(**params)
    model.fit(X_train, 
              y_train, 
              eval_set=[(X_val, y_val)], 
              verbose=100
            )
    pred = np.expm1(model.predict(X_val))
    smape = 100 * np.mean(2 * np.abs(pred - y_val_exp) / (np.abs(pred) + np.abs(y_val_exp)))
    return smape

# CatBoost
def objective_cat(trial):
    params = {
        "iterations": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "loss_function": "MAE",
        "verbose": 100,
        "random_seed": 42
    }
    cat_features = [i for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    model = CatBoostRegressor(**params, cat_features=cat_features)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    pred = np.expm1(model.predict(X_val))
    smape = 100 * np.mean(2 * np.abs(pred - y_val_exp) / (np.abs(pred) + np.abs(y_val_exp)))
    return smape

### 7. 하이퍼파라미터 튜닝
print(" LightGBM 튜닝 중...")
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=10)

print(" XGBoost 튜닝 중...")
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=10)

print(" CatBoost 튜닝 중...")
study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(objective_cat, n_trials=10)

### 8. 최적 모델 학습
model_lgb = LGBMRegressor(**study_lgb.best_params)
model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(50)])

model_xgb = XGBRegressor(**study_xgb.best_params)
model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

cat_features = [i for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
model_cat = CatBoostRegressor(**study_cat.best_params, cat_features=cat_features, verbose=0)
model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

### 9. 앙상블 예측
pred_lgb = np.expm1(model_lgb.predict(X_val))
pred_xgb = np.expm1(model_xgb.predict(X_val))
pred_cat = np.expm1(model_cat.predict(X_val))
final_pred = 0.3 * pred_lgb + 0.3 * pred_xgb + 0.4 * pred_cat

mae = mean_absolute_error(np.expm1(y_val), final_pred)

print(f" MAE: {mae:.4f}")
print(f" SMAPE: {smape:.4f}")

### 10. 최종 예측 및 저장
final_test_pred = (
    0.3 * np.expm1(model_lgb.predict(X_test)) +
    0.3 * np.expm1(model_xgb.predict(X_test)) +
    0.4 * np.expm1(model_cat.predict(X_test))
)
