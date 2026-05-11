import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc

# 데이터 불러오기
train = pd.read_csv("")
test = pd.read_csv("")
building_info = pd.read_csv(r"")
sample_submission = pd.read_csv("")

# 건물 정보 병합
train = pd.merge(train, building_info, on="건물번호", how="left")
test = pd.merge(test, building_info, on="건물번호", how="left")

# 시간 파생 변수 생성
def add_time_features(df):
    df["datetime"] = pd.to_datetime(df["일시"])
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df.drop(columns=["datetime", "일시"], inplace=True)
    return df

train = add_time_features(train)
test = add_time_features(test)

# 범주형 변수 인코딩
categorical_cols = ["건물유형"]
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 학습/테스트 데이터 구성
X = train.drop(columns=["전력소비량(kWh)", "num_date_time"])
y = train["전력소비량(kWh)"]
X_test = test.drop(columns=["num_date_time"])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoostRegressor
cat_model = CatBoostRegressor(
    iterations=10000,
    learning_rate=0.03,
    depth=8,
    eval_metric="MAE",
    random_seed=42,
    task_type="GPU",
    early_stopping_rounds=300,
    verbose=200
)

cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[X.columns.get_loc(c) for c in categorical_cols])
cat_valid_pred = cat_model.predict(X_valid)
cat_test_pred = cat_model.predict(X_test)

# LightGBMRegressor
lgb_model = LGBMRegressor(
    n_estimators=10000,
    learning_rate=0.03,
    max_depth=8,
    random_state=42,
    device="cpu"
)

lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=300, verbose=200)
lgb_valid_pred = lgb_model.predict(X_valid)
lgb_test_pred = lgb_model.predict(X_test)

# 앙상블
valid_pred = 0.8 * cat_valid_pred + 0.2 * lgb_valid_pred
test_pred = 0.8 * cat_test_pred + 0.2 * lgb_test_pred

# SMAPE 정의
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

val_smape = smape(y_valid, valid_pred)
print(f"Validation SMAPE: {val_smape:.4f}")

# 제출 파일 저장
sample_submission["answer"] = test_pred
sample_submission.to_csv("/mnt/data/submission_ensemble.csv", index=False)

# GPU 메모리 정리
del cat_model, lgb_model
gc.collect()

try:
    import torch
    torch.cyda.empty_cache()
except:
    pass
