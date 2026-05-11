import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from scipy.special import boxcox1p, inv_boxcox1p
import lightgbm as lgb

# Load data
train = pd.read_csv(r"")
test = pd.read_csv(r"")
building = pd.read_csv(r"")
submission = pd.read_csv(r"")

# Numeric 처리
num_cols = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for col in num_cols:
    building[col] = pd.to_numeric(building[col].replace('-', 0))

# Merge building info
train = train.merge(building, on='건물번호', how='left')
test = test.merge(building, on='건물번호', how='left')

# Fill NA for weather
weather_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
for col in weather_cols:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(test[col].mean())

# Date features
for df in [train, test]:
    df['일시'] = pd.to_datetime(df['일시'])
    df['hour'] = df['일시'].dt.hour
    df['month'] = df['일시'].dt.month
    df['weekday'] = df['일시'].dt.weekday

def get_season(month):
    if month in [12, 1, 2]: return 'winter'
    elif month in [3, 4, 5]: return 'spring'
    elif month in [6, 7, 8]: return 'summer'
    else: return 'autumn'

train['season'] = train['month'].apply(get_season)
test['season'] = test['month'].apply(get_season)
train['건물유형_season'] = train['건물유형'] + '_' + train['season']
test['건물유형_season'] = test['건물유형'] + '_' + test['season']

# Label Encoding
cat_features = ['건물유형', '건물번호', '건물유형_season', 'season']
for col in cat_features:
    le = LabelEncoder()
    full = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Target Transform
y = np.log1p(train['전력소비량(kWh)'])

# Features
feature_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
                '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
                '건물유형', '건물번호', 'hour', 'month', 'weekday', 'season', '건물유형_season']
X = train[feature_cols]
X_test = test[feature_cols]

# Cross Validation
folds = GroupKFold(n_splits=5)
preds = np.zeros(len(test))
mae_list = []

for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y, groups=train["건물번호"])):
    print(f"\n[Fold {fold}] Training...")

    X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_train, y_val = y[tr_idx], y[val_idx]

    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, y_val, categorical_feature=cat_features)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.005,
        'num_leaves': 16,
        'max_depth': 4,
        'min_data_in_leaf': 200,
        'lambda_l1': 5.0,
        'lambda_l2': 5.0,
        'verbosity': -1,
        'feature_pre_filter': False
    }

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        num_boost_round=20000,
        callbacks=[
            lgb.early_stopping(500, verbose=True),
            lgb.log_evaluation(500)
        ]
    )

    val_pred = np.expm1(model.predict(X_val))
    y_true = train['전력소비량(kWh)'].iloc[val_idx].values
    print(val_pred[:10])
    print(y_true[:10])
    mae = np.mean(np.abs(val_pred - y_true))
    print(f"[Fold {fold}] MAE: {mae:.4f}")
    mae_list.append(mae)

    test_pred = inv_boxcox1p(model.predict(X_test), 0.05)
    preds += test_pred / folds.n_splits

print(f"\nAverage MAE: {np.mean(mae_list):.4f}")

# Save submission
submission = pd.DataFrame({
    'num_date_time': test['num_date_time'],
    '전력소비량(kWh)': preds
})
submission.to_csv("submission_lgb_optimized.csv", index=False)
