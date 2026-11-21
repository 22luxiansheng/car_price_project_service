import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from features import (
    CAT_COLS, FEATURE_COLS, add_base_features, clean_categoricals, transform
)

# 特征顺序必须和 features.FEATURE_COLS 一致：
# make, fuel, transmission, body, kms_run, Car_Age, Avg_Mileage_Per_Year,
# Age_Mileage_Interaction, Condition_Score
# 0 = 不约束，-1 = 越大价格越低，+1 = 越大价格越高
MONOTONE_CONSTRAINTS = (0, 0, 0, 0, -1, -1, -1, -1, 1)

assert len(MONOTONE_CONSTRAINTS) == len(FEATURE_COLS)

DATA_PATH = "data/raw/car_data.csv"
MODEL_PATH = "models/car_price_model.pkl"
BUNDLE_PATH = "models/preprocessor.pkl"

# ---------- 1. 读数据 + 清洗 ----------
df = pd.read_csv(DATA_PATH)

# 挂牌年份：数据是 2019-2021 年的挂牌记录，车龄必须按挂牌年份算
df["ad_year"] = pd.to_datetime(df["ad_created_on"], errors="coerce").dt.year
df = df[df["ad_year"].notna()]

df = df.drop_duplicates()
df = df[(df["sale_price"] > 0) & (df["sale_price"] < 10_000_000) & (df["kms_run"] > 0)]
df = df[(df["yr_mfr"] >= 1980) & (df["yr_mfr"] <= df["ad_year"])]
print(f"清洗后: {len(df)} 行")

# 参考年份 = 数据里最常见的挂牌年份（2021），线上推理时用它计算车龄
REFERENCE_YEAR = int(df["ad_year"].mode().iloc[0])
print(f"价格口径参考年份: {REFERENCE_YEAR}")

df = clean_categoricals(df)

# ---------- 2. 先划分，再用训练集拟合一切 ----------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 类别缺失值：用训练集的众数填，测试集也用训练集的众数
cat_fill_values = {}
for c in CAT_COLS:
    mode = train_df[c].mode().iloc[0]
    cat_fill_values[c] = str(mode)
    train_df[c] = train_df[c].fillna(mode)
    test_df[c] = test_df[c].fillna(mode)

train_df = add_base_features(train_df, REFERENCE_YEAR)
test_df = add_base_features(test_df, REFERENCE_YEAR)

scaler_age = MinMaxScaler().fit(train_df[["Car_Age"]])
scaler_kms = MinMaxScaler().fit(train_df[["kms_run"]])

# 类别 -> 整数 的映射，只在训练集上建立
mappings = {}
for c in CAT_COLS:
    classes = sorted(train_df[c].dropna().unique().tolist())
    mappings[c] = {cls: i for i, cls in enumerate(classes)}

bundle = {
    "reference_year": REFERENCE_YEAR,
    "cat_cols": CAT_COLS,
    "feature_cols": FEATURE_COLS,
    "cat_mappings": mappings,
    "cat_fill_values": cat_fill_values,
    "scaler_age": scaler_age,
    "scaler_kms": scaler_kms,
    "currency": "INR",
    "valid_ranges": {
        "yr_mfr": [int(df["yr_mfr"].min()), int(REFERENCE_YEAR)],
        "kms_run": [float(df["kms_run"].min()), float(df["kms_run"].max())],
    },
}

# ---------- 3. 用同一个 transform 生成特征 ----------
X_train = transform(train_df, bundle)
X_test = transform(test_df, bundle)
y_train = train_df["sale_price"]
y_test = test_df["sale_price"]

print("特征列:", list(X_train.columns))

# ---------- 4. 训练 ----------
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    monotone_constraints=MONOTONE_CONSTRAINTS,
)
model.fit(X_train, y_train)

# ---------- 5. 评估 ----------
pred = model.predict(X_test)
metrics = {
    "r2": float(r2_score(y_test, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "mae": float(mean_absolute_error(y_test, pred)),
}
print(json.dumps(metrics, indent=2, ensure_ascii=False))

# ---------- 6. 保存 ----------
bundle["metrics"] = metrics
joblib.dump(bundle, BUNDLE_PATH)
joblib.dump(model, MODEL_PATH)
print(f"已保存: {MODEL_PATH}")
print(f"已保存: {BUNDLE_PATH}")