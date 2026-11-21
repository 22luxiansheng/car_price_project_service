"""训练与线上推理共用的特征处理。两边都调用这里，避免逻辑走样。"""
import pandas as pd

CAT_COLS = ["make", "fuel_type", "transmission", "body_type"]
NUM_COLS = ["kms_run", "Car_Age", "Avg_Mileage_Per_Year", "Age_Mileage_Interaction"]
FEATURE_COLS = [f"{c}_encoded" for c in CAT_COLS] + NUM_COLS + ["Condition_Score"]


def clean_categoricals(df):
    """统一大小写和空格，避免 'Toyota' 和 'toyota' 被当成两个品牌。"""
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("string").str.strip().str.lower()
    return df


def add_base_features(df, reference_year):
    df = df.copy()
    df["yr_mfr"] = pd.to_numeric(df["yr_mfr"], errors="coerce")
    df["kms_run"] = pd.to_numeric(df["kms_run"], errors="coerce")

    if "ad_year" in df.columns:
        # 训练数据：用挂牌年份计算车龄，避免把 2021 年的价格当成 2026 年的车
        age_base = pd.to_numeric(df["ad_year"], errors="coerce")
    else:
        # 线上推理：没有挂牌年份，用训练时保存的参考年份
        age_base = reference_year

    df["Car_Age"] = age_base - df["yr_mfr"]
    df["Avg_Mileage_Per_Year"] = df["kms_run"] / (df["Car_Age"] + 1)
    df["Age_Mileage_Interaction"] = df["Car_Age"] * df["kms_run"] / 10000
    return df


def add_condition_score(df, scaler_age, scaler_kms):
    """用训练集拟合好的 scaler 做 transform，单行输入也不会变成常量 1。"""
    df = df.copy()
    df["Age_Score"] = 1 - scaler_age.transform(df[["Car_Age"]])
    df["Mileage_Score"] = 1 - scaler_kms.transform(df[["kms_run"]])
    df["Condition_Score"] = (df["Age_Score"] + df["Mileage_Score"]) / 2
    return df


def encode_categoricals(df, mappings):
    """用训练时保存的映射编码；没见过的类别给 -1，而不是让程序报错。"""
    df = df.copy()
    for c, mapping in mappings.items():
        df[f"{c}_encoded"] = df[c].map(mapping).fillna(-1).astype(int)
    return df


def transform(raw_df, bundle):
    """
    训练和推理都走这一个函数。
    raw_df 只需要包含：make/fuel_type/transmission/body_type/yr_mfr/kms_run
    """
    df = clean_categoricals(raw_df)
    for col, value in bundle.get("cat_fill_values", {}).items():
        df[col] = df[col].fillna(value)
    df = add_base_features(df, bundle["reference_year"])
    df = add_condition_score(df, bundle["scaler_age"], bundle["scaler_kms"])
    df = encode_categoricals(df, bundle["cat_mappings"])
    return df[bundle["feature_cols"]]