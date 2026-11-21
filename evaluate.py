"""可复现的模型评估：5 折交叉验证、baseline 对比、SHAP 可解释性。"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from features import (
    CAT_COLS,
    FEATURE_COLS,
    add_base_features,
    add_condition_score,
    clean_categoricals,
    encode_categoricals,
    transform,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "car_data.csv"
MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"
BUNDLE_PATH = BASE_DIR / "models" / "preprocessor.pkl"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
MONOTONE_CONSTRAINTS = (0, 0, 0, 0, -1, -1, -1, -1, 1)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["ad_year"] = pd.to_datetime(df["ad_created_on"], errors="coerce").dt.year
    df = df[df["ad_year"].notna()].drop_duplicates()
    df = df[(df["sale_price"] > 0) & (df["sale_price"] < 10_000_000) & (df["kms_run"] > 0)]
    df = df[(df["yr_mfr"] >= 1980) & (df["yr_mfr"] <= df["ad_year"])]
    return clean_categoricals(df)


def prepare_fold(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    for col in CAT_COLS:
        mode = train_df[col].mode().iloc[0]
        train_df[col] = train_df[col].fillna(mode)
        valid_df[col] = valid_df[col].fillna(mode)

    train_df = add_base_features(train_df, 2021)
    valid_df = add_base_features(valid_df, 2021)

    scaler_age = MinMaxScaler().fit(train_df[["Car_Age"]])
    scaler_kms = MinMaxScaler().fit(train_df[["kms_run"]])
    train_df = add_condition_score(train_df, scaler_age, scaler_kms)
    valid_df = add_condition_score(valid_df, scaler_age, scaler_kms)

    mappings = {
        col: {value: i for i, value in enumerate(sorted(train_df[col].dropna().unique().tolist()))}
        for col in CAT_COLS
    }
    train_df = encode_categoricals(train_df, mappings)
    valid_df = encode_categoricals(valid_df, mappings)

    return (
        train_df[FEATURE_COLS],
        valid_df[FEATURE_COLS],
        train_df["sale_price"],
        valid_df["sale_price"],
    )


def build_models() -> dict:
    return {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost (no constraint)": xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost (monotone)": xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            monotone_constraints=MONOTONE_CONSTRAINTS,
        ),
    }


def calc_metrics(y_true, y_pred) -> dict:
    ape = np.abs((y_true - y_pred) / y_true)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(ape) * 100),
        "mdape": float(np.median(ape) * 100),
    }


def cross_validate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in build_models():
        fold_metrics = []
        for train_idx, valid_idx in KFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ).split(df):
            X_train, X_valid, y_train, y_valid = prepare_fold(
                df.iloc[train_idx], df.iloc[valid_idx]
            )
            model = build_models()[name]
            model.fit(X_train, y_train)
            fold_metrics.append(calc_metrics(y_valid, model.predict(X_valid)))

        row = {"model": name}
        for key in ["r2", "rmse", "mae", "mape", "mdape"]:
            values = [item[key] for item in fold_metrics]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)


def holdout_saved_model(df: pd.DataFrame) -> dict:
    _, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    bundle = joblib.load(BUNDLE_PATH)
    model = joblib.load(MODEL_PATH)
    pred = model.predict(transform(test_df, bundle))
    return calc_metrics(test_df["sale_price"], pred)


def shap_analysis(df: pd.DataFrame) -> None:
    bundle = joblib.load(BUNDLE_PATH)
    model = joblib.load(MODEL_PATH)

    sample = df.sample(n=min(500, len(df)), random_state=RANDOM_STATE)
    X = transform(sample, bundle)

    # shap 0.49 与 XGBoost 3.x 的 TreeExplainer 存在 base_score 兼容问题，
    # 改用 XGBoost 内置的 pred_contribs（同样是精确的 TreeSHAP）。
    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    shap_values = contribs[:, :-1]  # 最后一列是 bias

    label_map = {
        "make_encoded": "品牌",
        "fuel_type_encoded": "燃油类型",
        "transmission_encoded": "变速箱",
        "body_type_encoded": "车身类型",
        "kms_run": "里程",
        "Car_Age": "车龄",
        "Avg_Mileage_Per_Year": "年均里程",
        "Age_Mileage_Interaction": "车龄×里程",
        "Condition_Score": "车况评分",
    }
    X_display = X.rename(columns=label_map)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_display, show=False, max_display=9)
    plt.xlabel("SHAP 值（对预测价格的影响，INR）")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_display, plot_type="bar", show=False, max_display=9
    )
    plt.xlabel("平均 |SHAP 值|（INR）")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_data()
    print(f"评估数据: {len(df)} 行, {N_FOLDS} 折交叉验证")

    comparison = cross_validate(df)
    holdout = holdout_saved_model(df)
    shap_analysis(df)

    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    payload = {
        "n_rows": len(df),
        "n_folds": N_FOLDS,
        "reference_year": int(df["ad_year"].mode().iloc[0]),
        "holdout_saved_model": holdout,
        "cross_validation": comparison.to_dict(orient="records"),
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n5 折交叉验证结果：")
    print(comparison.to_string(index=False))
    print("\n当前部署模型（20% holdout）：")
    print(json.dumps(holdout, ensure_ascii=False, indent=2))
    print("\n已保存：reports/model_comparison.csv, reports/metrics.json")
    print("已保存：reports/figures/08_shap_summary.png, 09_shap_bar.png")


if __name__ == "__main__":
    main()
