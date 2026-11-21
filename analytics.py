"""数据分析层：把原始 CSV 装载到 SQLite，并用 SQL 做分组统计。"""
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "raw" / "car_data.csv"
DB_PATH = BASE_DIR / "data" / "analytics.db"

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)

CAT_COLS = ["make", "fuel_type", "transmission", "body_type"]
COLUMNS = [
    "make", "model", "yr_mfr", "ad_year", "car_age", "kms_run",
    "fuel_type", "transmission", "body_type", "sale_price", "price_wan",
]


def build_listings() -> int:
    """重新读取 CSV 并覆盖 listings 表。"""
    df = pd.read_csv(CSV_PATH)
    df["ad_year"] = pd.to_datetime(df["ad_created_on"], errors="coerce").dt.year
    df = df[df["ad_year"].notna()].drop_duplicates()
    df = df[(df["sale_price"] > 0) & (df["sale_price"] < 10_000_000) & (df["kms_run"] > 0)]
    df = df[(df["yr_mfr"] >= 1980) & (df["yr_mfr"] <= df["ad_year"])]

    for col in CAT_COLS:
        df[col] = df[col].astype("string").str.strip().str.lower().fillna("unknown")

    df["car_age"] = df["ad_year"] - df["yr_mfr"]
    df["price_wan"] = df["sale_price"] / 10_000

    df[COLUMNS].to_sql("listings", engine, if_exists="replace", index=False)
    return len(df)


def ensure_listings() -> int:
    """表不存在或为空时自动导入，方便第一次启动。"""
    with engine.connect() as conn:
        table = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='listings'")
        ).first()
        if table is None:
            return build_listings()
        count = conn.execute(text("SELECT COUNT(*) FROM listings")).scalar_one()
    if not count:
        return build_listings()
    return int(count)


def read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def filter_options() -> dict:
    makes = read_sql("SELECT DISTINCT make FROM listings ORDER BY make")["make"].tolist()
    fuels = read_sql("SELECT DISTINCT fuel_type FROM listings ORDER BY fuel_type")["fuel_type"].tolist()
    return {"makes": makes, "fuels": fuels}


def where_clause(make: str = "", fuel_type: str = "") -> tuple[str, dict]:
    clauses = ["1=1"]
    params = {}
    if make:
        clauses.append("make = :make")
        params["make"] = make
    if fuel_type:
        clauses.append("fuel_type = :fuel_type")
        params["fuel_type"] = fuel_type
    return " AND ".join(clauses), params


def dashboard_data(make: str = "", fuel_type: str = "") -> dict:
    where, params = where_clause(make, fuel_type)

    kpi = read_sql(
        f"""
        SELECT COUNT(*) AS n,
               AVG(sale_price) AS avg_price,
               AVG(car_age) AS avg_age,
               AVG(kms_run) AS avg_kms
        FROM listings
        WHERE {where}
        """,
        params,
    ).iloc[0]

    prices = read_sql(
        f"SELECT sale_price FROM listings WHERE {where}", params
    )["sale_price"]

    brand = read_sql(
        f"""
        SELECT make, COUNT(*) AS n, AVG(sale_price) AS avg_price
        FROM listings
        WHERE {where}
        GROUP BY make
        HAVING n >= 10
        ORDER BY avg_price DESC
        LIMIT 12
        """,
        params,
    )
    brand["avg_price_wan"] = brand["avg_price"] / 10_000
    brand["make_label"] = (
        brand["make"].str.title().replace({"Bmw": "BMW", "Mg": "MG"})
    )

    age = read_sql(
        f"""
        SELECT CASE
                 WHEN car_age <= 3 THEN '0-3年'
                 WHEN car_age <= 6 THEN '4-6年'
                 WHEN car_age <= 10 THEN '7-10年'
                 WHEN car_age <= 15 THEN '11-15年'
                 ELSE '15年以上'
               END AS age_band,
               MIN(car_age) AS sort_key,
               COUNT(*) AS n,
               AVG(sale_price) AS avg_price
        FROM listings
        WHERE {where}
        GROUP BY age_band
        ORDER BY sort_key
        """,
        params,
    )
    age["avg_price_wan"] = age["avg_price"] / 10_000

    mileage = read_sql(
        f"""
        SELECT CASE
                 WHEN kms_run < 20000 THEN '0-2万'
                 WHEN kms_run < 50000 THEN '2-5万'
                 WHEN kms_run < 100000 THEN '5-10万'
                 WHEN kms_run < 150000 THEN '10-15万'
                 ELSE '15万+'
               END AS km_band,
               MIN(kms_run) AS sort_key,
               COUNT(*) AS n,
               AVG(sale_price) AS avg_price
        FROM listings
        WHERE {where}
        GROUP BY km_band
        ORDER BY sort_key
        """,
        params,
    )
    mileage["avg_price_wan"] = mileage["avg_price"] / 10_000

    trend = read_sql(
        f"""
        SELECT ad_year, COUNT(*) AS n, AVG(sale_price) AS avg_price
        FROM listings
        WHERE {where}
        GROUP BY ad_year
        ORDER BY ad_year
        """,
        params,
    )
    trend["avg_price_wan"] = trend["avg_price"] / 10_000

    fuel = read_sql(
        f"""
        SELECT fuel_type, COUNT(*) AS n, AVG(sale_price) AS avg_price
        FROM listings
        WHERE {where}
        GROUP BY fuel_type
        ORDER BY n DESC
        """,
        params,
    )
    fuel["avg_price_wan"] = fuel["avg_price"] / 10_000
    fuel["fuel_label"] = fuel["fuel_type"].str.title().replace(
        {"Petrol & Cng": "Petrol & CNG", "Petrol & Lpg": "Petrol & LPG"}
    )

    return {
        "n": int(kpi["n"]),
        "avg_price": float(kpi["avg_price"] or 0),
        "median_price": float(prices.median()) if len(prices) else 0.0,
        "avg_age": float(kpi["avg_age"] or 0),
        "avg_kms": float(kpi["avg_kms"] or 0),
        "brand": brand,
        "age": age,
        "mileage": mileage,
        "trend": trend,
        "fuel": fuel,
    }


if __name__ == "__main__":
    print("已写入 listings 表:", build_listings())
