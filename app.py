from datetime import timezone
from pathlib import Path

import html
import joblib
import pandas as pd
import plotly.express as px
from fastapi import Depends, FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
import uvicorn

from analytics import dashboard_data, ensure_listings, filter_options
from database import Prediction, get_session, init_db
from features import CAT_COLS, transform

BASE_DIR = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")
MODEL = joblib.load(BASE_DIR / "models" / "car_price_model.pkl")

CURRENCY = BUNDLE.get("currency", "INR")
CURRENCY_SYMBOL = "₹" if CURRENCY == "INR" else "¥"

app = FastAPI(title="二手车价格预测")
init_db()
ensure_listings()

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
       padding: 40px 16px; background: #f5f7fa; color: #222; margin: 0; }
.container { max-width: 560px; margin: 0 auto; background: #fff; padding: 32px;
             border-radius: 12px; box-shadow: 0 2px 16px rgba(0,0,0,.08); }
.container.wide { max-width: 980px; }
h2 { margin-top: 0; }
label { display: block; margin-bottom: 14px; font-size: 14px; color: #555; }
select, input[type=number] { width: 100%; padding: 10px; margin-top: 6px; font-size: 15px;
       border: 1px solid #d0d7de; border-radius: 6px; background: #fff; }
.btn { width: 100%; margin-top: 8px; background: #2563eb; color: #fff; padding: 12px;
       border: 0; border-radius: 6px; cursor: pointer; font-size: 16px; }
.btn:hover { background: #1d4ed8; }
.result { margin: 18px 0; font-size: 32px; font-weight: 700; color: #16a34a; }
.hint { color: #666; font-size: 13px; line-height: 1.7; }
.error { background: #fef2f2; border: 1px solid #fecaca; color: #b91c1c;
         padding: 12px 14px; border-radius: 6px; margin-bottom: 16px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 16px; }
th, td { padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: left;
         white-space: nowrap; }
th { color: #6b7280; font-weight: 600; }
.feedback { display: flex; gap: 6px; align-items: center; }
input.inline { width: 100px; padding: 6px 8px; margin: 0; font-size: 13px; }
button.inline { min-width: 56px; padding: 7px 12px; font-size: 13px; white-space: nowrap;
                background: #2563eb; color: #fff; border: 0; border-radius: 4px;
                cursor: pointer; }
button.inline:hover { background: #1d4ed8; }
.ok { color: #16a34a; font-size: 13px; }
.nav { margin-top: 20px; font-size: 14px; }
.topnav { display: flex; gap: 18px; margin: 0 0 22px; padding-bottom: 12px;
          border-bottom: 1px solid #e5e7eb; font-size: 14px; }
.topnav a { color: #6b7280; text-decoration: none; }
.topnav a:hover { color: #2563eb; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0 8px; }
.kpi { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }
.kpi .label { font-size: 12px; color: #6b7280; }
.kpi .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
.filters { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; margin: 16px 0; }
.filters label { margin: 0; font-size: 12px; }
.filters select { width: auto; min-width: 140px; margin-top: 4px; padding: 7px 10px; font-size: 13px; }
.btn.small { width: auto; margin: 0; padding: 8px 16px; font-size: 13px; }
.reset { font-size: 13px; padding-bottom: 8px; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 8px; }
.chart-full { grid-column: 1 / -1; }
@media (max-width: 760px) {
  .chart-grid { grid-template-columns: 1fr; }
  .kpis { grid-template-columns: repeat(2, 1fr); }
}
a { color: #2563eb; }
"""


def render_page(content: str, status_code: int = 200, wide: bool = False) -> HTMLResponse:
    container_class = "container wide" if wide else "container"
    nav = """
<div class="topnav">
  <a href="/">价格预测</a>
  <a href="/history">历史记录</a>
  <a href="/insights">数据分析</a>
</div>
"""
    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>二手车价格预测</title>
<style>{CSS}</style>
</head>
<body>
<div class="{container_class}">{nav}{content}</div>
</body>
</html>"""
    return HTMLResponse(body, status_code=status_code)


def options_html(values, selected) -> str:
    parts = []
    for value in values:
        sel = " selected" if str(value) == str(selected) else ""
        parts.append(
            f'<option value="{html.escape(str(value))}"{sel}>'
            f"{html.escape(str(value).title())}</option>"
        )
    return "".join(parts)


def render_form(error_html: str = "", status_code: int = 200) -> HTMLResponse:
    mappings = BUNDLE["cat_mappings"]
    defaults = {
        "make": "maruti",
        "fuel_type": "petrol",
        "transmission": "manual",
        "body_type": "hatchback",
    }

    def pick(col: str) -> str:
        values = list(mappings[col].keys())
        default = defaults[col]
        return default if default in values else values[0]

    year_min, year_max = BUNDLE["valid_ranges"]["yr_mfr"]
    kms_min, kms_max = BUNDLE["valid_ranges"]["kms_run"]

    form = f"""
<h2>🚗 车辆信息</h2>
<p class="hint">价格口径：{BUNDLE["reference_year"]} 年印度市场挂牌价（INR），未做通胀调整。</p>
{f'<div class="error">{error_html}</div>' if error_html else ''}
<form method="post">
  <label>品牌
    <select name="make">{options_html(mappings["make"].keys(), pick("make"))}</select>
  </label>
  <label>制造年份
    <input type="number" name="yr_mfr" value="2018"
           min="{year_min}" max="{year_max}" step="1" required>
  </label>
  <label>行驶里程（km）
    <input type="number" name="kms_run" value="40000"
           min="{kms_min}" max="{kms_max}" step="100" required>
  </label>
  <label>燃油类型
    <select name="fuel_type">{options_html(mappings["fuel_type"].keys(), pick("fuel_type"))}</select>
  </label>
  <label>变速箱
    <select name="transmission">{options_html(mappings["transmission"].keys(), pick("transmission"))}</select>
  </label>
  <label>车身类型
    <select name="body_type">{options_html(mappings["body_type"].keys(), pick("body_type"))}</select>
  </label>
  <button class="btn" type="submit">🔮 预测价格</button>
</form>
"""
    return render_page(form, status_code)


@app.get("/", response_class=HTMLResponse)
async def home():
    return render_form()


@app.post("/", response_class=HTMLResponse)
def predict(
    make: str = Form(...),
    yr_mfr: int = Form(...),
    kms_run: float = Form(...),
    fuel_type: str = Form(...),
    transmission: str = Form(...),
    body_type: str = Form(...),
    session: Session = Depends(get_session),
):
    raw = {
        "make": make.strip().lower(),
        "yr_mfr": yr_mfr,
        "kms_run": kms_run,
        "fuel_type": fuel_type.strip().lower(),
        "transmission": transmission.strip().lower(),
        "body_type": body_type.strip().lower(),
    }

    errors = []
    for col in CAT_COLS:
        if raw[col] not in BUNDLE["cat_mappings"][col]:
            errors.append(f"不支持的{col}：{raw[col]}")

    year_min, year_max = BUNDLE["valid_ranges"]["yr_mfr"]
    if not (year_min <= yr_mfr <= year_max):
        errors.append(f"制造年份需在 {year_min}~{year_max} 之间")

    kms_min, kms_max = BUNDLE["valid_ranges"]["kms_run"]
    if not (kms_min <= kms_run <= kms_max):
        errors.append(f"里程需在 {kms_min:,.0f}~{kms_max:,.0f} km 之间")

    if errors:
        error_html = "<br>".join(html.escape(e) for e in errors)
        return render_form(error_html, status_code=400)

    try:
        X = transform(pd.DataFrame([raw]), BUNDLE)
        pred = float(MODEL.predict(X)[0])
    except Exception:
        import traceback
        traceback.print_exc()
        return render_form("预测失败，请稍后重试。", status_code=500)

    record = Prediction(**raw, predicted_price=pred)
    session.add(record)
    session.commit()
    session.refresh(record)

    content = f"""
<h2>🚗 预测结果</h2>
<div class="result">{CURRENCY_SYMBOL}{pred:,.0f}</div>
<p class="hint">
  {html.escape(raw["make"].title())} · {yr_mfr} 年 · {kms_run:,.0f} km ·
  {html.escape(raw["fuel_type"])} · {html.escape(raw["transmission"])} ·
  {html.escape(raw["body_type"])}
</p>
<p class="hint">数据集为 {BUNDLE["reference_year"]} 年印度市场挂牌数据，价格单位为印度卢比（INR），未做通胀调整。</p>
<p class="hint">记录已保存，编号 #{record.id}</p>
<p><a href="/">← 返回重新预测</a> · <a href="/history">查看历史记录 →</a></p>
"""
    return render_page(content)


def format_time(value) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone().strftime("%m-%d %H:%M")


@app.get("/history", response_class=HTMLResponse)
def history(session: Session = Depends(get_session)):
    records = session.scalars(
        select(Prediction).order_by(Prediction.id.desc()).limit(20)
    ).all()

    if not records:
        content = """
<h2>📋 预测历史</h2>
<p class="hint">还没有预测记录。</p>
<p><a href="/">← 去做一次预测</a></p>
"""
        return render_page(content)

    rows = []
    for r in records:
        has_actual = r.actual_price is not None
        value_attr = f' value="{r.actual_price:.0f}"' if has_actual else ""
        placeholder = "修改" if has_actual else "真实价"
        rows.append(f"""
<tr>
  <td>{r.id}</td>
  <td>{format_time(r.created_at)}</td>
  <td>{html.escape(str(r.make).title())}</td>
  <td>{r.yr_mfr}</td>
  <td>{r.kms_run:,.0f}</td>
  <td>{CURRENCY_SYMBOL}{r.predicted_price:,.0f}</td>
  <td>
    <form method="post" action="/feedback/{r.id}" class="feedback">
      <input class="inline" type="number" name="actual_price" min="1" step="1"
             placeholder="{placeholder}"{value_attr} required>
      <button class="inline" type="submit">保存</button>
    </form>
  </td>
</tr>""")

    content = f"""
<h2>📋 最近 20 条预测</h2>
<div class="table-wrap">
<table>
  <thead>
    <tr><th>#</th><th>时间</th><th>品牌</th><th>年份</th><th>里程</th>
        <th>预测价</th><th>真实价</th></tr>
  </thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</div>
<p class="nav"><a href="/">← 返回预测</a></p>
"""
    return render_page(content, wide=True)


@app.post("/feedback/{prediction_id}")
def feedback(
    prediction_id: int,
    actual_price: float = Form(...),
    session: Session = Depends(get_session),
):
    record = session.get(Prediction, prediction_id)
    if record is None:
        return render_page(
            "<h2>记录不存在</h2><p><a href='/history'>返回历史</a></p>",
            status_code=404,
        )
    if not (1 <= actual_price <= 10_000_000):
        return render_page(
            "<h2>真实价格不合法</h2><p><a href='/history'>返回历史</a></p>",
            status_code=400,
        )
    record.actual_price = actual_price
    session.commit()
    return RedirectResponse(url="/history", status_code=303)


def fig_html(fig, include_plotlyjs: bool = False) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        config={"displayModeBar": False, "responsive": False},
    )


@app.get("/insights", response_class=HTMLResponse)
def insights(make: str = "", fuel_type: str = ""):
    options = filter_options()
    make = make.strip().lower()
    fuel_type = fuel_type.strip().lower()
    if make not in options["makes"]:
        make = ""
    if fuel_type not in options["fuels"]:
        fuel_type = ""

    data = dashboard_data(make=make, fuel_type=fuel_type)

    make_options = '<option value="">全部品牌</option>' + options_html(options["makes"], make)
    fuel_options = '<option value="">全部燃油</option>' + options_html(options["fuels"], fuel_type)

    filters = f"""
<form method="get" action="/insights" class="filters">
  <label>品牌<select name="make">{make_options}</select></label>
  <label>燃油类型<select name="fuel_type">{fuel_options}</select></label>
  <button class="btn small" type="submit">筛选</button>
  <a class="reset" href="/insights">重置</a>
</form>
"""

    if data["n"] == 0:
        content = f"""
<h2>📊 数据分析看板</h2>
{filters}
<p class="hint">没有符合条件的数据，换一组筛选条件试试。</p>
"""
        return render_page(content, wide=True)

    kpis = f"""
<div class="kpis">
  <div class="kpi"><div class="label">挂牌量</div><div class="value">{data['n']:,}</div></div>
  <div class="kpi"><div class="label">中位价</div><div class="value">₹{data['median_price']:,.0f}</div></div>
  <div class="kpi"><div class="label">平均车龄</div><div class="value">{data['avg_age']:.1f} 年</div></div>
  <div class="kpi"><div class="label">平均里程</div><div class="value">{data['avg_kms']:,.0f} km</div></div>
</div>
"""

    fig_brand = px.bar(
        data["brand"], x="avg_price_wan", y="make_label", orientation="h", text="n",
        template="plotly_white",
        labels={"avg_price_wan": "平均价（万 INR）", "make_label": "", "n": "样本量"},
        title="品牌平均价（仅展示 n≥10）",
        color_discrete_sequence=["#0ea5e9"],
    )
    fig_brand.update_traces(
        texttemplate="n=%{text}", textposition="outside", cliponaxis=False
    )
    fig_brand.update_yaxes(automargin=True)
    fig_brand.update_xaxes(
        automargin=True,
        range=[0, float(data["brand"]["avg_price_wan"].max()) * 1.15],
    )
    fig_brand.update_layout(
        height=500,
        margin=dict(l=8, r=30, t=50, b=8),
        yaxis={"autorange": "reversed"},
    )

    fig_age = px.bar(
        data["age"], x="age_band", y="avg_price_wan", text="n",
        template="plotly_white",
        labels={"age_band": "车龄", "avg_price_wan": "平均价（万 INR）", "n": "样本量"},
        title="车龄与平均价",
        color_discrete_sequence=["#2563eb"],
    )
    fig_age.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig_age.update_layout(height=390, margin=dict(l=8, r=8, t=50, b=8))

    fig_km = px.bar(
        data["mileage"], x="km_band", y="avg_price_wan", text="n",
        template="plotly_white",
        labels={"km_band": "里程", "avg_price_wan": "平均价（万 INR）", "n": "样本量"},
        title="里程与平均价",
        color_discrete_sequence=["#8b5cf6"],
    )
    fig_km.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig_km.update_layout(height=390, margin=dict(l=8, r=8, t=50, b=8))

    fig_trend = px.line(
        data["trend"], x="ad_year", y="avg_price_wan", markers=True,
        template="plotly_white",
        labels={"ad_year": "挂牌年份", "avg_price_wan": "平均价（万 INR）"},
        title="挂牌年份与平均价",
        color_discrete_sequence=["#10b981"],
    )
    fig_trend.update_layout(height=390, margin=dict(l=8, r=8, t=50, b=8))

    fig_fuel = px.bar(
        data["fuel"], x="fuel_label", y="avg_price_wan", text="n",
        template="plotly_white",
        labels={"fuel_label": "燃油类型", "avg_price_wan": "平均价（万 INR）", "n": "样本量"},
        title="燃油类型与平均价",
        color_discrete_sequence=["#f59e0b"],
    )
    fig_fuel.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig_fuel.update_layout(height=390, margin=dict(l=8, r=8, t=50, b=8))

    charts = f"""
<div class="chart-grid">
  <div class="chart-full">{fig_html(fig_brand, include_plotlyjs=True)}</div>
  <div>{fig_html(fig_age)}</div>
  <div>{fig_html(fig_km)}</div>
  <div>{fig_html(fig_trend)}</div>
  <div>{fig_html(fig_fuel)}</div>
</div>
"""

    content = f"""
<h2>📊 数据分析看板</h2>
<p class="hint">数据口径：{BUNDLE["reference_year"]} 年印度市场挂牌价（INR），车龄按挂牌年份计算；所有统计由 SQLite 分组查询生成。</p>
{filters}
{kpis}
{charts}
<p class="hint">提示：品牌平均价只展示样本量 ≥ 10 的品牌；豪华品牌样本少，均值波动较大。</p>
<details class="hint" style="margin-top:24px">
  <summary>看板背后的 SQL 示例</summary>
  <pre>SELECT make, COUNT(*) AS n, AVG(sale_price) AS avg_price
FROM listings
GROUP BY make
HAVING n &gt;= 10
ORDER BY avg_price DESC;</pre>
</details>
"""
    return render_page(content, wide=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "reference_year": BUNDLE["reference_year"],
    }


if __name__ == "__main__":
    print("打开 http://127.0.0.1:8002")
    uvicorn.run(app, host="127.0.0.1", port=8002)
