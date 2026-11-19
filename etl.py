#!/usr/bin/env python3
r"""
Nightly transaction-log cleaning pipeline.

Reads a raw transaction CSV with pandas, validates and cleans it, writes the
clean dataset to CSV, and puts *every* view/table/chart into one standalone
self-contained HTML report (no external CSS/JS/CDN needed).

Default input : /home/user/uploads/CDNOW_master.csv
Default output: /home/user/clean_transactions.csv
                /home/user/etl.html
                /home/user/model_input.csv
                /home/user/pnbd_input.csv
                /home/user/gamma_gamma_input.csv
                /home/user/validation_report.csv

A separate validation-only stage (validate_model_input.py) builds the exact
per-customer schema for pymc-marketing Pareto/NBD and Gamma-Gamma models and
checks columns, dtypes, nulls, uniqueness and value invariants before any
model execution. NO model is created or fitted.

Usage
-----
    python3 etl.py
    python3 etl.py --input /path/to/nightly_log.csv \
                                    --output-dir /path/to/output

Cron example (nightly at 02:15):
    15 2 * * * /usr/bin/python3 /home/user/etl.py \
        --input /data/logs/transactions_$(date +\%Y\%m\%d).csv \
        --output-dir /data/cleaned >> /var/log/txn_pipeline.log 2>&1

Cleaning rules (configurable constants below)
----------------------------------------------
* Invalid/missing customer id, date, quantity or spend -> removed.
* Quantity <= 0 (negative quantity = return/cancellation) -> removed.
* Spend <= 0 (zero/negative charge = void/cancel/free row) -> removed.
* Exact duplicate transactions (same customer/date/quantity/spend) -> removed.
* If a status-like column exists in a future log, canceled/failed/refunded/
  pending/etc. statuses are removed as well; only settled/completed statuses
  survive. Unknown statuses are dropped conservatively.
* Extreme spend outliers are FLAGGED in the report but kept by default; pass
  --drop-outliers to remove values above Q3 + 3*IQR.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # headless: charts are embedded in HTML, never shown here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from validate_model_input import validate_and_export  # validation only; no model fitting

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_INPUT = "/home/user/uploads/CDNOW_master.csv"
DEFAULT_OUTPUT_DIR = "/home/user"
CLEAN_CSV_NAME = "clean_transactions.csv"
REPORT_HTML_NAME = "etl.html"

COLUMN_ALIASES = {
    "customer_id": "cust_id", "client_id": "cust_id", "user_id": "cust_id",
    "custid": "cust_id", "cust": "cust_id", "id": "cust_id",
    "order_date": "date", "transaction_date": "date", "txn_date": "date",
    "event_date": "date", "purchase_date": "date", "day": "date",
    "qty": "quantity", "units": "quantity", "num_units": "quantity",
    "amount": "spend", "total": "spend", "revenue": "spend",
    "order_value": "spend", "price": "spend", "spend_amount": "spend",
}
REQUIRED_COLUMNS = ["cust_id", "date", "quantity", "spend"]

STATUS_SUCCESS = {
    "completed", "complete", "success", "successful", "settled", "paid",
    "approved", "captured", "processed", "ok",
}
STATUS_BAD = {
    "canceled", "cancelled", "cancel", "failed", "failure", "declined",
    "refunded", "refund", "chargeback", "void", "voided", "reversed",
    "expired", "rejected", "pending", "error", "fraud", "returned", "return",
}
STATUS_LIKE_COLUMNS = {
    "status", "txn_status", "transaction_status", "payment_status",
    "order_status", "state", "result",
}
CANCEL_FLAG_COLUMNS = {
    "is_canceled", "is_cancelled", "cancelled", "canceled", "cancel_flag",
    "is_refunded", "refunded", "is_void", "void",
}
CURRENCY_SYMBOLS = "$€£₹¥, "

PALETTE = {
    "primary": "#2563eb", "accent": "#0ea5e9", "good": "#059669",
    "bad": "#dc2626", "warn": "#d97706", "ink": "#0f172a",
    "muted": "#64748b", "grid": "#e2e8f0", "bg": "#f8fafc",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_raw(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (typed preview frame, all-string working frame)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input dataset not found: {path}")
    preview = pd.read_csv(path)
    work = pd.read_csv(path, dtype=str, keep_default_na=False)
    work.columns = [str(c).strip().lower().replace(" ", "_") for c in work.columns]
    work = work.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in work.columns})
    missing = [c for c in REQUIRED_COLUMNS if c not in work.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Found: {list(work.columns)}. "
            "Add the source name to COLUMN_ALIASES if your log uses different headers."
        )
    return preview, work


def parse_dates(series: pd.Series) -> pd.Series:
    """Accept YYYYMMDD, ISO dates, or common US/EU formats."""
    text = series.astype(str).str.strip()
    compact = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    mixed = pd.to_datetime(text, format="mixed", errors="coerce")
    return compact.fillna(mixed)


def parse_money(series: pd.Series) -> pd.Series:
    """Handle '$1,234.50', '(45.00)', 'N/A', and plain numbers."""
    text = series.astype(str).str.strip()
    negative = text.str.startswith("(") & text.str.endswith(")")
    cleaned = (
        text.str.replace(r"[\"'\s]", "", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    cleaned = cleaned.mask(cleaned == "", np.nan)
    values = pd.to_numeric(cleaned, errors="coerce")
    return values.mask(negative, -values.abs())


def find_status_column(work: pd.DataFrame) -> str | None:
    for col in work.columns:
        if col in STATUS_LIKE_COLUMNS or col in CANCEL_FLAG_COLUMNS:
            return col
    return None


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_data(
    work: pd.DataFrame, drop_outliers: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], dict]:
    raw_columns = list(work.columns)
    df = work.copy()
    log: list[dict] = []
    removed_parts: list[pd.DataFrame] = []
    stats: dict = {}

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    def drop(mask: pd.Series, reason: str, step: str) -> None:
        nonlocal df
        mask = mask.fillna(False).astype(bool)
        n = int(mask.sum())
        if n:
            bad = df.loc[mask, raw_columns].copy()
            bad.insert(0, "removal_reason", reason)
            bad.insert(0, "step", step)
            removed_parts.append(bad)
        log.append({
            "step": step,
            "reason": reason,
            "rows_removed": n,
            "rows_remaining": len(df) - n,
        })
        df = df.loc[~mask].copy()

    initial_rows = len(df)
    log.append({
        "step": "0. Load raw log",
        "reason": "Rows read from CSV with pandas",
        "rows_removed": 0,
        "rows_remaining": initial_rows,
    })

    # --- typed fields -----------------------------------------------------
    df["cust_id"] = pd.to_numeric(df["cust_id"], errors="coerce")
    drop(df["cust_id"].isna(), "invalid_or_missing_customer_id", "1. Validate customer id")

    df["date"] = parse_dates(df["date"])
    drop(df["date"].isna(), "invalid_or_missing_date", "2. Validate date")

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    drop(df["quantity"].isna(), "invalid_or_missing_quantity", "3. Validate quantity")
    drop(
        (df["quantity"] <= 0) & df["quantity"].notna(),
        "non_positive_quantity_(return/cancel)",
        "4. Remove canceled/return quantities",
    )

    df["spend"] = parse_money(df["spend"])
    drop(df["spend"].isna(), "invalid_or_missing_spend", "5. Validate spend")
    drop(
        (df["spend"] <= 0) & df["spend"].notna(),
        "non_positive_spend_(void/cancel/free)",
        "6. Remove void/canceled amounts",
    )

    # --- status column in richer nightly logs -----------------------------
    status_col = find_status_column(work)
    if status_col and status_col in df.columns:
        if status_col in CANCEL_FLAG_COLUMNS:
            raw_flag = df[status_col].astype(str).str.strip().str.lower()
            bad_flag = raw_flag.isin({"1", "true", "yes", "y", "t"})
            good_flag = raw_flag.isin({"0", "false", "no", "n", "f"})
            df["status"] = np.where(bad_flag, "canceled", "completed")
            drop(bad_flag, f"canceled_flag:{status_col}", "7. Remove canceled flags")
            drop(~good_flag & (df["spend"].notna()),
                 f"unknown_flag:{status_col}", "7. Remove canceled flags")
        else:
            norm = df[status_col].astype(str).str.strip().str.lower()
            mapped = norm.apply(lambda s: "completed" if s in STATUS_SUCCESS
                                else ("canceled/bad" if s in STATUS_BAD else "unknown"))
            stats["status_counts"] = norm.value_counts().to_dict()
            drop(mapped != "completed", f"status_not_settled:{status_col}",
                 "7. Remove canceled/failed statuses")
            df["status"] = "completed"
        stats["status_column"] = status_col
    else:
        stats["status_column"] = None
        log.append({
            "step": "7. Status check",
            "reason": "No status/cancel column in source; non-positive amounts "
                      "are used as the cancel/void signal",
            "rows_removed": 0,
            "rows_remaining": len(df),
        })

    # --- duplicates --------------------------------------------------------
    key = [c for c in ["cust_id", "date", "quantity", "spend"] if c in df.columns]
    dup_mask = df.duplicated(subset=key, keep="first")
    drop(dup_mask, "duplicate_transaction", "8. Remove duplicate transactions")

    # --- outliers (flag by default, optional removal) ---------------------
    q1, q3 = df["spend"].quantile(0.25), df["spend"].quantile(0.75)
    upper_fence = q3 + 3.0 * (q3 - q1)
    outlier_mask = df["spend"] > upper_fence
    stats["outlier_fence"] = float(upper_fence)
    stats["outliers_flagged"] = int(outlier_mask.sum())
    if drop_outliers:
        drop(outlier_mask, "extreme_spend_outlier", "9. Remove spend outliers")
    else:
        log.append({
            "step": "9. Outlier check",
            "reason": f"Flagged {int(outlier_mask.sum())} values above "
                      f"{upper_fence:,.2f} (kept; use --drop-outliers to remove)",
            "rows_removed": 0,
            "rows_remaining": len(df),
        })

    # --- final clean frame -------------------------------------------------
    clean = df.copy()
    clean["cust_id"] = clean["cust_id"].astype("int64")
    clean["quantity"] = clean["quantity"].astype("int64")
    clean["spend"] = clean["spend"].round(2)
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    clean["year_month"] = pd.to_datetime(clean["date"]).dt.strftime("%Y-%m")
    clean["unit_price"] = (clean["spend"] / clean["quantity"]).round(2)
    clean["high_value_flag"] = outlier_mask.reindex(df.index).fillna(False).map(
        {True: "YES", False: "NO"})
    ordered = ["cust_id", "date", "year_month", "quantity", "unit_price", "spend"]
    if "status" in clean.columns:
        ordered.append("status")
    ordered.append("high_value_flag")
    clean = clean[ordered].reset_index(drop=True)

    removed = (pd.concat(removed_parts, ignore_index=True)
               if removed_parts else pd.DataFrame(columns=["step", "removal_reason"]))
    stats["initial_rows"] = initial_rows
    stats["final_rows"] = len(clean)
    stats["removed_rows"] = initial_rows - len(clean)
    return clean, removed, pd.DataFrame(log), log, stats


# --------------------------------------------------------------------------- #
# Charts -> base64 PNG
# --------------------------------------------------------------------------- #
def fig_to_img(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)


def build_charts(clean: pd.DataFrame, removed: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return list of (title, caption, base64 image)."""
    charts: list[tuple[str, str, str]] = []
    if clean.empty:
        return charts

    monthly = (
        clean.groupby("year_month", as_index=False)
        .agg(revenue=("spend", "sum"), orders=("spend", "size"))
        .sort_values("year_month")
    )
    x = pd.PeriodIndex(monthly["year_month"], freq="M").astype(str)

    # 1. Monthly revenue trend
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    ax.fill_between(range(len(x)), monthly["revenue"], color=PALETTE["accent"], alpha=0.14)
    ax.plot(range(len(x)), monthly["revenue"], color=PALETTE["primary"],
            marker="o", linewidth=2.4, markersize=5)
    ax.set_xticks(range(len(x)), x, rotation=35, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax.set_title("Revenue by month", color=PALETTE["ink"], fontsize=12,
                 fontweight="bold", loc="left")
    style_axes(ax)
    charts.append(("Revenue trend", "Total spend of valid transactions per month.",
                   fig_to_img(fig)))

    # 2. Monthly order volume
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    ax.bar(range(len(x)), monthly["orders"], color=PALETTE["primary"],
           alpha=0.85, width=0.62)
    ax.set_xticks(range(len(x)), x, rotation=35, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Valid orders by month", color=PALETTE["ink"], fontsize=12,
                 fontweight="bold", loc="left")
    style_axes(ax)
    charts.append(("Order volume", "Count of transactions that passed every cleaning rule.",
                   fig_to_img(fig)))

    # 3. Spend distribution (log scale)
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    ax.hist(np.log10(clean["spend"] + 1), bins=45, color=PALETTE["accent"],
            edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.set_xticks(np.log10(np.array([1, 5, 10, 25, 50, 100, 250, 500, 1000]) + 1))
    ax.set_xticklabels(["$1", "$5", "$10", "$25", "$50", "$100", "$250", "$500", "$1k"])
    ax.set_title("Transaction value distribution", color=PALETTE["ink"],
                 fontsize=12, fontweight="bold", loc="left")
    style_axes(ax)
    charts.append(("Spend distribution", "Log-spaced x-axis: shape of order values after cleaning.",
                   fig_to_img(fig)))

    # 4. Top customers
    top = (clean.groupby("cust_id", as_index=False)["spend"].sum()
           .nlargest(10, "spend").sort_values("spend"))
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    ax.barh(top["cust_id"].astype(str), top["spend"], color=PALETTE["good"], alpha=0.9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_title("Top 10 customers by revenue", color=PALETTE["ink"],
                 fontsize=12, fontweight="bold", loc="left")
    style_axes(ax)
    charts.append(("Top customers", "Highest lifetime spend among cleaned transactions.",
                   fig_to_img(fig)))

    # 5. Units per order
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    counts = clean["quantity"].value_counts().sort_index().head(20)
    ax.bar(counts.index.astype(str), counts.values, color=PALETTE["primary"], alpha=0.85)
    ax.set_xlabel("units per transaction", fontsize=9, color=PALETTE["muted"])
    ax.set_title("Quantity per transaction", color=PALETTE["ink"],
                 fontsize=12, fontweight="bold", loc="left")
    style_axes(ax)
    charts.append(("Quantity mix", "How many units were purchased per valid transaction.",
                   fig_to_img(fig)))

    # 6. Removed rows by reason
    if not removed.empty:
        reasons = removed["removal_reason"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(9.4, 3.5))
        bars = ax.barh(reasons.index, reasons.values, color=PALETTE["bad"], alpha=0.85)
        for bar, value in zip(bars, reasons.values):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,}",
                    va="center", fontsize=9, color=PALETTE["ink"])
        ax.set_xlim(0, reasons.max() * 1.18)
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_title("Rows removed by cleaning rule", color=PALETTE["ink"],
                     fontsize=12, fontweight="bold", loc="left")
        style_axes(ax)
        charts.append(("Data quality", "Why raw records were rejected before export.",
                       fig_to_img(fig)))
    return charts


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def df_table(df: pd.DataFrame, max_rows: int | None = None, cls: str = "data-table") -> str:
    shown = df if max_rows is None else df.head(max_rows)
    return shown.to_html(escape=True, index=False, classes=cls, border=0,
                         justify="left")


def build_report(
    preview: pd.DataFrame,
    work: pd.DataFrame,
    clean: pd.DataFrame,
    removed: pd.DataFrame,
    log_df: pd.DataFrame,
    stats: dict,
    charts: list[tuple[str, str, str]],
    input_path: str,
    output_csv: str,
    drop_outliers: bool,
    validation: dict | None = None,
) -> str:
    generated = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    total = stats["initial_rows"]
    kept = stats["final_rows"]
    dropped = stats["removed_rows"]
    retention = 100 * kept / total if total else 0
    revenue = clean["spend"].sum() if not clean.empty else 0
    customers = clean["cust_id"].nunique() if not clean.empty else 0
    date_min = clean["date"].min() if not clean.empty else "—"
    date_max = clean["date"].max() if not clean.empty else "—"

    dtype_info = pd.DataFrame({
        "column": preview.columns,
        "pandas_dtype": [str(t) for t in preview.dtypes],
        "non_null": preview.notna().sum().values,
        "null": preview.isna().sum().values,
        "unique": preview.nunique(dropna=True).values,
    })
    numeric_cols = preview.select_dtypes(include="number").columns
    describe = preview[numeric_cols].describe().T.reset_index().rename(
        columns={"index": "column"}) if len(numeric_cols) else pd.DataFrame()

    reasons = (removed["removal_reason"].value_counts().rename_axis("removal_reason")
               .reset_index(name="rows") if not removed.empty else pd.DataFrame())
    status_note = (
        f"Status column <code>{html.escape(str(stats['status_column']))}</code> "
        "found: non-settled statuses removed." if stats.get("status_column")
        else "No status/cancel column in the source. Zero or negative spend and "
             "non-positive quantities are treated as canceled/void rows."
    )
    outlier_note = (
        f"Extreme values above {stats.get('outlier_fence', 0):,.2f} were REMOVED "
        f"({stats.get('outliers_flagged', 0):,} flagged)."
        if drop_outliers else
        f"{stats.get('outliers_flagged', 0):,} high-value rows were FLAGGED, "
        "not removed (high spend with high quantity is plausible here)."
    )

    chart_html = "\n".join(
        f"""
        <article class="chart-card">
          <h3>{html.escape(title)}</h3>
          <p class="caption">{html.escape(caption)}</p>
          <img src="data:image/png;base64,{img}" alt="{html.escape(title)} chart">
        </article>""" for title, caption, img in charts
    ) or '<p class="caption">No charts: the clean dataset is empty.</p>'

    if validation:
        checks = validation["checks"]
        statuses = checks["status"].value_counts().to_dict()
        validation_html = f"""
<h2><span class="num">06</span> Model-input validation (Pareto/NBD + Gamma-Gamma)</h2>
<p class="caption">Validation runs after cleaning and <strong>before any model
  execution</strong> — no model is created here. Checks: {len(checks)} ·
  PASS {statuses.get('PASS', 0)} · WARN {statuses.get('WARN', 0)} ·
  FAIL {statuses.get('FAIL', 0)}.
  Required exact types: <code>customer_id:int64</code>,
  <code>frequency/recency/T/monetary_value:float64</code>.
  Recency and T are in {html.escape(str(validation.get('time_unit', 'weeks')))}.</p>
<div class="panel"><div class="table-scroll">{df_table(checks)}</div></div>
<p class="caption" style="margin-top:12px">Validated CSV exports:</p>
<ul class="caption">
  <li><code>model_input.csv</code> — all customers, exact 5-column schema</li>
  <li><code>pnbd_input.csv</code> — customer_id, frequency, recency, T</li>
  <li><code>gamma_gamma_input.csv</code> — repeat buyers only: customer_id, frequency, monetary_value</li>
  <li><code>validation_report.csv</code> — this check table as CSV</li>
</ul>"""
    else:
        validation_html = ""

    css = """
    :root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--blue:#2563eb;
          --green:#059669;--red:#dc2626;--bg:#f8fafc;}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);
         font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
    .wrap{max-width:1180px;margin:0 auto;padding:28px 22px 70px}
    header.hero{background:linear-gradient(135deg,#1e3a8a,#2563eb 55%,#0ea5e9);
         color:#fff;border-radius:16px;padding:28px 32px;box-shadow:0 12px 30px #1e3a8a26}
    header.hero h1{margin:0 0 6px;font-size:27px;letter-spacing:-.4px}
    header.hero p{margin:3px 0;color:#dbeafe;font-size:13px}
    code{background:#eef2ff;color:#3730a3;padding:1px 5px;border-radius:4px;font-size:12px}
    h2{font-size:19px;margin:38px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
    h2 .num{color:var(--blue);margin-right:8px}
    h3{font-size:14px;margin:0 0 2px}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:20px 0}
    .card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:15px 17px;
          box-shadow:0 2px 8px #0f172a0a}
    .card .label{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted)}
    .card .value{font-size:23px;font-weight:700;margin-top:4px;letter-spacing:-.5px}
    .card.good .value{color:var(--green)} .card.bad .value{color:var(--red)}
    .panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px 4px 4px;
           overflow:hidden}
    .table-scroll{overflow:auto;max-height:440px;margin:8px; border-radius:8px}
    table.data-table{border-collapse:collapse;width:100%;font-size:12px}
    table.data-table thead th{position:sticky;top:0;background:#f1f5f9;color:#334155;
        text-align:left;padding:9px 10px;border-bottom:2px solid var(--line);white-space:nowrap;z-index:1}
    table.data-table tbody td{padding:7px 10px;border-bottom:1px solid #f1f5f9;white-space:nowrap}
    table.data-table tbody tr:hover{background:#f8fafc}
    table.data-table tbody tr:nth-child(even){background:#fcfdff}
    .note{background:#eff6ff;border-left:4px solid var(--blue);padding:11px 14px;
          border-radius:0 8px 8px 0;color:#1e40af;font-size:13px;margin:12px 0}
    .note.warn{background:#fffbeb;border-color:#f59e0b;color:#92400e}
    .caption{color:var(--muted);font-size:12px;margin:0 0 10px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}
    .chart-card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px}
    .chart-card img{width:100%;display:block}
    .pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:600}
    .pill.ok{background:#d1fae5;color:#065f46}.pill.no{background:#fee2e2;color:#991b1b}
    footer{margin-top:40px;color:var(--muted);font-size:12px;text-align:center}
    """

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transaction Cleaning Report</title><style>{css}</style></head>
<body><div class="wrap">

<header class="hero">
  <h1>Nightly Transaction Log — Cleaning Report</h1>
  <p>Source: <code>{html.escape(input_path)}</code></p>
  <p>Generated: {generated} &nbsp;·&nbsp; pandas {pd.__version__} &nbsp;·&nbsp;
     matplotlib {matplotlib.__version__}</p>
  <p>Clean CSV: <code>{html.escape(output_csv)}</code></p>
</header>

<div class="cards">
  <div class="card"><div class="label">Raw rows</div><div class="value">{total:,}</div></div>
  <div class="card good"><div class="label">Clean rows</div><div class="value">{kept:,}</div></div>
  <div class="card bad"><div class="label">Removed</div><div class="value">{dropped:,}</div></div>
  <div class="card"><div class="label">Retained</div><div class="value">{retention:.2f}%</div></div>
  <div class="card"><div class="label">Total valid spend</div><div class="value">${revenue:,.0f}</div></div>
  <div class="card"><div class="label">Customers</div><div class="value">{customers:,}</div></div>
</div>

<div class="note"><strong>Canceled / bad-data policy.</strong> {html.escape(status_note)}
  Duplicate transactions are removed after validation. {html.escape(outlier_note)}
</div>

<h2><span class="num">01</span> Raw dataset as loaded in pandas</h2>
<p class="caption">Shape: <strong>{preview.shape[0]:,} rows × {preview.shape[1]} columns</strong>.
  First 12 rows shown; the working copy is read as strings so no raw value is silently changed.</p>
<div class="panel"><div class="table-scroll">{df_table(preview, 12)}</div></div>
<p class="caption" style="margin-top:12px">Column types, nulls and cardinality:</p>
<div class="panel"><div class="table-scroll">{df_table(dtype_info)}</div></div>
{'<p class="caption" style="margin-top:12px">Numeric summary:</p><div class="panel"><div class="table-scroll">' + df_table(describe.round(2)) + '</div></div>' if not describe.empty else ''}

<h2><span class="num">02</span> Cleaning log</h2>
<p class="caption">Every rule, applied in order. Counts are cumulative against the surviving rows.</p>
<div class="panel"><div class="table-scroll">{df_table(log_df)}</div></div>

<h2><span class="num">03</span> Rejected records</h2>
<p class="caption">{dropped:,} rows removed. Reason breakdown and up to 100 example rows below.</p>
<div class="panel"><div class="table-scroll">{df_table(reasons)}</div></div>
<div class="panel" style="margin-top:12px"><div class="table-scroll">
  {df_table(removed.head(100)) if not removed.empty else '<p class="caption">Nothing removed.</p>'}
</div></div>

<h2><span class="num">04</span> Clean dataset (output preview)</h2>
<p class="caption">{kept:,} rows × {clean.shape[1]} columns · {date_min} to {date_max} ·
  first 200 rows shown here; the complete dataset is exported to
  <code>{html.escape(os.path.basename(output_csv))}</code>.</p>
<div class="panel"><div class="table-scroll">
  {df_table(clean, 200) if not clean.empty else '<p class="caption">No clean rows.</p>'}
</div></div>

<h2><span class="num">05</span> Visualizations</h2>
<div class="grid">{chart_html}</div>
{validation_html}

<footer>Generated by etl.py · Everything on this page is embedded:
  no external stylesheets, scripts or chart requests.</footer>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Clean transaction logs nightly.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Raw transaction CSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Folder for clean CSV and HTML report")
    parser.add_argument("--drop-outliers", action="store_true",
                        help="Also remove spend above Q3 + 3*IQR (flagged by default)")
    parser.add_argument("--end-date", default=None,
                        help="Observation end YYYY-MM-DD for recency/T (default: latest date)")
    parser.add_argument("--time-unit", choices=["W", "D"], default="W",
                        help="Units for recency and T: weeks (default) or days")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, CLEAN_CSV_NAME)
    html_path = os.path.join(args.output_dir, REPORT_HTML_NAME)

    preview, work = load_raw(args.input)
    clean, removed, log_df, _, stats = clean_data(work, args.drop_outliers)
    clean.to_csv(csv_path, index=False)

    # Validation gate: exact schema/types are checked before the HTML is built.
    # No model is created, imported or fitted here.
    validation = validate_and_export(clean, args.output_dir, args.end_date, args.time_unit)

    charts = build_charts(clean, removed)
    report = build_report(preview, work, clean, removed, log_df, stats,
                          charts, args.input, csv_path, args.drop_outliers, validation)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    # Only file locations are printed; all viewing/tables/charts live in HTML.
    print(f"Clean dataset : {csv_path} ({stats['final_rows']:,} rows)")
    print(f"HTML report   : {html_path}")
    for key in ("model_input", "pnbd_input", "gamma_gamma_input", "validation_report"):
        print(f"{key + ' :':16s} {validation[key]}")


if __name__ == "__main__":
    main()
