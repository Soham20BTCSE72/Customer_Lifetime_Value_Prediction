#!/usr/bin/env python3
"""
RFM feature engineering for the augmented CLV dataset.

Imports Augmented_model_input.csv, validates the four canonical model columns,
then computes Recency / Frequency / Tenure / Monetary features plus:
  * time-windowed estimates (4 / 8 / 12 / 26 weeks),
  * behaviour ratios and log transforms,
  * quintile R/F/M scores, combined RFM score and segment label.

Outputs:
    rfm_features.csv   full feature table (one row per customer)
    rfm_features.html  self-contained webpage with every table and chart

No model is created or fitted. Time-windowed columns are estimates derived from
the customer-level summary (the augmented file has no transaction timestamps):
purchases are assumed to accrue evenly across the observed tenure, and a
customer with recency > window contributes zero purchases in that window.

Usage
-----
    python3 rfm_features.py
    python3 rfm_features.py --input Augmented_model_input.csv \
        --output-dir /home/user --windows 4 8 12 26
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from validate_model_input import validate_model_input

CANONICAL = ["customer_id", "frequency", "recency", "T", "monetary_value"]
PALETTE = {
    "blue": "#2563eb", "cyan": "#0891b2", "green": "#059669",
    "red": "#dc2626", "ink": "#0f172a", "muted": "#64748b",
    "grid": "#e2e8f0",
}
FEATURE_DICTIONARY = {
    "customer_id": "Unique customer identifier (int64).",
    "recency": "Weeks from first to most recent purchase (R).",
    "frequency": "Number of repeat purchases (F).",
    "tenure": "Weeks from first purchase to observation end (T).",
    "monetary": "Mean value of repeat purchases (M).",
    "recency_ratio": "recency / tenure: how recently the customer was active within their lifetime.",
    "frequency_per_week": "Repeat purchases per week of tenure.",
    "monetary_per_week": "Estimated average spend per week (monetary * frequency / tenure).",
    "avg_interpurchase_weeks": "tenure / frequency when frequency > 0, else 0.",
    "is_repeat_buyer": "1 when frequency > 0, else 0.",
    "log1p_frequency": "log(1 + frequency), tames the long right tail.",
    "log1p_monetary": "log(1 + monetary), tames the long right tail.",
    "active_last_4w": "1 when the last purchase was within 4 weeks.",
    "frequency_last_4w": "Estimated repeat purchases in the last 4 weeks.",
    "monetary_last_4w": "Estimated spend in the last 4 weeks.",
    "active_last_8w": "1 when the last purchase was within 8 weeks.",
    "frequency_last_8w": "Estimated repeat purchases in the last 8 weeks.",
    "monetary_last_8w": "Estimated spend in the last 8 weeks.",
    "active_last_12w": "1 when the last purchase was within 12 weeks.",
    "frequency_last_12w": "Estimated repeat purchases in the last 12 weeks.",
    "monetary_last_12w": "Estimated spend in the last 12 weeks.",
    "active_last_26w": "1 when the last purchase was within 26 weeks.",
    "frequency_last_26w": "Estimated repeat purchases in the last 26 weeks.",
    "monetary_last_26w": "Estimated spend in the last 26 weeks.",
    "r_score": "Recency quintile 5 (most recent) to 1 (oldest).",
    "f_score": "Frequency quintile 1 (fewest) to 5 (most repeat purchases).",
    "m_score": "Monetary quintile 1 (lowest) to 5 (highest).",
    "rfm_score": "r_score + f_score + m_score, range 3-15.",
    "rfm_segment": "Behavioural label derived from R and F scores.",
}


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def quintile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    pct = series.rank(method="average", pct=True)
    score = np.ceil(pct * 5).clip(1, 5).astype(int)
    if not higher_is_better:
        score = 6 - score
    return pd.Series(score, index=series.index)


def segment_label(r: int, f: int) -> str:
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New / Promising"
    if r <= 2 and f >= 4:
        return "At Risk"
    if r <= 2 and f >= 1:
        return "Needs Attention"
    return "Hibernating"


def compute_rfm(df: pd.DataFrame, windows=(4, 8, 12, 26)) -> pd.DataFrame:
    missing = [c for c in CANONICAL if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing column(s): {missing}")

    # Validation gate before feature engineering / model hand-off.
    checks, errors = validate_model_input(df[CANONICAL])
    if errors:
        raise ValueError("Input failed model-schema validation: " + ", ".join(errors))

    out = pd.DataFrame({
        "customer_id": df["customer_id"].astype("int64"),
        "recency": df["recency"].astype("float64"),
        "frequency": df["frequency"].astype("float64"),
        "tenure": df["T"].astype("float64"),
        "monetary": df["monetary_value"].astype("float64"),
    })

    tenure_safe = out["tenure"].replace(0, np.nan)
    out["recency_ratio"] = (out["recency"] / tenure_safe).round(4)
    out["frequency_per_week"] = (out["frequency"] / tenure_safe).round(4)
    out["monetary_per_week"] = (
        out["monetary"] * out["frequency"] / tenure_safe).round(4)
    out["avg_interpurchase_weeks"] = np.where(
        out["frequency"] > 0, tenure_safe / out["frequency"].replace(0, np.nan), 0.0
    ).round(2)
    out["is_repeat_buyer"] = (out["frequency"] > 0).astype(int)
    out["log1p_frequency"] = np.log1p(out["frequency"]).round(4)
    out["log1p_monetary"] = np.log1p(out["monetary"]).round(4)

    for w in windows:
        active = out["recency"] <= w
        coverage = np.minimum(w, out["tenure"]) / tenure_safe
        freq_w = np.where(active, out["frequency"] * coverage, 0.0)
        out[f"active_last_{w}w"] = active.astype(int)
        out[f"frequency_last_{w}w"] = np.round(freq_w, 4)
        out[f"monetary_last_{w}w"] = np.round(freq_w * out["monetary"], 2)

    out["r_score"] = quintile_score(out["recency"], higher_is_better=False)
    out["f_score"] = quintile_score(out["frequency"])
    out["m_score"] = quintile_score(out["monetary"])
    out["rfm_score"] = out[["r_score", "f_score", "m_score"]].sum(axis=1).astype(int)
    out["rfm_segment"] = [
        segment_label(r, f) for r, f in zip(out["r_score"], out["f_score"])
    ]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Charts / HTML
# --------------------------------------------------------------------------- #
def fig_img(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ax_style(ax: plt.Axes, x: bool = False) -> None:
    ax.grid(axis="x" if x else "y", color=PALETTE["grid"], linewidth=0.8)
    ax.grid(axis="y" if x else "x", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)


def build_charts(rfm: pd.DataFrame) -> list[tuple[str, str, str]]:
    charts: list[tuple[str, str, str]] = []
    specs = [
        ("Recency distribution", "Weeks since last purchase; lower is more recent.",
         rfm["recency"], 45, PALETTE["blue"], "weeks"),
        ("Frequency distribution", "Repeat purchases per customer (log-spaced).",
         np.log1p(rfm["frequency"]), 40, PALETTE["cyan"], "log(1 + repeat purchases)"),
        ("Tenure distribution", "Weeks from first purchase to observation end.",
         rfm["tenure"], 45, PALETTE["green"], "weeks"),
        ("Monetary distribution", "Mean repeat-purchase value (log-spaced).",
         np.log1p(rfm["monetary"]), 45, PALETTE["blue"], "log(1 + mean spend)"),
    ]
    for title, caption, values, bins, color, xlabel in specs:
        fig, ax = plt.subplots(figsize=(8.8, 3.1))
        ax.hist(values, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(xlabel, fontsize=9, color=PALETTE["muted"])
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=PALETTE["ink"])
        ax_style(ax)
        charts.append((title, caption, fig_img(fig)))

    # RFM score distribution
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    for name, col, color in [("Recency score", "r_score", PALETTE["blue"]),
                             ("Frequency score", "f_score", PALETTE["cyan"]),
                             ("Monetary score", "m_score", PALETTE["green"])]:
        counts = rfm[col].value_counts().sort_index()
        ax.plot(counts.index, counts.values, marker="o", label=name, color=color, linewidth=2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("R / F / M quintile scores", loc="left", fontsize=12,
                 fontweight="bold", color=PALETTE["ink"])
    ax_style(ax)
    charts.append(("RFM scores", "Quintile scores 1-5 for each RFM dimension.", fig_img(fig)))

    # Segment counts
    seg = rfm["rfm_segment"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    bars = ax.barh(seg.index, seg.values, color=PALETTE["green"], alpha=0.9)
    for bar, v in zip(bars, seg.values):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v:,}", va="center", fontsize=8)
    ax.set_xlim(0, seg.max() * 1.2)
    ax.set_title("Customers by RFM segment", loc="left", fontsize=12,
                 fontweight="bold", color=PALETTE["ink"])
    ax_style(ax, x=True)
    charts.append(("Segments", "Behavioural segments from R and F quintiles.", fig_img(fig)))

    # Correlation heatmap
    cols = ["recency", "frequency", "tenure", "monetary", "log1p_frequency", "log1p_monetary"]
    corr = rfm[cols].corr()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols)), cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr.iloc[i, j]) > 0.55 else PALETTE["ink"])
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlations", loc="left", fontsize=12,
                 fontweight="bold", color=PALETTE["ink"])
    charts.append(("Correlations", "Relationships among the engineered numeric features.", fig_img(fig)))
    return charts


def table(df: pd.DataFrame, rows: int | None = None) -> str:
    shown = df if rows is None else df.head(rows)
    return shown.to_html(escape=True, index=False, classes="data-table", border=0,
                         justify="left")


def build_html(rfm: pd.DataFrame, charts: list[tuple[str, str, str]],
               input_path: str, output_csv: str, windows) -> str:
    css = """
    :root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--blue:#2563eb;--green:#059669}
    *{box-sizing:border-box}
    body{margin:0;background:#f8fafc;color:var(--ink);font:14px/1.55 -apple-system,
         BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
    .wrap{max-width:1200px;margin:0 auto;padding:28px 22px 70px}
    header{background:linear-gradient(135deg,#0f766e,#0891b2 55%,#2563eb);color:white;
           border-radius:16px;padding:28px 32px;box-shadow:0 12px 30px #0f766e26}
    header h1{margin:0 0 6px;font-size:27px} header p{margin:3px 0;color:#cffafe;font-size:13px}
    code{background:#eef2ff;color:#3730a3;padding:1px 5px;border-radius:4px;font-size:12px}
    h2{font-size:19px;margin:38px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
    h2 .num{color:var(--blue);margin-right:8px} h3{font-size:14px;margin:0 0 2px}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
    .card{background:white;border:1px solid var(--line);border-radius:12px;padding:15px 17px}
    .card .label{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
    .card .value{font-size:22px;font-weight:700;margin-top:4px}
    .panel{background:white;border:1px solid var(--line);border-radius:12px;padding:8px 4px}
    .scroll{overflow:auto;max-height:430px;margin:8px}
    table.data-table{border-collapse:collapse;width:100%;font-size:12px}
    th{position:sticky;top:0;background:#f1f5f9;text-align:left;padding:9px 10px;
       border-bottom:2px solid var(--line);white-space:nowrap}
    td{padding:7px 10px;border-bottom:1px solid #f1f5f9;white-space:nowrap}
    tbody tr:nth-child(even){background:#fcfdff}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}
    .chart{background:white;border:1px solid var(--line);border-radius:12px;padding:16px 18px}
    .chart img{width:100%;display:block} .caption{color:var(--muted);font-size:12px;margin:0 0 10px}
    .note{background:#eff6ff;border-left:4px solid var(--blue);padding:11px 14px;
          border-radius:0 8px 8px 0;color:#1e40af;font-size:13px;margin:12px 0}
    footer{margin-top:40px;color:var(--muted);font-size:12px;text-align:center}
    """
    kpis = [
        ("Customers", f"{len(rfm):,}"),
        ("Repeat buyers", f"{int(rfm['is_repeat_buyer'].sum()):,}"),
        ("Avg recency (wk)", f"{rfm['recency'].mean():.2f}"),
        ("Avg frequency", f"{rfm['frequency'].mean():.2f}"),
        ("Avg tenure (wk)", f"{rfm['tenure'].mean():.2f}"),
        ("Avg monetary", f"${rfm['monetary'].mean():.2f}"),
    ]
    cards = "".join(f'<div class="card"><div class="label">{l}</div>'
                    f'<div class="value">{v}</div></div>' for l, v in kpis)
    charts_html = "\n".join(
        f'<article class="chart"><h3>{html.escape(t)}</h3>'
        f'<p class="caption">{html.escape(c)}</p>'
        f'<img src="data:image/png;base64,{img}" alt="{html.escape(t)} chart"></article>'
        for t, c, img in charts)
    dictionary = pd.DataFrame(
        [(k, v) for k, v in FEATURE_DICTIONARY.items() if k in rfm.columns],
        columns=["feature", "definition"],
    )
    stats = rfm.select_dtypes(include=[np.number]).describe().T.reset_index().rename(
        columns={"index": "feature"})
    seg_counts = (rfm["rfm_segment"].value_counts().rename_axis("rfm_segment")
                  .reset_index(name="customers"))
    generated = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RFM Features Dashboard</title><style>{css}</style></head>
<body><div class="wrap">
<header><h1>RFM Feature Engineering Dashboard</h1>
<p>Source: <code>{html.escape(input_path)}</code></p>
<p>Generated: {generated} · pandas {pd.__version__} · rows: {len(rfm):,}</p>
<p>Feature CSV: <code>{html.escape(output_csv)}</code></p></header>
<div class="cards">{cards}</div>
<div class="note"><strong>Time-window columns</strong> cover {', '.join(str(w) for w in windows)} weeks.
They are estimates: the augmented input is customer-level, so purchases are assumed
to accrue evenly over tenure, and customers whose last purchase is older than the
window contribute zero activity in it. Canonical R/F/T/M columns are unchanged and
still match the Pareto/NBD + Gamma-Gamma input schema.</div>

<h2><span class="num">01</span> Feature dictionary</h2>
<div class="panel"><div class="scroll">{table(dictionary)}</div></div>

<h2><span class="num">02</span> Numeric feature summary</h2>
<div class="panel"><div class="scroll">{table(stats.round(3))}</div></div>
<p class="caption" style="margin-top:12px">Segment counts:</p>
<div class="panel"><div class="scroll">{table(seg_counts)}</div></div>

<h2><span class="num">03</span> Visualizations</h2>
<div class="grid">{charts_html}</div>

<h2><span class="num">04</span> RFM feature table (first 200 rows)</h2>
<div class="panel"><div class="scroll">{table(rfm, 200)}</div></div>
<footer>Generated by rfm_features.py · tables and charts are embedded; no external assets.</footer>
</div></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RFM features before model input.")
    parser.add_argument("--input", default="/home/user/Augmented_model_input.csv")
    parser.add_argument("--output-dir", default="/home/user")
    parser.add_argument("--windows", type=int, nargs="+", default=[4, 8, 12, 26],
                        help="Time windows in weeks (default: 4 8 12 26)")
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    rfm = compute_rfm(source, tuple(args.windows))
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "rfm_features.csv")
    html_path = os.path.join(args.output_dir, "rfm_features.html")
    rfm.to_csv(csv_path, index=False)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(build_html(rfm, build_charts(rfm), args.input, csv_path, args.windows))
    print(f"RFM features CSV : {csv_path} ({len(rfm):,} rows, {rfm.shape[1]} columns)")
    print(f"HTML webpage     : {html_path}")


if __name__ == "__main__":
    main()
