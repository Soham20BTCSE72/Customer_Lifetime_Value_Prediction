#!/usr/bin/env python3
"""
Out-of-sample decile calibration for the 90-day CLV forecasts.

Models are fitted ONLY on the pre-holdout window (last 90 days held out), then
customers are sorted into 10 equal-count buckets by predicted CLV. Each bucket's
predicted spend is compared with the actual spend recorded in the holdout.

Outputs
    decile_calibration.csv   one row per decile with predicted vs actual spend
    decile_calibration.html  self-contained chart + table report

Run: python3 decile_calibration.py
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from clv_forecast import (
    BLUE, FUTURE_MONTHS, GREEN, GRID, HORIZON_DAYS, INK, MUTED, RED,
    build_model_input, fit_gg, fit_pnbd, mean_draws, sanitize,
)

TX_PATH = "/home/user/clean_transactions.csv"
CSV_OUT = "/home/user/decile_calibration.csv"
HTML_OUT = "/home/user/decile_calibration.html"


def fig_img(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def run(tx_path: str = TX_PATH) -> tuple[pd.DataFrame, dict]:
    tx = pd.read_csv(tx_path)
    dates = pd.to_datetime(tx["date"])
    cutoff = dates.max() - pd.Timedelta(days=HORIZON_DAYS)
    calibration_tx = tx.loc[dates <= cutoff]
    holdout_tx = tx.loc[dates > cutoff]

    calib = sanitize(build_model_input(calibration_tx, end_date=str(cutoff.date()),
                                       time_unit="W"))
    pnbd, _ = fit_pnbd(calib)
    gg, _ = fit_gg(calib[(calib["frequency"] > 0) & (calib["monetary_value"] > 0)])

    pred_clv = mean_draws(gg.expected_customer_lifetime_value(
        transaction_model=pnbd, data=calib, future_t=FUTURE_MONTHS,
        discount_rate=0.0, time_unit="W"), len(calib))
    pred_clv = np.where(calib["frequency"].to_numpy() == 0, 0.0, pred_clv)

    actual_spend = (holdout_tx.groupby("cust_id")["spend"].sum()
                    .reindex(calib["customer_id"], fill_value=0).to_numpy(float))

    # Rank-based deciles guarantee 10 equally sized buckets even with many zeros.
    ranks = pd.Series(pred_clv).rank(method="first", pct=True).to_numpy()
    decile = np.minimum((ranks * 10).astype(int) + 1, 10)

    cust = pd.DataFrame({
        "customer_id": calib["customer_id"].to_numpy(),
        "decile": decile,
        "predicted_clv": pred_clv,
        "actual_spend_90d": actual_spend,
    })
    total_actual = actual_spend.sum()
    grp = cust.groupby("decile", as_index=False).agg(
        customers=("customer_id", "size"),
        predicted_spend=("predicted_clv", "sum"),
        actual_spend=("actual_spend_90d", "sum"),
        predicted_mean=("predicted_clv", "mean"),
        actual_mean=("actual_spend_90d", "mean"),
    )
    grp["actual_to_predicted_ratio"] = np.where(
        grp["predicted_spend"] > 0,
        grp["actual_spend"] / grp["predicted_spend"].replace(0, np.nan), np.nan)
    grp["lift_vs_overall"] = grp["actual_mean"] / (total_actual / len(cust))
    grp = grp.sort_values("decile", ascending=False).reset_index(drop=True)
    grp["cumulative_actual_share"] = grp["actual_spend"].cumsum() / total_actual
    grp = grp.round(4)

    top = grp.iloc[0]
    info = {
        "cutoff": str(cutoff.date()), "end": str(dates.max().date()),
        "customers": len(cust), "holdout_tx": len(holdout_tx),
        "total_actual": total_actual,
        "total_predicted": pred_clv.sum(),
        "top_share": float(top["cumulative_actual_share"]),
        "top_ratio": float(top["actual_to_predicted_ratio"]),
        "spearman": float(pd.Series(pred_clv).corr(pd.Series(actual_spend),
                                                   method="spearman")),
    }
    return grp, info


def build_html(grp: pd.DataFrame, info: dict) -> str:
    # Predicted vs actual mean spend by decile
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    x = np.arange(len(grp))
    ax.bar(x - 0.19, grp["predicted_mean"], width=0.38, label="predicted CLV",
           color=BLUE, alpha=0.9)
    ax.bar(x + 0.19, grp["actual_mean"], width=0.38, label="actual 90-day spend",
           color=GREEN, alpha=0.9)
    ax.set_xticks(x, [f"D{int(d)}" for d in grp["decile"]])
    ax.set_xlabel("Predicted CLV decile (D10 = highest)", fontsize=9, color=MUTED)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Predicted vs actual spend by CLV decile", loc="left",
                 fontsize=12, fontweight="bold", color=INK)
    style(ax)
    chart1 = fig_img(fig)

    # Cumulative actual spend captured by top deciles
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.plot(range(1, 11), grp["cumulative_actual_share"] * 100,
            marker="o", color=BLUE, linewidth=2.2)
    ax.axhline(100, color=RED, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Top N predicted-CLV deciles", fontsize=9, color=MUTED)
    ax.set_ylabel("% of actual spend captured", fontsize=9, color=MUTED)
    ax.set_ylim(0, 105)
    ax.set_title("Cumulative actual spend captured (ranking quality)", loc="left",
                 fontsize=12, fontweight="bold", color=INK)
    style(ax)
    chart2 = fig_img(fig)

    display = grp.rename(columns={
        "decile": "Decile", "customers": "Customers",
        "predicted_spend": "Predicted spend", "actual_spend": "Actual spend",
        "predicted_mean": "Predicted mean", "actual_mean": "Actual mean",
        "actual_to_predicted_ratio": "Actual / predicted",
        "lift_vs_overall": "Lift", "cumulative_actual_share": "Cumulative actual share"})
    table_html = display.to_html(index=False, border=0, classes="tbl", justify="left")
    css = """
    :root{--line:#e2e8f0}*{box-sizing:border-box}body{margin:0;background:#f8fafc;
      color:#0f172a;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
    .wrap{max-width:1120px;margin:0 auto;padding:28px 22px 70px}
    header{background:linear-gradient(135deg,#1e3a8a,#2563eb,#0ea5e9);color:#fff;
      border-radius:16px;padding:28px 32px}header h1{margin:0 0 6px;font-size:26px}
    header p{margin:3px 0;color:#dbeafe;font-size:13px}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}
    .card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}
    .card .l{font-size:11px;text-transform:uppercase;color:#64748b}
    .card .v{font-size:22px;font-weight:700;margin-top:4px;color:#2563eb}
    h2{font-size:19px;margin:32px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
    .panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
    .panel img{width:100%;display:block}
    .scroll{overflow:auto}.tbl{border-collapse:collapse;width:100%;font-size:12px}
    .tbl th,.tbl td{padding:8px 10px;border-bottom:1px solid #f1f5f9;text-align:left}
    .tbl th{background:#f1f5f9;position:sticky;top:0}
    .note{background:#eff6ff;border-left:4px solid #2563eb;padding:11px 14px;
      color:#1e40af;border-radius:0 8px 8px 0;font-size:13px}
    footer{margin-top:40px;text-align:center;color:#64748b;font-size:12px}
    """
    cards = [
        ("Customers scored", f"{info['customers']:,}"),
        ("Top decile captures", f"{info['top_share']*100:.1f}% of actual spend"),
        ("Top decile actual/pred", f"{info['top_ratio']:.2f}×"),
        ("Spearman rank corr", f"{info['spearman']:.3f}"),
    ]
    card_html = "".join(f'<div class="card"><div class="l">{l}</div>'
                        f'<div class="v">{v}</div></div>' for l, v in cards)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decile Calibration — 90-day CLV</title><style>{css}</style></head>
<body><div class="wrap">
<header><h1>Decile Calibration — 90-day CLV</h1>
<p>Out-of-sample: fitted before {info['cutoff']}, actual spend measured to {info['end']}</p>
<p>{info['holdout_tx']:,} holdout transactions · {info['customers']:,} customers · generated {datetime.now().strftime('%d %b %Y, %H:%M:%S')}</p></header>
<div class="cards">{card_html}</div>
<div class="note"><strong>Reading the table.</strong> D10 holds the 10% of customers with the highest predicted CLV.
Actual spend is the total recorded in the 90-day holdout. A ratio near 1.0 means good calibration;
a falling ratio from D1 to D10 would mean the model over-ranks top spenders.</div>
<h2>1. Predicted vs actual by decile</h2>
<div class="panel"><img src="data:image/png;base64,{chart1}" alt="decile bars"></div>
<h2>2. Ranking quality — cumulative actual spend</h2>
<div class="panel"><img src="data:image/png;base64,{chart2}" alt="cumulative capture"></div>
<h2>3. Decile table</h2>
<div class="panel scroll">{table_html}</div>
<footer>Generated by decile_calibration.py · all charts and tables embedded.</footer>
</div></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Decile calibration of 90-day CLV forecasts.")
    parser.add_argument("--transactions", default=TX_PATH, help="Clean transaction CSV")
    parser.add_argument("--csv-out", default=CSV_OUT, help="Decile CSV output")
    parser.add_argument("--html-out", default=HTML_OUT, help="Decile HTML output")
    args = parser.parse_args()
    grp, info = run(args.transactions)
    grp.to_csv(args.csv_out, index=False)
    with open(args.html_out, "w", encoding="utf-8") as fh:
        fh.write(build_html(grp, info))
    print(f"Decile CSV  : {args.csv_out}")
    print(f"Decile HTML : {args.html_out}")
    print(grp[["decile", "predicted_mean", "actual_mean",
               "actual_to_predicted_ratio", "cumulative_actual_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
