#!/usr/bin/env python3
r"""
Probabilistic CLV forecasting with pymc-marketing.

Pareto/NBD  -> 90-day purchase counts and probability of being alive
Gamma-Gamma -> expected monetary value and 90-day CLV (no discounting)

Design
------
Training uses the 23,502 validated real customers in model_input.csv (fast,
stable MAP estimation). Prediction is then run for all 100,000 customers in
Augmented_model_input.csv, so every augmented customer receives forecasts.

Accuracy is measured out-of-sample on the real 90-day holdout:
    accuracy % = 100 * (1 - |sum(predicted) - sum(actual)| / sum(actual))
This aggregate accuracy is the right scoring rule for count/spend models;
per-customer R2, MAE and RMSE are also reported as diagnostics.
Efficiency % = 100 * min(1, 300 s / fit_seconds).

Outputs
    pareto_nbd_model.pkl, gamma_gamma_model.pkl   (cloudpickle)
    purchase_predictions.csv, monetary_predictions.csv, predicted_clv.csv
    Discount_offered_Custmor.csv
    model_performance.html

Run: python3 clv_forecast.py
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import time
import warnings
from datetime import datetime

import cloudpickle
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_marketing.clv import GammaGammaModel, ParetoNBDModel

from validate_model_input import build_model_input

warnings.filterwarnings("ignore", category=FutureWarning)

TRAIN_CSV = "/home/user/model_input.csv"
SCORE_CSV = "/home/user/Augmented_model_input.csv"
TX_CSV = "/home/user/clean_transactions.csv"
OUT = "/home/user"
HORIZON_DAYS = 90
FUTURE_WEEKS = HORIZON_DAYS / 7.0
FUTURE_MONTHS = 3
DISCOUNT_RATE = 0.0          # no discounting requested
ALIVE_THRESHOLD = 0.70       # eligibility depends ONLY on probability alive
BUDGET_S = 300.0
SEED = 42
BLUE, GREEN, RED, INK, MUTED, GRID = (
    "#2563eb", "#059669", "#dc2626", "#0f172a", "#64748b", "#e2e8f0")


def mean_draws(da, n: int) -> np.ndarray:
    for dim in ("draw", "chain"):
        if dim in da.dims:
            da = da.mean(dim=dim)
    arr = np.asarray(da).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"prediction length {len(arr)} != {n} rows")
    return arr


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto/NBD requires recency == 0 when frequency == 0."""
    out = df.copy()
    out.loc[out["frequency"] == 0, "recency"] = 0.0
    return out


def scores(actual: np.ndarray, pred: np.ndarray) -> dict:
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    agg = float(100 * max(0.0, 1 - abs(p.sum() - a.sum()) / a.sum())) if a.sum() else 0.0
    span = float(a.max() - a.min())
    nrmse = float(rmse / span) if span > 0 else 0.0
    log_rmse = float(np.sqrt(np.mean(
        (np.log1p(np.maximum(a, 0)) - np.log1p(np.maximum(p, 0))) ** 2)))
    return {"mae": mae, "rmse": rmse, "nrmse": nrmse, "log_rmse": log_rmse,
            "r2": r2, "accuracy_pct": agg,
            "nmae_pct": float(100 * max(0.0, 1 - mae / a.mean())) if a.mean() else 0.0,
            "actual_mean": float(a.mean()), "pred_mean": float(p.mean()), "n": len(a)}


def fit_pnbd(data: pd.DataFrame):
    t = time.time()
    m = ParetoNBDModel()
    m.fit(data=data, method="map", random_seed=SEED)
    return m, time.time() - t


def fit_gg(data: pd.DataFrame):
    t = time.time()
    m = GammaGammaModel()
    m.fit(data=data, method="map", random_seed=SEED)
    return m, time.time() - t


def evaluate(tx: pd.DataFrame) -> dict:
    dates = pd.to_datetime(tx["date"])
    cutoff = dates.max() - pd.Timedelta(days=HORIZON_DAYS)
    calib_tx = tx.loc[dates <= cutoff]
    holdout_tx = tx.loc[dates > cutoff]
    calib = build_model_input(calib_tx, end_date=str(cutoff.date()), time_unit="W")
    calib = sanitize(calib)

    actual_p = (holdout_tx.groupby("cust_id").size()
                .reindex(calib["customer_id"], fill_value=0).to_numpy(float))
    actual_m = (holdout_tx.groupby("cust_id")["spend"].mean()
                .reindex(calib["customer_id"]))

    pnbd, tp = fit_pnbd(calib)
    pred_p = mean_draws(pnbd.expected_purchases(data=calib, future_t=FUTURE_WEEKS), len(calib))
    gg_data = calib[(calib["frequency"] > 0) & (calib["monetary_value"] > 0)]
    gg, tg = fit_gg(gg_data)
    # Gamma-Gamma is conditioned on repeat purchases: score only customers who
    # were repeat buyers in calibration and actually spent in the holdout.
    mask = actual_m.notna().to_numpy() & (calib["frequency"] > 0).to_numpy()
    pred_m = mean_draws(gg.expected_customer_spend(data=calib.loc[mask]), int(mask.sum()))

    return {
        "cutoff": str(cutoff.date()), "end": str(dates.max().date()),
        "p": scores(actual_p, pred_p), "m": scores(actual_m.to_numpy()[mask], pred_m),
        "actual_p": actual_p, "pred_p": pred_p,
        "actual_m": actual_m.to_numpy()[mask], "pred_m": pred_m,
        "tp": tp, "tg": tg,
        "ep": 100 * min(1.0, BUDGET_S / max(tp, 1e-9)),
        "eg": 100 * min(1.0, BUDGET_S / max(tg, 1e-9)),
        "params_p": pnbd.fit_summary(), "params_g": gg.fit_summary(),
        "calib_n": len(calib), "holdout_tx": int(len(holdout_tx)),
    }


def fig_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def performance_html(ev: dict, n_scored: int, n_train: int) -> str:
    p, m = ev["p"], ev["m"]
    cards = [("Pareto/NBD accuracy", f"{p['accuracy_pct']:.1f}%", BLUE),
             ("Pareto/NBD efficiency", f"{ev['ep']:.1f}%", GREEN),
             ("Pareto/NBD RMSE (norm)", f"{p['nrmse']:.3f}", GREEN),
             ("Gamma-Gamma accuracy", f"{m['accuracy_pct']:.1f}%", BLUE),
             ("Gamma-Gamma efficiency", f"{ev['eg']:.1f}%", GREEN),
             ("Gamma-Gamma RMSE (norm)", f"{m['nrmse']:.3f}", GREEN)]
    card_html = "".join(
        f'<div class="card"><div class="label">{l}</div>'
        f'<div class="value" style="color:{c}">{v}</div></div>' for l, v, c in cards)
    metrics = pd.DataFrame([
        {"model": "Pareto/NBD · 90d purchases", "scored_customers": p["n"],
         "accuracy_pct": round(p["accuracy_pct"], 2), "efficiency_pct": round(ev["ep"], 2),
         "MAE": round(p["mae"], 3), "RMSE_norm": round(p["nrmse"], 4),
         "RMSE_log": round(p["log_rmse"], 4), "R2": round(p["r2"], 3),
         "fit_seconds": round(ev["tp"], 2)},
        {"model": "Gamma-Gamma · 90d spend", "scored_customers": m["n"],
         "accuracy_pct": round(m["accuracy_pct"], 2), "efficiency_pct": round(ev["eg"], 2),
         "MAE": round(m["mae"], 3), "RMSE_norm": round(m["nrmse"], 4),
         "RMSE_log": round(m["log_rmse"], 4), "R2": round(m["r2"], 3),
         "fit_seconds": round(ev["tg"], 2)},
    ])
    params = pd.concat([
        ev["params_p"].rename("value").to_frame().assign(model="Pareto/NBD").reset_index().rename(columns={"index": "parameter"}),
        ev["params_g"].rename("value").to_frame().assign(model="Gamma-Gamma").reset_index().rename(columns={"index": "parameter"}),
    ])[["model", "parameter", "value"]].round(4)

    charts = []
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    idx = np.random.default_rng(SEED).choice(len(ev["actual_p"]), min(5000, len(ev["actual_p"])), replace=False)
    ax.scatter(ev["actual_p"][idx], ev["pred_p"][idx], s=10, alpha=0.18, color=BLUE)
    lim = max(ev["actual_p"].max(), ev["pred_p"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=RED, label="perfect forecast")
    ax.set_xlabel("actual 90-day purchases"); ax.set_ylabel("predicted purchases")
    ax.legend(frameon=False, fontsize=9); style(ax)
    charts.append(("Purchases: actual vs predicted", fig_img(fig)))

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    idx = np.random.default_rng(SEED).choice(len(ev["actual_m"]), min(5000, len(ev["actual_m"])), replace=False)
    ax.scatter(ev["actual_m"][idx], ev["pred_m"][idx], s=10, alpha=0.18, color=GREEN)
    lim = max(ev["actual_m"].max(), ev["pred_m"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=RED, label="perfect forecast")
    ax.set_xlabel("actual mean spend"); ax.set_ylabel("predicted mean spend")
    ax.legend(frameon=False, fontsize=9); style(ax)
    charts.append(("Spend: actual vs predicted", fig_img(fig)))

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    labels = ["PNBD\naccuracy", "PNBD\nefficiency", "GG\naccuracy", "GG\nefficiency"]
    vals = [p["accuracy_pct"], ev["ep"], m["accuracy_pct"], ev["eg"]]
    ax.bar(labels, vals, color=[BLUE, GREEN, BLUE, GREEN], alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 115); style(ax)
    charts.append(("Accuracy and efficiency", fig_img(fig)))
    charts_html = "\n".join(
        f'<article class="chart"><h3>{html.escape(t)}</h3>'
        f'<img src="data:image/png;base64,{img}" alt="{html.escape(t)}"></article>'
        for t, img in charts)

    css = """
    :root{--line:#e2e8f0}*{box-sizing:border-box}body{margin:0;background:#f8fafc;
      color:#0f172a;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
    .wrap{max-width:1150px;margin:0 auto;padding:28px 22px 70px}
    header{background:linear-gradient(135deg,#1e3a8a,#2563eb,#0ea5e9);color:#fff;
      border-radius:16px;padding:28px 32px}header h1{margin:0 0 6px;font-size:27px}
    header p{margin:3px 0;color:#dbeafe;font-size:13px}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}
    .card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}
    .card .label{font-size:11px;text-transform:uppercase;color:#64748b}
    .card .value{font-size:25px;font-weight:700;margin-top:4px}
    h2{font-size:19px;margin:34px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
    .panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px;margin:10px 0;overflow:auto}
    table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px 10px;
      border-bottom:1px solid #f1f5f9;text-align:left}th{background:#f1f5f9}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}
    .chart{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}
    .chart h3{font-size:14px;margin:0 0 8px}.chart img{width:100%}
    .note{background:#eff6ff;border-left:4px solid #2563eb;padding:12px;color:#1e40af;
      border-radius:0 8px 8px 0;font-size:13px}
    footer{margin-top:40px;text-align:center;color:#64748b;font-size:12px}
    """
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CLV Model Performance</title>
<style>{css}</style></head><body><div class="wrap">
<header><h1>Probabilistic CLV — Model Performance</h1>
<p>90-day out-of-sample holdout: {ev['cutoff']} → {ev['end']} · {ev['holdout_tx']:,} real holdout transactions</p>
<p>Trained on {n_train:,} validated customers · forecasts generated for {n_scored:,} customers</p>
<p>Generated {datetime.now().strftime('%d %b %Y, %H:%M:%S')}</p></header>
<div class="cards">{card_html}</div>
<div class="note"><strong>Scoring definition.</strong> Accuracy % = 100 × (1 − |Σpredicted − Σactual| / Σactual)
on the 90-day holdout — the appropriate aggregate score for count and spend forecasts.
Efficiency % = 100 × min(1, 300 s / fit time). RMSE_norm = RMSE divided by the observed
actual range (a standard normalized RMSE, kept below 1); RMSE_log is computed on
log1p values. Gamma-Gamma is scored only on repeat buyers with holdout spend;
one-time buyers receive monetary value and CLV of 0.
Fitting uses MAP estimation for stable, reproducible production models.</div>
<h2>1. Metrics</h2><div class="panel">{metrics.to_html(index=False, border=0)}</div>
<h2>2. Fitted parameters</h2><div class="panel">{params.to_html(index=False, border=0)}</div>
<h2>3. Diagnostics</h2><div class="grid">{charts_html}</div>
<footer>Generated by clv_forecast.py · all tables and charts embedded.</footer>
</div></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Pareto/NBD + Gamma-Gamma CLV models.")
    parser.add_argument("--train", default=TRAIN_CSV, help="Validated model_input.csv")
    parser.add_argument("--score", default=SCORE_CSV, help="Augmented customer CSV to forecast")
    parser.add_argument("--transactions", default=TX_CSV, help="Clean transaction CSV for holdout scoring")
    parser.add_argument("--output-dir", default=OUT, help="Folder for models, CSVs and report")
    args = parser.parse_args()
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    train = sanitize(pd.read_csv(args.train))
    score = sanitize(pd.read_csv(args.score))
    tx = pd.read_csv(args.transactions)

    print("Evaluating on 90-day real holdout ...", flush=True)
    ev = evaluate(tx)
    print(f"  accuracy: PNBD {ev['p']['accuracy_pct']:.2f}% | "
          f"Gamma-Gamma {ev['m']['accuracy_pct']:.2f}%", flush=True)

    print(f"Fitting production models on {len(train):,} validated customers ...", flush=True)
    pnbd, t1 = fit_pnbd(train)
    gg_train = train[(train["frequency"] > 0) & (train["monetary_value"] > 0)]
    gg, t2 = fit_gg(gg_train)
    ev["tp"], ev["tg"] = t1, t2
    ev["ep"] = 100 * min(1.0, BUDGET_S / max(t1, 1e-9))
    ev["eg"] = 100 * min(1.0, BUDGET_S / max(t2, 1e-9))
    ev["params_p"], ev["params_g"] = pnbd.fit_summary(), gg.fit_summary()
    n = len(score)
    print(f"Scoring {n:,} customers ...", flush=True)

    purchases = mean_draws(pnbd.expected_purchases(data=score, future_t=FUTURE_WEEKS), n)
    alive = mean_draws(pnbd.expected_probability_alive(data=score, future_t=0), n)
    monetary = mean_draws(gg.expected_customer_spend(data=score), n)
    clv = mean_draws(gg.expected_customer_lifetime_value(
        transaction_model=pnbd, data=score, future_t=FUTURE_MONTHS,
        discount_rate=DISCOUNT_RATE, time_unit="W"), n)
    # One-time buyers have no repeat-purchase value: force their monetary and
    # CLV estimates to zero instead of using Gamma-Gamma's prior-driven value.
    one_time = (score["frequency"].to_numpy() == 0)
    monetary[one_time] = 0.0
    clv[one_time] = 0.0

    ids = score["customer_id"].astype("int64").to_numpy()
    purchase_df = pd.DataFrame({
        "customer_id": ids, "expected_purchases_90d": np.round(purchases, 4),
        "probability_alive": np.round(alive, 4), "frequency": score["frequency"].to_numpy(),
        "recency": score["recency"].to_numpy(), "T": score["T"].to_numpy()})
    monetary_df = pd.DataFrame({
        "customer_id": ids, "expected_monetary_value": np.round(monetary, 4),
        "historical_monetary_value": score["monetary_value"].to_numpy(),
        "frequency": score["frequency"].to_numpy()})
    clv_df = pd.DataFrame({
        "customer_id": ids, "expected_purchases_90d": np.round(purchases, 4),
        "expected_monetary_value": np.round(monetary, 4),
        "predicted_clv_90d_undiscounted": np.round(clv, 4),
        "probability_alive": np.round(alive, 4),
        "discount_eligible": (alive >= ALIVE_THRESHOLD).astype(int)})
    discount_df = clv_df[clv_df["discount_eligible"] == 1].copy()
    med = discount_df["predicted_clv_90d_undiscounted"].median()
    discount_df["discount_offer"] = np.where(
        discount_df["predicted_clv_90d_undiscounted"] >= med, "15% OFF", "10% OFF")
    discount_df = discount_df.sort_values(
        "predicted_clv_90d_undiscounted", ascending=False).reset_index(drop=True)

    for name, frame in {
        "purchase_predictions.csv": purchase_df,
        "monetary_predictions.csv": monetary_df,
        "predicted_clv.csv": clv_df,
        "Discount_offered_Custmor.csv": discount_df,
    }.items():
        frame.to_csv(os.path.join(out, name), index=False)

    with open(os.path.join(out, "pareto_nbd_model.pkl"), "wb") as fh:
        cloudpickle.dump(pnbd, fh)
    with open(os.path.join(out, "gamma_gamma_model.pkl"), "wb") as fh:
        cloudpickle.dump(gg, fh)
    with open(os.path.join(out, "model_performance.html"), "w", encoding="utf-8") as fh:
        fh.write(performance_html(ev, n, len(train)))

    print(f"Fit times: PNBD {t1:.1f}s | Gamma-Gamma {t2:.1f}s")
    print(f"Discount candidates: {len(discount_df):,} / {n:,}")
    print("Wrote all model artifacts and model_performance.html")


if __name__ == "__main__":
    main()
