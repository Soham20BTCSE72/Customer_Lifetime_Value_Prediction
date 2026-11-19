#!/usr/bin/env python3
"""
Deployment app with a 2-90 day forecast slider.

Search/select a customer ID, then move the horizon slider: purchase count,
spend-per-purchase, total forecast spend and alive probability are recomputed
live by the saved Pareto/NBD and Gamma-Gamma models.

Unnecessary duplicate sections and the redundant "Load records" button were
removed; only RFM features and raw transaction records remain below the cards.

Run: python3 deploy_app.py   (binds 0.0.0.0:5000)
"""

from __future__ import annotations

import json
import os
import warnings

import cloudpickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE, name))


print("Loading models and customer data ...", flush=True)
with open(os.path.join(BASE, "pareto_nbd_model.pkl"), "rb") as fh:
    pnbd = cloudpickle.load(fh)
with open(os.path.join(BASE, "gamma_gamma_model.pkl"), "rb") as fh:
    gg = cloudpickle.load(fh)

AUGMENTED = load("Augmented_model_input.csv")
RFM = load("rfm_features.csv")
TX = load("clean_transactions.csv")
DISCOUNT = load("Discount_offered_Custmor.csv")
AUG_INDEX = AUGMENTED.set_index("customer_id")
IDS = sorted(int(x) for x in AUGMENTED["customer_id"].unique())
ID_STRINGS = [str(x) for x in IDS]
ID_SET = set(IDS)
OFFER_MEDIAN = float(DISCOUNT["predicted_clv_90d_undiscounted"].median())
print(f"Ready: {len(IDS):,} customers.", flush=True)


def mean1(da) -> float:
    for dim in ("draw", "chain"):
        if dim in da.dims:
            da = da.mean(dim=dim)
    return float(np.asarray(da).reshape(-1)[0])


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def customer_row(customer_id: int) -> pd.DataFrame:
    row = AUG_INDEX.loc[[customer_id]].reset_index().copy()
    row = row[["customer_id", "frequency", "recency", "T", "monetary_value"]]
    if float(row.iloc[0]["frequency"]) == 0:
        row.loc[row.index[0], "recency"] = 0.0
    return row


def forecast(customer_id: int, days: int) -> dict:
    days = int(np.clip(days, 2, 90))
    row = customer_row(customer_id)
    freq = float(row.iloc[0]["frequency"])

    purchases = mean1(pnbd.expected_purchases(data=row, future_t=days / 7.0))
    alive_now = mean1(pnbd.expected_probability_alive(data=row, future_t=0))
    alive_future = mean1(pnbd.expected_probability_alive(data=row, future_t=days / 7.0))
    spend_each = mean1(gg.expected_customer_spend(data=row)) if freq > 0 else 0.0
    total_spend = purchases * spend_each
    eligible = bool(alive_now >= 0.70)

    return {
        "ok": True, "customer_id": customer_id, "days": days,
        "expected_purchases": round(purchases, 4),
        "probability_alive": round(alive_now, 4),
        "probability_alive_at_horizon": round(alive_future, 4),
        "expected_spend_per_purchase": round(spend_each, 4),
        "predicted_total_spend": round(total_spend, 4),
        "discount_eligible": eligible,
        "discount_offer": ("15% OFF" if eligible and total_spend >= OFFER_MEDIAN
                           else "10% OFF" if eligible else "Not eligible"),
    }


@app.route("/")
def home():
    return send_from_directory(BASE, "model_deployment.html")


@app.route("/api/search")
def search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"ok": True, "ids": IDS[:50]})
    exact = [x for x in ID_STRINGS if x == q]
    starts = [x for x in ID_STRINGS if x.startswith(q) and x != q]
    contains = [x for x in ID_STRINGS if q in x and not x.startswith(q) and x != q]
    return jsonify({"ok": True, "ids": [int(x) for x in (exact + starts + contains)[:50]]})


@app.route("/api/customer/<int:customer_id>")
def customer(customer_id: int):
    if customer_id not in ID_SET:
        return jsonify({"ok": False, "error": f"Customer {customer_id} not found"}), 404
    rfm_row = RFM[RFM["customer_id"] == customer_id]
    tx_rows = TX[TX["cust_id"] == customer_id].copy()
    sections = [
        {"title": "RFM features", "rows": records(rfm_row)},
        {"title": f"Transaction records ({len(tx_rows)})",
         "rows": records(tx_rows) if len(tx_rows)
         else [{"status": "No historical transactions — synthetic customer"}]},
    ]
    return jsonify({"ok": True, "customer_id": customer_id, "sections": sections})


@app.route("/api/forecast/<int:customer_id>")
def api_forecast(customer_id: int):
    if customer_id not in ID_SET:
        return jsonify({"ok": False, "error": f"Customer {customer_id} not found"}), 404
    try:
        days = int(request.args.get("days", 90))
    except ValueError:
        days = 90
    return jsonify(forecast(customer_id, days))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
