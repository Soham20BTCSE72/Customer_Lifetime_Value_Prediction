#!/usr/bin/env python3
"""
Validation stage for CLV model input. NO model is created or fitted here.

It turns cleaned transactions into the exact per-customer schema used by
pymc-marketing's Pareto/NBD and Gamma-Gamma models, validates columns, dtypes,
nulls, uniqueness and value invariants, and writes CSV files:

    model_input.csv         customer_id, frequency, recency, T, monetary_value
    pnbd_input.csv          customer_id, frequency, recency, T
    gamma_gamma_input.csv   customer_id, frequency, monetary_value (repeat buyers)
    validation_report.csv   one row per validation check (PASS / WARN / FAIL)

If any check FAILs, validation_report.csv is written but the model-input CSVs
are NOT written, so an invalid frame can never reach model execution.

CLI
---
    python3 validate_model_input.py \
        --input /home/user/clean_transactions.csv --output-dir /home/user
    python3 validate_model_input.py --input clean.csv --end-date 1998-06-30

Schema facts (verified against pymc-marketing docs):
  * frequency = number of REPEAT purchases (= total purchases - 1)
  * recency   = time between first and last purchase (0 for one-time buyers)
  * T         = time between first purchase and end of observation; T >= recency
  * monetary_value = mean value of REPEAT purchases (0 for one-time buyers)
  * customer_id int64; frequency/recency/T/monetary_value float64
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

MODEL_INPUT_COLUMNS = ["customer_id", "frequency", "recency", "T", "monetary_value"]
PNBD_COLUMNS = ["customer_id", "frequency", "recency", "T"]
GAMMA_GAMMA_COLUMNS = ["customer_id", "frequency", "monetary_value"]
EXPECTED_DTYPES = {
    "customer_id": "int64",
    "frequency": "float64",
    "recency": "float64",
    "T": "float64",
    "monetary_value": "float64",
}


def build_model_input(
    clean: pd.DataFrame, end_date: str | None = None, time_unit: str = "W"
) -> pd.DataFrame:
    """Aggregate cleaned transactions into one validated-ready row per customer.

    time_unit="W" (default) matches pymc-marketing's CLV examples, which model
    recency and T in weeks. Use "D" for days.
    """
    required = {"cust_id", "date", "spend"}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"Cleaned data is missing column(s): {sorted(missing)}")

    tx = clean[["cust_id", "date", "spend"]].copy()
    tx["cust_id"] = pd.to_numeric(tx["cust_id"], errors="raise").astype("int64")
    tx["date"] = pd.to_datetime(tx["date"], errors="raise")
    tx["spend"] = pd.to_numeric(tx["spend"], errors="raise")
    if tx.empty:
        raise ValueError("Cleaned dataset has no rows; cannot build model input.")

    snapshot = (pd.to_datetime(end_date) if end_date else tx["date"].max())
    tx = tx.sort_values(["cust_id", "date"], kind="mergesort")
    tx["repeat_index"] = tx.groupby("cust_id").cumcount()

    grouped = tx.groupby("cust_id", sort=True)
    summary = pd.DataFrame({
        "first_date": grouped["date"].min(),
        "last_date": grouped["date"].max(),
        "total_purchases": grouped["spend"].size(),
    })
    repeat_mean = tx.loc[tx["repeat_index"] > 0].groupby("cust_id")["spend"].mean()

    divisor = 7.0 if time_unit.upper() == "W" else 1.0
    out = pd.DataFrame({
        "customer_id": summary.index.astype("int64"),
        "frequency": (summary["total_purchases"] - 1).astype("int64").astype("float64"),
        "recency": (summary["last_date"] - summary["first_date"]).dt.days.astype("float64") / divisor,
        "T": (snapshot - summary["first_date"]).dt.days.astype("float64") / divisor,
        "monetary_value": repeat_mean.reindex(summary.index).fillna(0.0).astype("float64"),
    })
    return out.reset_index(drop=True)


def validate_model_input(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (checks table, blocking errors). Never raises on bad data."""
    checks: list[dict] = []

    def check(name: str, scope: str, passed: bool, detail: str,
              blocking: bool = True, action: str = "Fix input before export") -> None:
        checks.append({
            "check": name,
            "scope": scope,
            "status": "PASS" if passed else ("FAIL" if blocking else "WARN"),
            "detail": detail,
            "action": "None" if passed else action,
        })

    columns_ok = list(df.columns) == MODEL_INPUT_COLUMNS
    check("exact_columns_and_order", "schema", columns_ok,
          f"expected {MODEL_INPUT_COLUMNS}, got {list(df.columns)}")
    check("minimum_row_count", "schema", len(df) >= 1,
          f"{len(df):,} customer rows")

    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            check(f"dtype:{col}", "dtype", False, "column missing")
            continue
        actual = str(df[col].dtype)
        check(f"dtype:{col}", "dtype", actual == expected,
              f"expected {expected}, got {actual}")

    check("no_nulls", "values", not df.isna().any().any(),
          f"null cells: {int(df.isna().sum().sum())}")
    numeric = df.select_dtypes(include=[np.number])
    check("all_finite", "values", bool(np.isfinite(numeric.to_numpy()).all()),
          "no NaN, +inf or -inf in numeric columns")

    if "customer_id" in df:
        check("customer_id_unique", "identity", df["customer_id"].is_unique,
              f"duplicates: {int(df['customer_id'].duplicated().sum())}")

    if "frequency" in df:
        f = df["frequency"]
        check("frequency_non_negative", "Pareto/NBD",
              bool((f >= 0).all()), f"min={f.min():.2f}")
        check("frequency_integer_valued", "Pareto/NBD",
              bool(np.isclose(f, np.round(f)).all()),
              "repeat counts must be whole numbers stored as float64")

    if "recency" in df:
        r = df["recency"]
        check("recency_non_negative", "Pareto/NBD",
              bool((r >= 0).all()), f"min={r.min():.2f}")

    if {"T", "recency"}.issubset(df.columns):
        ok = bool((df["T"] > 0).all() and (df["T"] >= df["recency"]).all())
        check("T_positive_and_ge_recency", "Pareto/NBD", ok,
              f"min T={df['T'].min():.2f}, max recency={df['recency'].max():.2f}")

    if "monetary_value" in df:
        m = df["monetary_value"]
        check("monetary_value_non_negative", "Gamma-Gamma",
              bool((m >= 0).all()), f"min={m.min():.2f}, max={m.max():.2f}")

    if {"frequency", "monetary_value"}.issubset(df.columns):
        repeat_buyers = df[(df["frequency"] > 0) & (df["monetary_value"] > 0)]
        check("gamma_gamma_eligible_rows", "Gamma-Gamma", len(repeat_buyers) > 0,
              f"{len(repeat_buyers):,} customers with repeat purchases and spend > 0")
        if len(repeat_buyers) > 1:
            corr = repeat_buyers[["frequency", "monetary_value"]].corr().iloc[0, 1]
            check("gamma_gamma_uncorrelated_assumption", "Gamma-Gamma",
                  abs(corr) < 0.3,
                  f"Pearson r={corr:.4f} between frequency and monetary_value "
                  "(model assumes |r| < 0.3)",
                  blocking=False,
                  action="Model assumption may be violated; review before fitting")

    checks_df = pd.DataFrame(checks, columns=["check", "scope", "status", "detail", "action"])
    errors = checks_df.loc[checks_df["status"] == "FAIL", "check"].tolist()
    return checks_df, errors


def validate_and_export(
    clean: pd.DataFrame,
    output_dir: str,
    end_date: str | None = None,
    time_unit: str = "W",
) -> dict:
    """Build, validate and export. Raises ValueError on any blocking failure."""
    os.makedirs(output_dir, exist_ok=True)
    model_input = build_model_input(clean, end_date, time_unit)
    checks, errors = validate_model_input(model_input)

    report_path = os.path.join(output_dir, "validation_report.csv")
    checks.to_csv(report_path, index=False)

    paths: dict = {"validation_report": report_path, "checks": checks, "errors": errors}
    if errors:
        raise ValueError(
            "Validation FAILED; model-input CSVs were not written. "
            "Failed checks: " + ", ".join(errors) + f". See {report_path}"
        )

    main_path = os.path.join(output_dir, "model_input.csv")
    pnbd_path = os.path.join(output_dir, "pnbd_input.csv")
    gg_path = os.path.join(output_dir, "gamma_gamma_input.csv")
    model_input.to_csv(main_path, index=False)
    model_input[PNBD_COLUMNS].to_csv(pnbd_path, index=False)
    model_input.loc[
        (model_input["frequency"] > 0) & (model_input["monetary_value"] > 0),
        GAMMA_GAMMA_COLUMNS,
    ].reset_index(drop=True).to_csv(gg_path, index=False)

    paths.update({
        "model_input": main_path,
        "pnbd_input": pnbd_path,
        "gamma_gamma_input": gg_path,
        "model_input_df": model_input,
        "end_date": str(end_date) if end_date else "max date in data",
        "time_unit": "weeks" if time_unit.upper() == "W" else "days",
    })
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate cleaned transactions for Pareto/NBD and Gamma-Gamma.")
    parser.add_argument("--input", default="/home/user/clean_transactions.csv",
                        help="Cleaned transaction CSV")
    parser.add_argument("--output-dir", default="/home/user", help="Folder for CSV output")
    parser.add_argument("--end-date", default=None,
                        help="Observation end YYYY-MM-DD (default: latest transaction date)")
    parser.add_argument("--time-unit", choices=["W", "D"], default="W",
                        help="Units for recency and T: weeks (default) or days")
    args = parser.parse_args()

    clean = pd.read_csv(args.input)
    result = validate_and_export(clean, args.output_dir, args.end_date, args.time_unit)
    for key in ("validation_report", "model_input", "pnbd_input", "gamma_gamma_input"):
        print(f"{key:20s}: {result[key]}")


if __name__ == "__main__":
    main()
