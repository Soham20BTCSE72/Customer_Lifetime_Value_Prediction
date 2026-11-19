#!/usr/bin/env python3
"""
Augment model_input.csv to 100,000 rows using Faker-driven synthetic sampling.

No model is created or fitted. The augmented rows are sampled from the empirical
distributions of the real validated data, so the CLV assumptions stay intact:

  * customer_id   unique int64, no collision with the original customers
  * frequency     integer-valued float64 >= 0 (repeat purchases)
  * recency       float64, 0 <= recency <= T
  * T             float64 > 0 (weeks, same unit as model_input.csv)
  * monetary_value float64 >= 0; exactly 0 when frequency == 0

The finished CSV is passed through validate_model_input.validate_model_input()
before it is written. If any blocking check fails, nothing is exported.

Usage
-----
    python3 augment_model_input.py
    python3 augment_model_input.py --input model_input.csv \
        --output Augmented_model_input.csv --target-rows 100000 --seed 42

Faker is used for sampling and ID shuffling. If Faker is not installed, the
script falls back to Python's standard random.Random with the same seed.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from validate_model_input import MODEL_INPUT_COLUMNS, validate_model_input

try:
    from faker import Faker
except ImportError:  # "or other tools" fallback
    Faker = None

DEFAULT_INPUT = "/home/user/model_input.csv"
DEFAULT_OUTPUT = "/home/user/Augmented_model_input.csv"
DEFAULT_TARGET = 100_000


def get_rng(seed: int):
    """Return (sampler, label). Sampler mimics random.Random API."""
    if Faker is not None:
        fake = Faker()
        Faker.seed(seed)
        fake.seed_instance(seed)
        return fake.random, "Faker"
    import random
    return random.Random(seed), "random.Random (Faker not installed)"


def augment(
    input_path: str,
    output_path: str,
    target_rows: int = DEFAULT_TARGET,
    seed: int = 42,
) -> pd.DataFrame:
    base = pd.read_csv(input_path)
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in base.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s): {missing}")
    base = base[MODEL_INPUT_COLUMNS].astype({
        "customer_id": "int64",
        "frequency": "float64",
        "recency": "float64",
        "T": "float64",
        "monetary_value": "float64",
    })
    n_existing = len(base)
    n_new = target_rows - n_existing
    if n_new <= 0:
        raise ValueError(
            f"Input already has {n_existing:,} rows; target is {target_rows:,}. "
            "Nothing to augment."
        )

    sampler, sampler_label = get_rng(seed)
    print(f"Random source: {sampler_label} | seed: {seed}")

    # Empirical pools drawn from the validated real customers.
    freq_pool = np.rint(base["frequency"].to_numpy()).astype(int)
    t_pool = base["T"].to_numpy(dtype=float)
    ratio_series = (base["recency"] / base["T"].replace(0, np.nan)).dropna()
    ratio_pool = np.clip(ratio_series.to_numpy(dtype=float), 0.0, 1.0)
    money_pool = base.loc[base["monetary_value"] > 0, "monetary_value"].to_numpy(dtype=float)
    if len(ratio_pool) == 0 or len(money_pool) == 0:
        raise ValueError("Not enough non-zero real values to sample from.")

    # Faker-driven draws from the empirical distributions.
    new_freq = np.array([sampler.choice(freq_pool) for _ in range(n_new)], dtype=float)
    new_t = np.array([sampler.choice(t_pool) for _ in range(n_new)], dtype=float)
    new_ratio = np.array([sampler.choice(ratio_pool) for _ in range(n_new)], dtype=float)

    # Small perturbation keeps synthetic rows from being exact duplicates while
    # staying inside the model's valid domain.
    jitter = np.random.default_rng(seed)
    new_freq = np.clip(np.rint(new_freq + jitter.poisson(0.12, n_new)), 0, None)
    new_t = np.clip(new_t + jitter.normal(0.0, 0.75, n_new), 1.0, None)
    new_recency = np.clip(
        new_t * new_ratio + jitter.normal(0.0, 0.6, n_new), 0.0, new_t
    ).round(2)
    # Pareto/NBD requires recency == 0 for one-time buyers (frequency == 0).
    new_recency = np.where(new_freq == 0, 0.0, new_recency)

    money_draws = np.array([sampler.choice(money_pool) for _ in range(n_new)], dtype=float)
    money_draws = money_draws * np.exp(jitter.normal(0.0, 0.12, n_new))
    new_money = np.where(new_freq > 0, np.round(money_draws, 2), 0.0)

    # Unique IDs: continue after the largest real ID, shuffled by Faker.
    max_id = int(base["customer_id"].max())
    new_ids = np.arange(max_id + 1, max_id + 1 + n_new, dtype=np.int64)
    sampler.shuffle(new_ids)

    synthetic = pd.DataFrame({
        "customer_id": new_ids.astype("int64"),
        "frequency": new_freq.astype("float64"),
        "recency": new_recency.astype("float64"),
        "T": np.round(new_t, 2).astype("float64"),
        "monetary_value": new_money.astype("float64"),
    })

    augmented = pd.concat([base, synthetic], ignore_index=True)[MODEL_INPUT_COLUMNS]
    checks, errors = validate_model_input(augmented)
    warnings = checks.loc[checks["status"] == "WARN", "check"].tolist()
    if warnings:
        print(f"Warnings (non-blocking): {', '.join(warnings)}")
    if errors:
        raise ValueError(
            "Augmented data failed validation; CSV not written. "
            "Failed checks: " + ", ".join(errors)
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    augmented.to_csv(output_path, index=False)
    print(f"Original rows : {n_existing:,}")
    print(f"Synthetic rows: {n_new:,}")
    print(f"Total rows    : {len(augmented):,}")
    print(f"Output        : {output_path}")
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment model_input.csv with Faker.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Validated model_input.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET,
                        help="Final row count including originals (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    args = parser.parse_args()
    augment(args.input, args.output, args.target_rows, args.seed)


if __name__ == "__main__":
    main()
