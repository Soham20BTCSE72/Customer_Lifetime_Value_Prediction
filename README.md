# Probabilistic CLV Forecasting Pipeline

An end-to-end Python project that cleans raw transaction logs, validates and augments customer data, engineers RFM features, fits **Pareto/NBD** and **Gamma-Gamma** models with `pymc-marketing`, forecasts 90-day customer lifetime value, and serves the results through an interactive deployment webpage.

Everything is reproducible from `CDNOW_master.csv` using the scripts in this repository.

---

## What the pipeline does

| Stage | Script | Main output |
|---|---|---|
| 1. Clean raw logs | `etl.py` | `clean_transactions.csv`, `etl.html` |
| 2. Validate model schema | `validate_model_input.py` | `model_input.csv`, `validation_report.csv` |
| 3. Augment to 100,000 customers | `augment_model_input.py` | `Augmented_model_input.csv` |
| 4. RFM feature engineering | `rfm_features.py` | `rfm_features.csv`, `rfm_features.html` |
| 5. Fit models + forecast | `clv_forecast.py` | `*.pkl`, prediction CSVs, `model_performance.html` |
| 6. Decile calibration | `decile_calibration.py` | `decile_calibration.csv`, `decile_calibration.html` |
| 7. Deploy | `deploy_app.py` | Live webpage at `http://127.0.0.1:5000` |

---

## Models and scoring rules

- **Pareto/NBD** (`ParetoNBDModel`) predicts 90-day purchase counts and the probability a customer is still alive.
- **Gamma-Gamma** (`GammaGammaModel`) predicts expected monetary value per purchase and 90-day CLV.
- Fitting uses MAP estimation (`method="map"`) for speed and reproducibility.
- CLV is **not discounted** (`discount_rate=0`).
- Discount eligibility depends **only on alive status**: `probability_alive >= 0.70`.
- One-time buyers (`frequency = 0`) receive monetary value and CLV of `0`.

Measured on a real 90-day out-of-sample holdout:

| Metric | Pareto/NBD | Gamma-Gamma |
|---|---:|---:|
| Aggregate accuracy | 87.62% | 98.57% |
| Normalized RMSE | 0.016 | 0.028 |
| Fit efficiency | 100% | 100% |

Decile check: the top predicted-CLV decile captures **61.3%** of actual holdout spend.

---

## Quick start

### Use the already-trained models

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python deploy_app.py
```

Open <http://127.0.0.1:5000>. Search for a customer ID, then move the slider between 2 and 90 days to recompute predictions live.

### Rebuild everything from raw data

```bash
python etl.py --input CDNOW_master.csv --output-dir .
python validate_model_input.py --input clean_transactions.csv --output-dir .
python augment_model_input.py --input model_input.csv --output Augmented_model_input.csv --target-rows 100000
python rfm_features.py --input Augmented_model_input.csv --output-dir .
python clv_forecast.py --train model_input.csv --score Augmented_model_input.csv \
    --transactions clean_transactions.csv --output-dir .
python decile_calibration.py --transactions clean_transactions.csv \
    --csv-out decile_calibration.csv --html-out decile_calibration.html
python deploy_app.py
```

See `RUN_LOCAL.md` for detailed Windows/macOS/Linux setup and troubleshooting.

---

## Deployment webpage features

- Searchable customer ID dropdown with **Show ID list** and **Search IDs** buttons
- RFM features and raw transaction records for the selected customer
- **2–90 day forecast slider** that updates:
  - expected purchases
  - alive probability now and at the chosen horizon
  - expected spend per purchase
  - predicted total spend
  - discount eligibility and offer
- Discount status shown as `ELIGIBLE FOR DISCOUNT` or `NOT ELIGIBLE FOR DISCOUNTS`

---

## Output files

| File | Description |
|---|---|
| `clean_transactions.csv` | Cleaned raw transaction log |
| `etl.html` | Cleaning report with data previews and charts |
| `model_input.csv` | Validated 5-column model schema |
| `validation_report.csv` | Schema, dtype and invariant checks |
| `Augmented_model_input.csv` | 100,000-customer augmented dataset |
| `rfm_features.csv` / `rfm_features.html` | RFM features and dashboard |
| `pareto_nbd_model.pkl` | Fitted purchase model (cloudpickle) |
| `gamma_gamma_model.pkl` | Fitted monetary model (cloudpickle) |
| `purchase_predictions.csv` | 90-day purchase forecasts |
| `monetary_predictions.csv` | Expected spend per purchase |
| `predicted_clv.csv` | 90-day undiscounted CLV and eligibility |
| `Discount_offered_Custmor.csv` | Alive customers with discount offers |
| `model_performance.html` | Accuracy and efficiency dashboard |
| `decile_calibration.csv` / `.html` | Predicted-vs-actual spend by decile |

---

## Requirements

- Python 3.11+ (tested on 3.13)
- Packages listed in `requirements.txt`, notably `pymc-marketing`, `pymc`, `pandas`, `matplotlib`, `flask`, `faker`, and `cloudpickle`

The `.pkl` files are saved with `cloudpickle`. If you change the PyMC/PyMC-Marketing versions, regenerate them with `clv_forecast.py` to avoid deserialization errors.

---

## Notes

- All HTML reports are self-contained: charts and tables are embedded, with no external CSS/JS/CDN requests.
- `clv_forecast.py` and `decile_calibration.py` accept CLI paths, so the project can run from any folder.
- The deployment service binds `0.0.0.0` and uses relative URLs, so it works behind a preview proxy or on localhost.
