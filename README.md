# Sentinel — Explainable Payment Fraud Risk Engine

Sentinel is a defense-only payment fraud risk engine developed for **Razorpay AI Buildathon Track 2 (AI Risk Manager)**. It evaluates real-time transactions, estimates fraud probabilities via machine learning, routes decisions into **ALLOW**, **REVIEW**, or **BLOCK** risk tiers, and outputs human-interpretable evidence without relying on non-deterministic LLMs.

---

## 1. Project Overview

Payment systems operate in high-throughput, latency-critical environments with extreme class imbalance (~3.5% fraud rate). Sentinel replaces naive thresholding with:
- **Supervised Machine Learning:** Non-linear decision boundaries capturing velocity counts, timedeltas, amounts, and card fingerprints.
- **Cost-Sensitive 3-Tier Routing:** Automated clearance for low-risk transactions (ALLOW), friction-free step-up verification / analyst queue for borderline transactions (REVIEW), and automated decline for high-confidence fraud (BLOCK).
- **Defense-Only Explainability:** Deterministic rule and model-derived evidence (no LLM hallucination in the financial decision path).
- **Audit Logging:** SQLite-backed audit trails recording all scoring events for dispute analysis.

---

## 2. Architecture

```
Inbound Payment Transaction (JSON / REST API / Streamlit Form)
                  │
                  ▼
         [FastAPI: POST /predict]
                  │
                  ▼
         [FraudDetector (src/inference.py)]
                  │
  ┌───────────────┴────────────────────────┐
  ▼                                        ▼
[Feature Engineering]             [Preprocessing: ColumnTransformer]
- log(TransactionAmt)             - Median Imputation + StandardScaler (23 numeric)
- DT_hour, DT_day_of_week         - Constant Imputation + OneHotEncoder (4 categoric)
- Cleaned email domain            - Fitted strictly on Training set (zero leakage)
  └───────────────┬────────────────────────┘
                  ▼
   [Trained Model: RandomForestClassifier]
   - Model predicted fraud probability `p`
                  │
                  ▼
     [Validation-Tuned Risk Engine]
   - p < 0.2977             ──▶  ALLOW  (Low Risk - Instant Clearance)
   - 0.2977 <= p < 0.8023   ──▶  REVIEW (Medium Risk - Step-Up Auth / Queue)
   - p >= 0.8023            ──▶  BLOCK  (High Risk - Automated Decline)
                  │
                  ▼
   [Structured Evidence Generation]
   - Transaction amount & category
   - Velocity signals (C1-C14 counts, D1-D15 timedeltas)
   - Top model feature importance weights
                  │
  ┌───────────────┴────────────────────────┐
  ▼                                        ▼
[Response Payload]              [SQLite Audit Log: audit_logs]
- fraud_probability             - id, timestamp, transaction_id
- risk_level                    - fraud_probability, risk_level
- decision                      - decision, amount, evidence_json
- evidence & confidence
```

---

## 3. Dataset & Known Limitations

- **Dataset:** IEEE-CIS Fraud Detection benchmark (`data/raw/train_transaction.csv`).
- **Volume:** 590,540 real-world e-commerce transactions across a 6-month timeline.
- **Class Imbalance:** 20,663 fraudulent transactions (3.50% fraud rate).
- **Target:** `isFraud` binary indicator (1 = fraudulent chargeback, 0 = legitimate).
- **Chronological Splitting:**
  - **Train (70%):** 413,378 transactions (`TransactionDT`: 86,400 to 10,437,996)
  - **Validation (15%):** 88,581 transactions (`TransactionDT`: 10,438,003 to 13,151,840)
  - **Held-Out Test (15%):** 88,581 transactions (`TransactionDT`: 13,151,880 to 15,811,131)
  - *Zero future-data leakage:* Validation and test partitions strictly follow training timeframes.
- **Documented Limitation:** `train_identity.csv` is absent from `data/raw/`. Sentinel strictly avoids falling back to Kaggle `test_identity.csv` (which has disjoint IDs 3.66M+ vs 2.98M-3.57M and causes 100% NaN joins). Sentinel operates purely on transaction, card, address, velocity count, and timedelta attributes.

---

## 4. Model Selection & Comparison

Trained on 413,378 rows and evaluated on 88,581 validation transactions:

| Model | Val PR-AUC (Primary) | Val ROC-AUC | Val F1 (th=0.5) | Val Precision | Val Recall | Val FPR |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | `0.2312` | `0.7972` | `0.1739` | `0.0991` | `0.8038` | `25.20%` |
| **Random Forest (Selected)** | **`0.4713`** | **`0.8912`** | **`0.3360`** | **`0.2458`** | **`0.5316`** | **`5.60%`** |

**Selection Rationale:** Random Forest achieved more than **double** the PR-AUC of Logistic Regression (0.4713 vs 0.2312) and superior ROC-AUC (0.8912 vs 0.7972). Payment fraud is characterized by non-linear combinations of velocity counts (`C5`, `C1`, `C13`), timedeltas (`D2`, `D3`), and amounts. Constraining tree depth to `max_depth=12` and `min_samples_leaf=10` ensured rapid training (<20 seconds) while preventing overfitting.

**Top Feature Importance Weights:**
1. `C5` (Transaction count / velocity): `0.0967`
2. `C1` (Card association count): `0.0837`
3. `C13` (Cumulative count): `0.0665`
4. `C14` (Transaction count 14): `0.0643`
5. `D3` (Timedelta from last transaction): `0.0623`

---

## 5. Threshold Logic & Cost Optimization

Thresholds were optimized **exclusively on validation data** using cost-sensitive financial trade-offs:
- **Cost Assumptions (Project Assumptions):**
  - False Positive (FP) friction cost: `$10.00` (customer checkout drop-off, support inquiry)
  - False Negative (FN) fraud cost: `$100.00` (chargeback loss, penalty fees)

**Active Operating Thresholds:**
- **ALLOW (Low Risk):** `p < 0.2977` (instant approval)
- **REVIEW (Medium Risk):** `0.2977 <= p < 0.8023` (stepped-up authentication / analyst queue)
- **BLOCK (High Risk):** `p >= 0.8023` (automated rejection)
- **Cost-Optimal Binary Threshold:** `0.5643` (minimizes expected total validation cost to `$169,390.00`)

---

## 6. Final Held-Out Test Evaluation

Evaluated strictly once on the 88,581 held-out test transactions with frozen thresholds:

- **Held-Out Test PR-AUC (Primary Metric):** `0.4537`
- **Held-Out Test ROC-AUC:** `0.8812`
- **Performance at Cost-Optimal Threshold (0.5643):**
  - **Precision:** `0.2592`
  - **Recall:** `0.5958`
  - **F1 Score:** `0.3612`
  - **False Positive Rate:** `0.0614` (6.14%)
  - **Confusion Matrix:** `TN = 80,247 | FP = 5,251 | FN = 1,246 | TP = 1,837`
  - **Expected Total Cost:** `$177,110.00` ($2.00 per transaction)
- **3-Tier Test Traffic Routing:**
  - **ALLOW (`p < 0.2977`):** 62,687 transactions (`70.8%` of traffic) | Fraud missed: 493
  - **REVIEW (`0.2977 <= p < 0.8023`):** 24,121 transactions (`27.2%` of traffic) | Fraud intercepted: 1,464
  - **BLOCK (`p >= 0.8023`):** 1,773 transactions (`2.0%` of traffic) | Fraud blocked: 1,126 | Block Precision: `63.5%`
  - **Total Fraud Intercepted (Review + Block):** **`84.0%`** (2,590 of 3,083 fraudulent transactions)

---

## 7. Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.14.0)
- Virtual environment (`venv`)

### Installation
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running Tests
```powershell
.\venv\Scripts\python -m pytest tests/ -v
```
*(Runs all 35 tests across imports, data loader, feature engineering, pipeline, evaluation, API, and audit log in ~7s)*

---

## 8. How to Train the Pipeline

To re-run the complete training, validation threshold optimization, and serialization pipeline:

```powershell
.\venv\Scripts\python run_pipeline.py
```

Optional arguments:
- `--data-dir`: Path to raw data directory (default: `data/raw`)
- `--output-dir`: Path to output directory (default: `data/processed`)
- `--fp-cost`: Assumed false positive cost (default: `10.0`)
- `--fn-cost`: Assumed false negative cost (default: `100.0`)

Model artifacts are serialized to `data/processed/model/`:
- `model_pipeline.pkl`: Full bundle (model, preprocessor, thresholds, metadata)
- `fraud_model.pkl`: Trained RandomForestClassifier
- `preprocessor.pkl`: Fitted ColumnTransformer
- `thresholds.pkl`: Selected decision thresholds
- `pipeline_summary.json`: Detailed evaluation report

---

## 9. How to Run the FastAPI Service

Start the production REST API:

```powershell
.\venv\Scripts\python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at: `http://localhost:8000/docs`

### API Endpoints

#### 1. `GET /health`
Returns system status, loaded model name, and active thresholds.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "RandomForest",
  "thresholds": {
    "review_threshold": 0.2977,
    "block_threshold": 0.8023
  },
  "defense_only": true
}
```

#### 2. `POST /predict`
Evaluates an inbound transaction and returns fraud risk decision.
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 750.00,
    "TransactionID": "TXN-101",
    "ProductCD": "W",
    "card1": 1004,
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "anonymous.com",
    "C1": 15.0,
    "C5": 3.0,
    "D1": 0.0,
    "D3": 0.0
  }'
```

**Response Example:**
```json
{
  "fraud_probability": 0.2941,
  "risk_level": "LOW",
  "decision": "ALLOW",
  "confidence": 0.7059,
  "evidence": {
    "transaction_amount": 750.0,
    "important_features": [
      "C5: 0.097",
      "C1: 0.084",
      "C13: 0.067",
      "C14: 0.064",
      "D3: 0.062"
    ],
    "transaction_signals": [
      "Above-average transaction amount ($750.00)",
      "Elevated velocity count (C1=15.0)",
      "Zero timedelta since card/account reference (D1=0)",
      "Privacy-focused email provider (anonymous.com)"
    ],
    "top_model_factors": [
      "C5: 0.097",
      "C1: 0.084",
      "C13: 0.067",
      "C14: 0.064",
      "D3: 0.062"
    ],
    "model_name": "RandomForest",
    "engine": "Sentinel-v1-DefenseOnly"
  },
  "audit_id": 22
}
```

#### 3. `GET /audit/recent?limit=20`
Fetches recent transaction decisions from the SQLite audit database (`data/processed/audit_log.db`).

---

## 10. How to Run the Streamlit Dashboard

Launch the interactive operator dashboard:

```powershell
.\venv\Scripts\streamlit run dashboard.py
```

The dashboard opens in your browser at `http://localhost:8501`:
- **Real-Time Scoring:** Interactive transaction parameter sliders, dropdowns, and quick-fill presets (Legitimate, Suspicious Spike, Borderline).
- **Batch CSV Risk Analysis:** Upload a CSV of transactions to score, review summary KPIs (Total, ALLOW, REVIEW, BLOCK, High Risk Rate), view decision distributions, and download the scored dataset.
- **Decision Display:** Color-coded visual indicator (Green ALLOW / Orange REVIEW / Red BLOCK) and risk meter.
- **Evidence Breakdown:** Live explanation badges displaying rule triggers and model feature factors.
- **Audit Log Table:** Real-time queryable list of recent scored transactions.
- **Model Performance Tab:** Metric transparency table, cost analysis, and confusion matrix.

### Batch CSV Risk Analysis Details
Accessible via the **"📁 Batch CSV Risk Analysis"** tab in the Streamlit dashboard:
- **What It Does:** Upload a CSV file of transactions to score multiple transactions simultaneously using the exact saved Random Forest pipeline and validation-derived decision thresholds (`ALLOW < 0.2977`, `REVIEW [0.2977, 0.8023)`, `BLOCK >= 0.8023`).
- **Input Requirements:** Requires `TransactionAmt` (or `TransactionAMT`). Missing optional features are automatically imputed using the saved pipeline median and constant values.
- **Resilience:** Invalid rows (e.g. non-positive or non-numeric amounts) are isolated into a viewable failed rows table without halting or corrupting the remainder of the batch.
- **Outputs Produced:** 
  - Executive summary KPIs: Total Transactions, ALLOW / REVIEW / BLOCK counts and percentages, and High Risk Rate (BLOCK %).
  - Visual decision distribution bar chart.
  - Interactive table of original transaction attributes enriched with `fraud_probability`, `risk_level`, and `decision`.
  - Automatic persistence of scored transactions to the SQLite audit log (`data/processed/audit_log.db`).
- **Download Scored Results:** Click the **"📥 Download Scored CSV"** button to export the complete scored dataset containing original features and risk decisions.