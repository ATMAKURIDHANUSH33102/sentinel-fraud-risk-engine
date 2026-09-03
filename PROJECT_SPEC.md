# Sentinel — Explainable Payment Fraud Risk Engine

## Objective

Build a defense-only machine-learning system that detects fraudulent payment transactions and routes them into ALLOW, REVIEW, or BLOCK decisions.

The project is being developed for the Razorpay AI Buildathon Track 2 — AI Risk Manager.

## Razorpay requirements

The system must:

1. Address one class of financial loss: payment fraud.
2. Be a working detector.
3. Use measurable precision and recall.
4. Evaluate on a held-out test set.
5. Report false-positive cost honestly.
6. Be strictly defense-only.
7. Demonstrate reliability and failure handling.
8. Provide a working product and clear architecture.
9. Document what broke and how it was fixed.

## Core architecture

Transaction
→ Feature Engineering
→ Fraud Risk Model
→ Risk Score
→ Decision Engine
→ Allow / Review / Block
→ Evidence
→ Audit Log
→ Dashboard

## AI/ML principles

Do not use an LLM for the core fraud decision.

The core fraud prediction must be based on a reproducible ML model.

Compare at least:
- Logistic Regression baseline
- Random Forest or XGBoost

Use validation data for model/threshold selection.

The final test set must remain untouched until final evaluation.

## Required metrics

Report:
- Precision
- Recall
- F1
- PR-AUC
- Confusion matrix
- False-positive rate
- False-positive cost
- False-negative cost
- Expected total cost

Do not use accuracy as the primary metric.

## Decision system

The model outputs a fraud probability.

The decision engine converts it into:

LOW → ALLOW
MEDIUM → REVIEW
HIGH → BLOCK

Thresholds must be configurable and justified using validation data.

## Explainability

For each flagged transaction, expose structured evidence such as:
- model score
- transaction amount
- relevant derived features
- model feature importance / explanation

Do not invent explanations using an LLM.

## Safety

The application must only perform defensive analysis.

It must not:
- generate fraud instructions
- simulate attack strategies
- optimize evasion
- provide offensive fraud capabilities

## Engineering requirements

Use:
- Python
- scikit-learn
- pandas
- FastAPI
- SQLite
- Streamlit

Keep the architecture simple enough to complete quickly.

## Quality requirements

Prioritize:
1. Correctness
2. Reproducibility
3. Honest evaluation
4. Clear architecture
5. Failure handling
6. Explainability
7. Usability

Avoid unnecessary frameworks and dependencies.

Never fabricate experimental results.

Never claim a model is good without measured evaluation.