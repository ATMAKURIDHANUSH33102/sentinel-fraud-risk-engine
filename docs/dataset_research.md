# Dataset Research — Phase 1: Problem and Data Understanding

**Date:** 2026-09-01  
**Status:** Research Complete — Recommendation Ready  
**Project:** Sentinel — Explainable Payment Fraud Risk Engine

---

## Executive Summary

This document compares two candidate fraud detection datasets for the Sentinel project. After evaluating both against criteria for problem understanding, feature engineering, temporal reasoning, and honest evaluation within a 13-day student project timeline, **the IEEE-CIS Fraud Detection dataset emerges as the stronger choice** — not because it produces higher model accuracy, but because it provides a realistic, bounded problem that supports demonstrating all seven learning objectives.

---

## Dataset Comparison

### 1. IEEE-CIS Fraud Detection Dataset (Kaggle Competition)

**Source & Access**
- **Official URL:** https://www.kaggle.com/competitions/ieee-fraud-detection
- **Data Provider:** Vesta Corporation (real-world payment fraud leader)
- **License:** Subject to Kaggle competition rules; publicly available
- **Data Type:** REAL e-commerce transaction data (not simulated)
- **Citation:** Kaggle Competition (2019), 27,053 teams participated

**Dataset Characteristics**

| Property | Details |
|----------|---------|
| **Transaction Records** | 590,541 training transactions |
| **Identity Records** | 144,233 (24% of transactions) |
| **Time Span** | 6 months of data |
| **Temporal Unit** | TransactionDT in seconds (86,400 = 1 day) |
| **Features** | 871 total columns |
| **Feature Types** | Mixed: categorical, numerical, engineered, temporal |
| **Target Variable** | `isFraud` (binary: 0/1) |
| **Class Imbalance** | Imbalanced (fraud minority) |
| **Missing Data** | Identity data missing for ~76% of transactions |
| **Data Format** | Train/test split provided; identity/transaction tables |

**Feature Categories**

| Category | Examples | Count |
|----------|----------|-------|
| **Transaction** | TransactionID, TransactionDT, TransactionAMT, ProductCD | Core |
| **Card Info** | card1-card6 (card type, category, bank, country) | 6 features |
| **Address** | addr1 (billing region), addr2 (billing country) | 2 features |
| **Email** | P_emaildomain, R_emaildomain | 2 features |
| **Counting** | C1-C14 (e.g., # of addresses per card) | 14 features |
| **Time Deltas** | D1-D15 (e.g., days since previous transaction) | 15 features |
| **Matching** | M1-M9 (e.g., address matches card holder name) | 9 features |
| **Identity** | id_12-id_38 (device rating, proxy rating, behavioral) | 27+ features |
| **Device** | DeviceType, DeviceInfo | 2 features |
| **Vesta Engineered** | V1-V339 (rich ranking, counting, entity relations) | 339+ features |

**Labeling Logic**
- **Fraud = 1:** Transaction is a reported chargeback OR linked to chargebacked transaction
- **Legit = 0:** No reported chargeback within 120 days
- **Caveat:** Unreported fraud may be labeled legit (acknowledged limitation)

**Advantages**

✅ **Real data** — e-commerce transactions from actual fraud prevention system  
✅ **Rich temporal information** — D1-D15 features capture transaction history patterns  
✅ **Entity relationships** — Can construct customer/card/merchant history  
✅ **Realistic class imbalance** — Mirrors real fraud prevalence  
✅ **Meaningful false-positive cost** — Decline cost vs. chargeback cost  
✅ **Legitimate train/val/test split** — Temporal ordering prevents leakage  
✅ **Bounded scope** — 590k rows manageable for 13-day project  
✅ **Well-documented** — 262-comment discussion thread on feature meanings  
✅ **Feature engineering potential** — Vesta features show example engineering  
✅ **Risk manager problem** — Detect fraud to prevent chargebacks & customer impact

**Limitations**

⚠️ **Missing identity data** — 76% of transactions lack DeviceType, DeviceInfo, id_12-id_38  
⚠️ **871 features is large** — Feature selection & dimensionality reduction needed  
⚠️ **Labeling delay** — 120-day window means recent transactions unlabeled  
⚠️ **Feature leakage risk** — V-features are Vesta pre-engineered; easy to overfit  
⚠️ **Masked fields** — Card numbers, email domains, IP addresses anonymized (privacy good, interpretability challenging)  

**Suitability for 13-Day Project**

| Criterion | Rating | Reasoning |
|-----------|--------|-----------|
| **Data Size** | ✅ Good | 590k rows: large enough for realistic patterns, small enough for iteration |
| **Feature Richness** | ✅ Excellent | 870+ features support deep exploration and engineering |
| **Temporal Patterns** | ✅ Excellent | D features + TransactionDT enable time-series reasoning |
| **Honest Evaluation** | ✅ Strong | Real labels, clear limitations, leakage risks identifiable |
| **Explainability** | ⚠️ Moderate | Vesta features require reverse-engineering; good learning opportunity |
| **Feature Engineering** | ✅ Excellent | Raw data + derived features = learn both approaches |
| **Time Constraint** | ✅ Good | Manageable if focused: ignore V-features, engineer from C/D/M |

---

### 2. Fraud Detection Handbook Simulated Dataset

**Source & Access**
- **Official Repository:** https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
- **Authors:** Yann-Aël Le Borgne, Wissam Siblini, Bertrand Lebichot, Gianluca Bontempi (ULB-MLG/Worldline)
- **License:** GPL v3.0 (code), CC BY-SA 4.0 (content/data)
- **Data Type:** SIMULATED credit card transactions
- **Temporal Scope:** Reproducible research handbook; active since 2020

**Dataset Characteristics**

| Property | Details |
|----------|---------|
| **Transaction Records** | ~100k simulated transactions (typical research size) |
| **Time Span** | Configurable (research focused) |
| **Temporal Unit** | Seconds (configurable) |
| **Features** | Configurable by simulator; ~20-30 typical base features |
| **Feature Types** | Controlled mix: card, merchant, time, amount, customer |
| **Target Variable** | `isFraud` (binary: 0/1) |
| **Class Imbalance** | Configurable; typically 1-2% fraud |
| **Missing Data** | Minimal; synthetic = complete |
| **Data Format** | Python simulator; reproducible |

**Simulator Capabilities**
- CCFRAUD simulator (Worldline research)
- Configurable fraud scenarios
- Repeatable random seed
- Base + sequential fraud patterns
- Customer behavior profiles
- Merchant/terminal profiles

**Advantages**

✅ **Fully reproducible** — Same random seed = same data  
✅ **Transparent** — Know exact fraud generation mechanism  
✅ **No licensing hassle** — GPL/CC-BY-SA clear  
✅ **Interpretable** — Understand why fraud is labeled fraud  
✅ **Configurable** — Can adjust fraud rate, patterns, features  
✅ **Educational** — Handbook teaches ML for fraud alongside data  
✅ **No missing data** — Simulated = complete  
✅ **Clean features** — Not pre-engineered; design from scratch  

**Limitations**

❌ **Simulated, not real** — Risk of learning simulator artifacts, not real fraud  
❌ **Smaller scale** — ~100k records vs. 590k (less diversity)  
❌ **Limited temporal complexity** — Simpler than real e-commerce patterns  
❌ **False sense of control** — Easy to achieve high accuracy on known patterns  
❌ **Leakage temptation** — Know exactly how fraud is generated  
❌ **Less honest evaluation** — "Good results" may reflect simulator, not fraud detection skill  
❌ **Limited business context** — No device fingerprinting, IP analysis, cross-border patterns  
❌ **Feature engineering plateau** — Features limited to simulator design  

**Suitability for 13-Day Project**

| Criterion | Rating | Reasoning |
|-----------|--------|-----------|
| **Data Size** | ✅ Good | 100k rows; smaller but manageable |
| **Feature Richness** | ⚠️ Limited | Simulator-dependent; may not cover real-world complexity |
| **Temporal Patterns** | ⚠️ Moderate | Configurable but simpler than real transactions |
| **Honest Evaluation** | ❌ Weak | Know simulation mechanism; risk of overfitting to simulator |
| **Explainability** | ✅ Excellent | Understand why every fraud flag is generated |
| **Feature Engineering** | ⚠️ Moderate | Design features, but within simulator boundaries |
| **Time Constraint** | ✅ Good | Fast iteration; no download/cleaning delays |

---

## Comparative Analysis

### Problem Understanding & Learning Objectives

**Objective 1: Problem Understanding**

| Dataset | Capability |
|---------|-----------|
| **IEEE-CIS** | ⭐⭐⭐⭐⭐ Understand real e-commerce fraud: chargebacks, false declines, customer friction |
| **Handbook** | ⭐⭐⭐ Understand fraud patterns, but in controlled environment; less stakeholder perspective |

**Objective 2: Feature Engineering**

| Dataset | Capability |
|---------|-----------|
| **IEEE-CIS** | ⭐⭐⭐⭐⭐ Engineer from raw + learn from Vesta examples; C/D/M features as reference |
| **Handbook** | ⭐⭐⭐ Engineer from scratch; limited by simulator feature space |

**Objective 3: Temporal Reasoning**

| Dataset | Capability |
|---------|-----------|
| **IEEE-CIS** | ⭐⭐⭐⭐⭐ D1-D15 model transaction history; 6-month timeline; seasonal patterns possible |
| **Handbook** | ⭐⭐⭐ Temporal features possible, but simpler; no seasonal/drift effects |

**Objective 4: Fraud-Risk Reasoning**

| Dataset | Capability |
|---------|-----------|
| **IEEE-CIS** | ⭐⭐⭐⭐⭐ Card-network fraud, device fingerprinting, merchant reputation, cross-border risk |
| **Handbook** | ⭐⭐⭐ Basic card+merchant fraud; missing device/IP/network aspects |

**Objective 5: Honest Evaluation**

| Dataset | Capability |
|---------|-----------|
| **IEEE-CIS** | ⭐⭐⭐⭐ Real labels (with documented caveats); real class imbalance; realistic leakage risks |
| **Handbook** | ⭐⭐ Known simulator mechanism; temptation to optimize for known patterns, not robustness |

**Objective 6: Failure Analysis**

| Dataset | Capability |
|---------|-----------|
| **IEEE-CIS** | ⭐⭐⭐⭐⭐ Understand false positives (legitimate declines), false negatives (undetected fraud) |
| **Handbook** | ⭐⭐⭐ Can analyze, but limited by simulator patterns |

**Objective 7: Explainability**

| Dataset | Capability |
|---------|-----------|
| **IEEE-CIS** | ⭐⭐⭐⭐ Feature importance shows real risk factors; V-features reverse-engineering challenge |
| **Handbook** | ⭐⭐⭐⭐⭐ Transparent fraud mechanism; easy to explain why fraud = fraud |

---

## Recommendation

### Primary Dataset: **IEEE-CIS Fraud Detection** ✅

**Rationale**

The IEEE-CIS dataset is the stronger choice for Sentinel because:

1. **Real-world grounding** — Builds intuition for actual fraud challenges (chargebacks, false declines, network effects)
2. **Bounded scope** — 590k rows is large enough for realistic ML but small enough for 13-day iteration
3. **Learning-complete** — Supports all seven learning objectives, especially temporal reasoning and honest evaluation
4. **Realistic constraints** — Missing identity data, V-feature leakage risks = learn defensive modeling
5. **Business relevance** — False-positive cost (decline) vs. false-negative cost (chargeback) = risk manager problem
6. **Documentation** — 262-comment Kaggle thread answers implementation questions
7. **Reproducibility** — Publicly available; can compare to 7-year competition leaderboard

**Success factors**
- Start with data exploration (identity + transaction join patterns)
- Feature engineer from C/D/M + basic raw features (avoid V-feature shortcut)
- Use temporal ordering for train/val/test split
- Model decision threshold on validation fraud prevalence
- Report both precision and recall; calculate false-positive cost

---

### Alternative: Fraud Detection Handbook Simulated Dataset (if needed)

**When to use:**
- If IEEE-CIS data becomes unavailable (unlikely)
- For SUPPLEMENTARY practice AFTER completing IEEE-CIS (learn model techniques)
- For teaching explanations (know exactly why fraud is fraud)

**Not recommended as PRIMARY** because:
- Simulated data risks learning simulator artifacts
- Smaller scale misses real-world complexity
- "Good results" may not transfer to production
- Undermines "never fabricate results" principle (simulator-engineered results)

---

## Problem Formulations for IEEE-CIS Dataset

The IEEE-CIS dataset supports multiple, well-defined problem formulations. Each has different prediction targets, information availability, and risk profiles.

---

### **Formulation A: Transaction Fraud Detector (RECOMMENDED)**

**Problem Statement**
Given a transaction's attributes (amount, card, address, email domain), predict whether it is fraudulent.

**Prediction Target**
- **What to predict:** `isFraud` (binary: 0 = legitimate, 1 = fraudulent)
- **When to predict:** At transaction submission (real-time)
- **Prediction action:** Output fraud probability → Decide ALLOW / REVIEW / BLOCK

**Information Available at Prediction Time**

✅ **Current transaction:**
- TransactionDT, TransactionAMT, ProductCD
- card1-card6 (card type, bank, country)
- addr1, addr2 (billing region/country)
- P_emaildomain, R_emaildomain
- Identity info (if available): DeviceType, DeviceInfo, id_12-id_38
- Vesta pre-engineered: V1-V339

⚠️ **Derived (engineer from history):**
- C1-C14 features: # transactions with this card/address/email in time windows (24h, 30d, etc.)
- D1-D15 features: days since previous transaction with same card/email/IP
- M1-M9 features: does address match card holder name? (cross-reference available)

❌ **NOT available:**
- Future chargebacks (labels only known retroactively)
- Outcomes of similar transactions (creates feedback loop risk)

**Action System**

| Fraud Score | Decision | Effect | Cost |
|------------|----------|--------|------|
| **0.0 - 0.30** | ALLOW | Approve transaction | 0 (if legit) / -$X (if fraud) |
| **0.30 - 0.70** | REVIEW | Send 2FA, call customer, ask questions | -$0.50 (overhead) |
| **0.70 - 1.0** | BLOCK | Decline transaction | 0 (if fraud) / -$1.00 (customer friction) |

**False Positive (FP) Definition**
- True label: Legitimate transaction
- System output: Fraud probability > threshold → BLOCK or REVIEW
- **Cost:** Declined legit card → customer frustration, incomplete purchase, lost revenue
- **Example:** Customer at foreign airport; high distance, unusual merchant, unfamiliar device → flagged as fraud but is real

**False Negative (FN) Definition**
- True label: Fraudulent transaction
- System output: Fraud probability < threshold → ALLOW
- **Cost:** Fraudster approves; chargeback filed 30 days later; refund + processing fee + dispute time
- **Example:** Fraudster uses stolen card at 3 AM from different country; historical patterns incomplete; model underestimates risk

**Key Metric**
- **Primary:** Precision at configurable recall threshold (optimize for business cost)
- **Secondary:** False-positive rate (customer experience), False-negative rate (chargeback loss)
- **NOT:** Accuracy (misleading when fraud is 1-2% of data)

**Implementation Difficulty: MEDIUM**

- ✅ Straightforward: binary classification pipeline
- ⚠️ Moderate challenge: temporal feature engineering (C/D features from transaction history)
- ⚠️ Leakage risk: C/D features require careful time window selection
- ⚠️ Imbalance: need weighted loss or resampling

**Leakage Risks**

1. **Information leakage via D-features:** If computing D1 (days to previous transaction), must ensure "previous" uses only past data at predict time
2. **Forward-looking C-features:** If C5 = "# transactions in next 30 days", that's future info; must use lookback windows only
3. **Vesta V-features:** Pre-engineered at Vesta; don't reverse-engineer their logic = risk of accidental overfitting

**Recommended Approach**
- Use C/D/M features as provided (assume Vesta engineered correctly for real-time)
- Focus on understanding fraud patterns in TransactionAMT, card info, address anomalies
- Use validation set to tune decision threshold
- Test on held-out test set with temporal ordering (no time leakage)

---

### **Formulation B: Fraud Spike Detector (Alternative)**

**Problem Statement**
Detect when a card, merchant, or customer suddenly exhibits abnormal fraud activity.

**Prediction Target**
- **What to predict:** Is this transaction part of a fraud campaign / burst?
- **Granularity:** Card-level, merchant-level, or customer-level risk score
- **Signal:** Sudden increase in transaction rate, amount, or geographic spread for a single entity

**Information Available at Prediction Time**

✅ **Current transaction + entity history:**
- All current transaction features (as in Formulation A)
- PLUS: historical transactions for the same card/customer/merchant
- Rolling statistics: # transactions in last 24h, 7d, 30d
- Anomaly indicators: deviation from entity's baseline behavior

**Action System**

| Spike Signal | Decision | Effect |
|--------------|----------|--------|
| **Baseline** | ALLOW | Normal behavior |
| **+2σ deviation** | REVIEW | Flag unusual but not blocking |
| **+3σ or +10 txns in 1h** | BLOCK | Likely compromised card |

**False Positive Definition**
- Customer suddenly makes 5 purchases in 1 hour (legitimate: holiday shopping, buying gifts)
- System flags as compromised card
- **Cost:** False block + customer support contact

**False Negative Definition**
- Fraudster makes 3 transactions in 30 minutes using stolen card
- Historical baseline is high (previous legit owner had high volume)
- Spike not detected
- **Cost:** Multiple fraudulent charges before block

**Implementation Difficulty: MEDIUM-HIGH**

- ⚠️ Requires entity grouping (same card/customer across transactions)
- ⚠️ Requires rolling/sliding window aggregation
- ⚠️ Baseline estimation tricky (how long history? cold-start new cards?)

**Leakage Risks**
- Using future transactions to compute rolling statistics
- Mixing training and test entities (must split by entity, not by row)

**Suitable if:**
You want to focus on **temporal aggregation & entity-level reasoning** over individual transaction features

---

### **Formulation C: Compromised Merchant/Terminal Detector (Alternative)**

**Problem Statement**
Detect when a merchant or terminal (POS machine) becomes a fraud hotbed.

**Prediction Target**
- **What to predict:** Is this transaction from a recently-compromised merchant/terminal?
- **Granularity:** Merchant risk score (not transaction-level)
- **Signal:** Sudden increase in fraud FROM this merchant (e.g., ATM used by fraudsters, e-commerce site hacked)

**Information Available at Prediction Time**

✅ **Merchant/terminal history (derived from ProductCD, addr1/addr2):**
- # fraudulent transactions from this merchant in last 7d/30d
- % fraud rate at this merchant (fraud count / total transactions)
- Geographic clustering of fraud from same merchant
- Time-of-day patterns (if fraud is concentrated at certain hours)

**Action System**

| Merchant Risk | Decision | Effect |
|----------------|----------|--------|
| **Fraud rate 0-0.5%** | ALLOW | Low-risk merchant |
| **Fraud rate 0.5-2%** | REVIEW_HIGH | Elevated scrutiny |
| **Fraud rate >2%** | BLOCK_MERCHANT | Suspend merchant temporarily |

**False Positive Definition**
- Legitimate merchant has bad luck: 2 fraudulent transactions in 100 from criminals using stolen cards
- System flags merchant as compromised
- **Cost:** Merchant loses transactions, revenue, reputation

**False Negative Definition**
- Merchant's POS system is hacked; 50 fraudulent transactions in 1 day
- Merchant fraud rate suddenly 5%
- System doesn't detect (uses 30-day window; surge is recent)
- **Cost:** Many more frauds before alert

**Implementation Difficulty: HARD**

- ⚠️ Requires merchant identification (may need to infer from ProductCD + addr)
- ⚠️ Requires significant statistical threshold tuning
- ⚠️ Class imbalance at merchant level (most merchants have 0% fraud)

**Leakage Risks**
- Using current transaction's label to compute merchant risk (circular reasoning)

**Suitable if:**
You want to focus on **anomaly detection & merchant risk attribution**

---

## Comparison: Problem Formulations

| Aspect | Formulation A (Transaction) | Formulation B (Spike) | Formulation C (Merchant) |
|--------|---------------------------|----------------------|------------------------|
| **Complexity** | Medium | Medium-High | Hard |
| **Data Availability** | Complete (all columns) | Needs aggregation | Needs merchant ID |
| **Learning Value** | Core: understand features + thresholds | Advanced: temporal aggregation | Advanced: entity analysis |
| **Realistic** | Most realistic (prod fraud detection) | Common in practice | Common in practice |
| **13-Day Fit** | ✅ Best | ⚠️ Good | ⚠️ Challenging |
| **Leakage Risk** | Medium | High | High |
| **Implementation** | Straightforward | Moderate complexity | Significant complexity |

---

## Recommendation: Problem Formulation

### **PRIMARY: Formulation A (Transaction Fraud Detector)**

**Why:**
1. Directly matches Razorpay AI Risk Manager spec
2. Clearest ALLOW/REVIEW/BLOCK decision logic
3. Manageable implementation within 13 days
4. Natural threshold tuning (precision vs. recall)
5. Honest false-positive/negative cost reporting

**Success criteria:**
- Precision, Recall, F1-score on test set
- False-positive rate (legitimate transactions declined)
- False-negative rate (frauds that passed through)
- Decision threshold justified by validation precision
- Explainability: show feature importance + sample explanations

**Optional extension (if ahead of schedule):**
- Add Formulation B spike detection as secondary model
- Compare dual-model approach (transaction + spike) vs. single model

---

## Implementation Roadmap

### Week 1 (Days 1-3): Problem & Data Understanding

```
Day 1: Download IEEE-CIS dataset (no code changes)
       - Understand transaction/identity table structure
       - Document feature descriptions
       - Identify missing data patterns

Day 2: Exploratory Data Analysis (EDA notebook)
       - Load data, inspect shapes, dtypes
       - Fraud prevalence (baseline rate)
       - Feature distributions (categorical vs. numerical)
       - Missing data heatmap (identity columns)
       - Transaction amount distribution
       - Temporal patterns (TransactionDT range)

Day 3: Formulation Decision
       - Choose Formulation A (transaction detector)
       - Define decision thresholds (ALLOW/REVIEW/BLOCK)
       - Document fraud cost assumptions
       - Plan train/val/test split strategy
```

### Week 2 (Days 4-7): Feature Engineering & Model Building

```
Day 4: Feature Engineering
       - Compute C-features: card/address/email counts (time windows)
       - Compute D-features: days since previous transaction
       - Compute M-features: matching checks
       - Handle missing identity data

Day 5-6: Model Comparison
       - Logistic Regression (baseline)
       - Random Forest (non-linear)
       - Model training + cross-validation on validation set
       - Compare PR-AUC, precision, recall

Day 7: Threshold Optimization
       - Use validation set to find optimal threshold
       - Justify ALLOW/REVIEW/BLOCK cutoffs
       - Calculate false-positive cost vs. false-negative cost
```

### Week 2 (Days 8-13): Evaluation & Reporting

```
Day 8-9: Final Evaluation
       - Test set evaluation (never touched during tuning)
       - Precision, Recall, F1, PR-AUC
       - Confusion matrix
       - Honest discussion of limitations

Day 10-11: Explainability & Failure Analysis
       - Feature importance plots
       - Sample explanations (why was this transaction flagged?)
       - Failure analysis (false positives + false negatives)
       - Document what went wrong + fixes applied

Day 12-13: Final Report & Documentation
       - Update DECISIONS.md with all choices
       - Create fraud_analysis.md with results
       - Wrap up: what we learned, what we'd change
```

---

## Risk Mitigation

**Data Risk: Are these real labels?**
- ✅ Vesta confirmed: real chargeback data, real e-commerce transactions
- ⚠️ Known limitation: unreported fraud labeled as legit (acknowledged)
- Mitigation: Report in final evaluation

**Leakage Risk: C/D features use future data?**
- ⚠️ Vesta engineered for real-time use; trust engineering
- Mitigation: Validate C/D features manually on sample data; ensure lookback only

**Class Imbalance Risk: 1-2% fraud skews evaluation?**
- ✅ Use PR-AUC, not ROC-AUC (better for imbalanced data)
- ✅ Use weighted loss or resampling
- Mitigation: Report both precision and recall, not accuracy

**Time Risk: 13 days too short?**
- ✅ Dataset is bounded (590k rows, not 10M)
- ✅ Formulation A is straightforward classification
- Mitigation: Skip Vesta V-features; focus on interpretable C/D/M features

---

## Conclusion

**Dataset Decision: IEEE-CIS Fraud Detection** ✅

- Supports all learning objectives
- Real fraud patterns + realistic constraints
- Bounded scope for 13-day delivery
- Well-documented dataset + 262-comment FAQ
- Risk manager problem = explicit ALLOW/REVIEW/BLOCK decisions
- Honest evaluation possible (clear leakage/limitation risks)

**Problem Formulation: Transaction Fraud Detector (Formulation A)** ✅

- Matches Razorpay spec exactly
- Binary classification with configured threshold
- Clear false-positive / false-negative trade-off
- Directly supports precision/recall/cost reporting
- Implementation clear and achievable

**Not Yet Decided (RESEARCH ONLY):**
- Specific model (Logistic Regression vs. Random Forest vs. XGBoost) — will compare in Phase 2
- Exact decision thresholds — will optimize on validation set
- Feature engineering specifics — will design in Phase 2 based on EDA

---

## Sources Researched

1. **IEEE-CIS Fraud Detection (Kaggle)**
   - Competition page: https://www.kaggle.com/competitions/ieee-fraud-detection
   - Data description: https://www.kaggle.com/competitions/ieee-fraud-detection/data
   - Feature discussion: https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203 (262 comments)
   
2. **Fraud Detection Handbook**
   - GitHub: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
   - Online: https://fraud-detection-handbook.github.io/
   - Authors: Le Borgne, Siblini, Lebichot, Bontempi (ULB-MLG & Worldline)

3. **Supporting Research**
   - Vesta Corporation: Real e-commerce fraud leader; provided IEEE-CIS data
   - Worldline: Collaborated on fraud detection handbook; provided CCFRAUD simulator
   - 7-year competition (2019-2026): 27,053 teams, 125k+ submissions

---

*Report prepared for Sentinel Phase 1 — Problem and Data Understanding*  
*Decision-ready; awaiting team approval before Phase 2 model development*
