"""
Sentinel Payment Fraud Risk Engine - Streamlit Dashboard

Provides an interactive operator interface for:
- Manual transaction evaluation and real-time risk scoring
- Decision inspection (ALLOW / REVIEW / BLOCK)
- Feature evidence breakdown (amount, velocity signals, model weights)
- Model performance metrics and threshold transparency
- Recent audit logs from SQLite
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FraudDetector
from src.audit import AuditLogger


# Page configuration
st.set_page_config(
    page_title="Sentinel — Payment Fraud Risk Engine",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_detector():
    """Load and cache the trained fraud detector."""
    try:
        return FraudDetector(model_dir="data/processed/model")
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None


@st.cache_resource
def load_audit_logger():
    """Initialize audit logger."""
    return AuditLogger()


detector = load_detector()
audit_logger = load_audit_logger()


# Header
st.title("🛡️ Sentinel — Explainable Payment Fraud Risk Engine")
st.caption("Razorpay AI Buildathon Track 2 | Defense-Only Deterministic ML Risk Manager")

if detector is None:
    st.error("⚠️ Model artifacts not found in `data/processed/model/`. Run `python run_pipeline.py` first.")
    st.stop()


# Sidebar: Performance & Threshold Reference
with st.sidebar:
    st.header("⚙️ Engine Architecture")
    st.markdown(f"**Model:** `{detector.model_name}`")
    st.markdown("**Pipeline:** `ColumnTransformer` (50 encoded features)")
    st.markdown("**Dataset:** IEEE-CIS Transactions (590,540 rows)")

    st.subheader("🎯 Verified Performance")
    col1, col2 = st.columns(2)
    col1.metric("Val PR-AUC", "0.4713")
    col2.metric("Test PR-AUC", "0.4537")
    
    col3, col4 = st.columns(2)
    col3.metric("Test ROC-AUC", "0.8812")
    col4.metric("Fraud Intercepted", "84.0%")

    st.subheader("⚖️ Decision Thresholds")
    st.markdown(f"- **ALLOW (Low Risk):** `p < {detector.th_review:.4f}`")
    st.markdown(f"- **REVIEW (Medium Risk):** `{detector.th_review:.4f} <= p < {detector.th_block:.4f}`")
    st.markdown(f"- **BLOCK (High Risk):** `p >= {detector.th_block:.4f}`")
    st.caption("Tuned strictly on validation data with FP=$10, FN=$100 cost trade-offs.")


# Main Body: Tabs
tab_predict, tab_audit, tab_model = st.tabs([
    "🔍 Real-Time Scoring", "📋 Audit Log", "📊 Model Performance"
])


# =====================================================================
# TAB 1: REAL-TIME SCORING
# =====================================================================
with tab_predict:
    st.subheader("Score Inbound Transaction")

    preset = st.selectbox(
        "Load Preset Profile:",
        [
            "Custom Transaction",
            "Typical Legitimate Transaction (Low Risk)",
            "Suspicious High-Velocity Spike (High Risk)",
            "Borderline Transaction (Medium Risk)",
        ],
    )

    if preset == "Typical Legitimate Transaction (Low Risk)":
        default_amt = 45.00
        default_pcd = "W"
        default_card1 = 1004
        default_card4 = "visa"
        default_card6 = "debit"
        default_email = "gmail.com"
        default_c1 = 1.0
        default_c5 = 0.0
        default_d1 = 14.0
        default_d3 = 5.0
    elif preset == "Suspicious High-Velocity Spike (High Risk)":
        default_amt = 1250.00
        default_pcd = "C"
        default_card1 = 9876
        default_card4 = "mastercard"
        default_card6 = "credit"
        default_email = "anonymous.com"
        default_c1 = 25.0
        default_c5 = 8.0
        default_d1 = 0.0
        default_d3 = 0.0
    elif preset == "Borderline Transaction (Medium Risk)":
        default_amt = 320.00
        default_pcd = "W"
        default_card1 = 4567
        default_card4 = "visa"
        default_card6 = "credit"
        default_email = "yahoo.com"
        default_c1 = 4.0
        default_c5 = 1.0
        default_d1 = 1.0
        default_d3 = 1.0
    else:
        default_amt = 100.00
        default_pcd = "W"
        default_card1 = 5000
        default_card4 = "visa"
        default_card6 = "debit"
        default_email = "gmail.com"
        default_c1 = 1.0
        default_c5 = 0.0
        default_d1 = 0.0
        default_d3 = 0.0

    with st.form("transaction_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("##### 💳 Payment Basics")
            amt = st.number_input("Transaction Amount ($)", min_value=0.01, value=float(default_amt), step=10.0)
            product_cd = st.selectbox("Product Code", ["W", "C", "H", "R", "S"], index=["W", "C", "H", "R", "S"].index(default_pcd))
            card1 = st.number_input("Card Identifier (card1)", min_value=1000, max_value=99999, value=int(default_card1))
            card4 = st.selectbox("Card Brand", ["visa", "mastercard", "discover", "american express"], index=["visa", "mastercard", "discover", "american express"].index(default_card4))
            card6 = st.selectbox("Card Type", ["debit", "credit"], index=["debit", "credit"].index(default_card6))

        with col_b:
            st.markdown("##### 📍 Location & Identity")
            addr1 = st.number_input("Billing State/Region (addr1)", min_value=0, max_value=600, value=315)
            addr2 = st.number_input("Billing Country (addr2)", min_value=0, max_value=120, value=87)
            dist1 = st.number_input("Distance from Zip (dist1)", min_value=0.0, value=12.0)
            email = st.selectbox("Email Domain", ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", "other", "missing"], index=["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", "other", "missing"].index(default_email if default_email in ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com"] else "other"))
            txn_id = st.text_input("Transaction ID (Optional)", value="TXN-DEMO-001")

        with col_c:
            st.markdown("##### ⏱️ Velocity & Timedeltas")
            c1 = st.number_input("Card Phone Count (C1)", min_value=0.0, value=float(default_c1))
            c5 = st.number_input("Transaction Velocity Count (C5)", min_value=0.0, value=float(default_c5))
            c13 = st.number_input("Cumulative Count (C13)", min_value=0.0, value=1.0)
            d1 = st.number_input("Days from First Activity (D1)", min_value=0.0, value=float(default_d1))
            d3 = st.number_input("Days from Last Transaction (D3)", min_value=0.0, value=float(default_d3))

        submitted = st.form_submit_button("⚡ Predict Fraud Risk", type="primary", use_container_width=True)

    if submitted:
        input_payload = {
            "TransactionAmt": amt,
            "TransactionID": txn_id,
            "ProductCD": product_cd,
            "card1": card1,
            "card4": card4,
            "card6": card6,
            "addr1": addr1,
            "addr2": addr2,
            "dist1": dist1,
            "P_emaildomain": email,
            "C1": c1,
            "C5": c5,
            "C13": c13,
            "D1": d1,
            "D3": d3,
        }

        # Predict
        result = detector.score(input_payload)
        p = result["fraud_probability"]
        decision = result["decision"]
        risk_level = result["risk_level"]
        confidence = result["confidence"]
        evidence = result["evidence"]

        # Record to SQLite audit
        audit_id = audit_logger.log(
            fraud_probability=p,
            risk_level=risk_level,
            decision=decision,
            transaction_id=txn_id,
            transaction_amount=amt,
            evidence=evidence,
        )

        st.markdown("---")
        st.subheader("🎯 Risk Decision")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Fraud Probability", f"{p:.1%}", help="Estimated posterior fraud likelihood from trained Random Forest.")
            st.progress(p)

        with res_col2:
            st.metric("Risk Level", risk_level)
            if risk_level == "LOW":
                st.success("🟢 LOW RISK: Safe for automated processing")
            elif risk_level == "MEDIUM":
                st.warning("🟠 MEDIUM RISK: Step-up authentication / analyst review needed")
            else:
                st.error("🔴 HIGH RISK: Automated decline")

        with res_col3:
            st.metric("Routing Action", decision)
            if decision == "ALLOW":
                st.info("✅ **ALLOW**: Instant frictionless clearance")
            elif decision == "REVIEW":
                st.info("⚠️ **REVIEW**: Routed to manual inspection / 3DS challenge")
            else:
                st.info("⛔ **BLOCK**: Transaction rejected to prevent chargeback")

        st.markdown("#### 🔬 Explainability Evidence")
        ev_col1, ev_col2 = st.columns(2)

        with ev_col1:
            st.markdown("**Rule & Context Signals:**")
            if evidence["transaction_signals"]:
                for s in evidence["transaction_signals"]:
                    st.markdown(f"- ⚠️ {s}")
            else:
                st.markdown("- ✅ No elevated transaction rule flags triggered")

        with ev_col2:
            st.markdown("**Top Contributing Model Factors:**")
            for f in evidence["top_model_factors"]:
                st.markdown(f"- 📊 `{f}`")

        st.caption(f"Audit log entry #{audit_id} created in `data/processed/audit_log.db`")


# =====================================================================
# TAB 2: AUDIT LOG
# =====================================================================
with tab_audit:
    st.subheader("Recent Prediction Audit Trail (SQLite)")
    recent_logs = audit_logger.get_recent(limit=30)

    if recent_logs:
        df_audit = pd.DataFrame(recent_logs)
        df_display = df_audit[[
            "id", "timestamp", "transaction_id", "transaction_amount",
            "fraud_probability", "risk_level", "decision"
        ]].copy()
        df_display["fraud_probability"] = df_display["fraud_probability"].map(lambda x: f"{x:.2%}")
        df_display["transaction_amount"] = df_display["transaction_amount"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No audit logs recorded yet. Score a transaction above to create an entry.")


# =====================================================================
# TAB 3: MODEL PERFORMANCE
# =====================================================================
with tab_model:
    st.subheader("Model Performance & Honest Evaluation")
    st.markdown("""
    **Evaluation Principles Applied:**
    - Strict chronological splitting via `TransactionDT` (no future leakage).
    - Thresholds tuned exclusively on validation data.
    - Honest reporting on held-out test set (touched once).
    - Metric focus: **PR-AUC, Precision, Recall, and Expected Cost** (Accuracy rejected due to class imbalance).
    """)

    perf_data = {
        "Metric": ["PR-AUC", "ROC-AUC", "F1 Score", "Precision", "Recall", "False Positive Rate"],
        "Validation Set": ["0.4713", "0.8912", "0.3855", "0.2862", "0.5904", "5.24%"],
        "Held-Out Test Set": ["0.4537", "0.8812", "0.3612", "0.2592", "0.5958", "6.14%"],
    }
    st.table(pd.DataFrame(perf_data))

    st.markdown("""
    **Test Confusion Matrix (88,581 held-out transactions at threshold 0.5643):**
    - **True Negatives (TN):** `80,247` (Legitimate correctly cleared)
    - **False Positives (FP):** `5,251` (Legitimate flagged - $10 friction cost)
    - **False Negatives (FN):** `1,246` (Fraud missed - $100 chargeback cost)
    - **True Positives (TP):** `1,837` (Fraud stopped)
    - **Total Test Cost:** `$177,110.00` ($2.00 per transaction)
    """)