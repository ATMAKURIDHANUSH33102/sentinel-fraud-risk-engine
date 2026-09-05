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
    st.metric("Test ROC-AUC", "0.8812")

    st.subheader("⚖️ Decision Thresholds")
    st.markdown(f"- **ALLOW (Low Risk):** `p < {detector.th_review:.4f}`")
    st.markdown(f"- **REVIEW (Medium Risk):** `{detector.th_review:.4f} <= p < {detector.th_block:.4f}`")
    st.markdown(f"- **BLOCK (High Risk):** `p >= {detector.th_block:.4f}`")

    st.subheader("📐 Evaluation Operating Threshold")
    st.markdown("**Threshold:** `0.5643`")
    st.caption("Selected on validation data minimizing total cost (FP = $10, FN = $100 simulation assumptions).")


# Main Body: Tabs
tab_predict, tab_batch, tab_audit, tab_model = st.tabs([
    "🔍 Real-Time Scoring", "📁 Batch CSV Risk Analysis", "📋 Audit Log", "📊 Model Performance"
])


# =====================================================================
# TAB 1: REAL-TIME SCORING
# =====================================================================
with tab_predict:
    st.subheader("Score Inbound Transaction")

    # Decision Logic
    st.markdown("##### 🚦 Decision Logic")
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.markdown(
            f"""
            <div style="background-color: rgba(46, 125, 50, 0.12); border-left: 4px solid #2e7d32; padding: 8px 12px; border-radius: 4px;">
                <span style="color: #4caf50; font-weight: bold; font-size: 0.85rem;">LOW RISK</span><br>
                <code style="font-size: 0.85rem;">p &lt; {detector.th_review:.4f}</code><br>
                <span style="font-weight: bold; font-size: 0.95rem;">&rarr; ALLOW</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dl_col2:
        st.markdown(
            f"""
            <div style="background-color: rgba(239, 108, 0, 0.12); border-left: 4px solid #ef6c00; padding: 8px 12px; border-radius: 4px;">
                <span style="color: #ffa726; font-weight: bold; font-size: 0.85rem;">MEDIUM RISK</span><br>
                <code style="font-size: 0.85rem;">{detector.th_review:.4f} &le; p &lt; {detector.th_block:.4f}</code><br>
                <span style="font-weight: bold; font-size: 0.95rem;">&rarr; REVIEW</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dl_col3:
        st.markdown(
            f"""
            <div style="background-color: rgba(198, 40, 40, 0.12); border-left: 4px solid #c62828; padding: 8px 12px; border-radius: 4px;">
                <span style="color: #ef5350; font-weight: bold; font-size: 0.85rem;">HIGH RISK</span><br>
                <code style="font-size: 0.85rem;">p &ge; {detector.th_block:.4f}</code><br>
                <span style="font-weight: bold; font-size: 0.95rem;">&rarr; BLOCK</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")

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
                st.info("⚠️ **REVIEW**: Additional verification / manual inspection recommended")
            else:
                st.info("⛔ **BLOCK**: Transaction rejected to prevent chargeback")

        # Decision Rationale
        if decision == "BLOCK":
            st.error(
                f"**Decision Rationale:** Fraud probability `{p:.4f}` meets or exceeds the high-risk block threshold "
                f"(`p >= {detector.th_block:.4f}`). Automated transaction block applied."
            )
        elif decision == "REVIEW":
            st.warning(
                f"**Decision Rationale:** Fraud probability `{p:.4f}` falls within the review range "
                f"(`{detector.th_review:.4f} <= p < {detector.th_block:.4f}`). Additional verification / manual inspection recommended."
            )
        else:
            st.success(
                f"**Decision Rationale:** Fraud probability `{p:.4f}` is below the review threshold "
                f"(`p < {detector.th_review:.4f}`). Instant frictionless clearance applied."
            )

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
# TAB 2: BATCH CSV RISK ANALYSIS
# =====================================================================
with tab_batch:
    st.subheader("📁 Batch CSV Risk Analysis")
    st.caption("Upload a transaction CSV to evaluate risk across multiple transactions simultaneously.")

    st.markdown("""
    **Required Column:** `TransactionAmt` (or `TransactionAMT`).  
    **Supported Columns:** `TransactionID`, `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `dist1`, `P_emaildomain`, `C1`–`C14`, `D1`–`D15`.  
    *(Missing optional fields are automatically imputed using the saved pipeline median and constant values).*
    """)

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"],
        key="batch_csv_uploader",
        help="Upload a CSV file containing transaction records.",
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse CSV file: {e}")
            input_df = None

        if input_df is not None:
            st.info(f"Loaded **{len(input_df)}** rows and **{len(input_df.columns)}** columns from uploaded file.")

            with st.expander("Preview Uploaded Data", expanded=False):
                st.dataframe(input_df.head(5), width="stretch")

            if st.button("⚡ Run Batch Risk Analysis", type="primary", key="btn_run_batch"):
                with st.spinner("Scoring transactions through Sentinel risk engine..."):
                    try:
                        scored_df, failed_rows, summary = detector.score_dataframe(input_df)

                        # KPI Summary
                        st.markdown("### 📊 Batch Risk Summary")
                        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
                        kpi_col1.metric("Total Transactions", summary["total_transactions"])
                        kpi_col2.metric("✅ ALLOW", f"{summary['allow_count']} ({summary['allow_rate']:.1%})")
                        kpi_col3.metric("⚠️ REVIEW", f"{summary['review_count']} ({summary['review_rate']:.1%})")
                        kpi_col4.metric("⛔ BLOCK", f"{summary['block_count']} ({summary['block_rate']:.1%})")
                        kpi_col5.metric("🚨 High Risk Rate", f"{summary['high_risk_rate']:.1%}")

                        st.markdown(
                            f"**Successfully Scored:** `{summary['successfully_scored']}` | "
                            f"**Failed Rows:** `{summary['failed_transactions']}`"
                        )

                        # Error Isolation - viewable failed rows
                        if failed_rows:
                            with st.expander(f"⚠️ View {len(failed_rows)} Failed Rows", expanded=True):
                                st.warning("The following rows failed input validation and were skipped without crashing the batch:")
                                st.dataframe(pd.DataFrame(failed_rows), width="stretch")

                        if summary["successfully_scored"] > 0:
                            # ALLOW / REVIEW / BLOCK bar chart
                            chart_df = pd.DataFrame({
                                "Decision": ["ALLOW", "REVIEW", "BLOCK"],
                                "Count": [summary["allow_count"], summary["review_count"], summary["block_count"]],
                            }).set_index("Decision")
                            st.bar_chart(chart_df)

                            # Audit Logging
                            batch_audit_records = []
                            for _, r in scored_df.iterrows():
                                raw_amt = r.get("TransactionAmt", r.get("TransactionAMT"))
                                batch_audit_records.append({
                                    "transaction_id": r.get("TransactionID"),
                                    "fraud_probability": r["fraud_probability"],
                                    "risk_level": r["risk_level"],
                                    "decision": r["decision"],
                                    "transaction_amount": float(raw_amt) if pd.notna(raw_amt) else None,
                                })
                            try:
                                n_logged = audit_logger.log_batch(batch_audit_records)
                                st.caption(f"Logged {n_logged} transactions to SQLite audit log (`data/processed/audit_log.db`).")
                            except Exception as audit_err:
                                st.warning(f"Could not persist batch to audit log: {audit_err}")

                            # Scored Transaction Table
                            st.markdown("### 📋 Scored Transactions")
                            display_df = scored_df.copy()
                            front_cols = [c for c in ["TransactionID", "TransactionAmt", "fraud_probability", "risk_level", "decision"] if c in display_df.columns]
                            other_cols = [c for c in display_df.columns if c not in front_cols]
                            st.dataframe(display_df[front_cols + other_cols], width="stretch")

                            # Download Scored CSV
                            csv_data = scored_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Scored CSV",
                                data=csv_data,
                                file_name="scored_transactions.csv",
                                mime="text/csv",
                                key="btn_download_scored_csv",
                            )

                    except ValueError as val_err:
                        st.error(f"CSV Validation Error: {val_err}")
                    except Exception as err:
                        st.error(f"Unexpected error during batch risk analysis: {err}")


# =====================================================================
# TAB 3: AUDIT LOG
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
# TAB 4: MODEL PERFORMANCE
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

    st.markdown("---")
    st.subheader("📐 Evaluation Operating Threshold")
    st.markdown("""
    - **Operating Threshold:** `0.5643`
    - **Selection Methodology:** Selected strictly on validation data to minimize total expected business cost.
    - **Simulation Cost Assumptions:**
      - False Positive (FP) = **$10.00** (customer friction / support overhead)
      - False Negative (FN) = **$100.00** (unrecovered chargeback loss)
    
    *(Note: This single operating threshold is used for binary benchmarking and confusion matrix evaluation. In live transaction processing, Sentinel routes decisions using the 3-tier Decision Thresholds: ALLOW < 0.2977, REVIEW [0.2977, 0.8023), and BLOCK >= 0.8023).*
    """)

    st.markdown("""
    **Test Confusion Matrix (88,581 held-out transactions at operating threshold 0.5643):**
    - **True Negatives (TN):** `80,247` (Legitimate correctly cleared)
    - **False Positives (FP):** `5,251` (Legitimate flagged - $10 friction cost)
    - **False Negatives (FN):** `1,246` (Fraud missed - $100 chargeback cost)
    - **True Positives (TP):** `1,837` (Fraud stopped)
    - **Total Test Cost:** `$177,110.00` ($2.00 per transaction)
    """)