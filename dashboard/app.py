"""
Telco Churn Prediction Dashboard
================================
End-to-end ML system: AWS S3 / SageMaker / RDS + local Streamlit UI.

Run locally:
    streamlit run app.py

Database modes (auto-detected):
    1. RDS MySQL   - if a .env file is present with RDS credentials
    2. SQLite      - default fallback, uses ./predictions.db with batch
                     predictions seeded from the training set on first run.
                     No AWS account required.

Folder structure expected:
    app.py
    requirements.txt
    .env                          (optional - gitignored)
    model/
        final_model_lr.pkl
        final_scaler.pkl
        model_metadata.json
    data/
        Telco-Customer-Churn.xlsx (optional - enables 'existing customer' picker)
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =============================================================================
# CONFIG
# =============================================================================
MODEL_DIR = Path("model")
DATA_PATHS = [Path("Telco-Customer-Churn.xlsx"),
              Path("data/Telco-Customer-Churn.xlsx")]

# Optional: read RDS credentials from environment (or a .env file). If any are
# missing we fall back to a local SQLite database, so the dashboard runs out of
# the box without an AWS account.
def _load_dotenv_if_present():
    """Tiny .env loader so users don't need python-dotenv installed."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


_load_dotenv_if_present()

RDS_HOST = os.environ.get("RDS_HOST")
RDS_USER = os.environ.get("RDS_USER")
RDS_PASSWORD = os.environ.get("RDS_PASSWORD")
RDS_DATABASE = os.environ.get("RDS_DATABASE")

USE_RDS = all([RDS_HOST, RDS_USER, RDS_PASSWORD, RDS_DATABASE])
SQLITE_PATH = Path("predictions.db")


# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Telco Customer Churn Prediction")
st.caption(
    "End-to-end ML system: S3 (model artifacts) → SageMaker (batch inference) "
    "→ RDS / SQLite (predictions) → Streamlit (UI)"
)


# =============================================================================
# DEMO PERSONAS (for the live prediction tab)
# =============================================================================
DEMO_PERSONAS = {
    "🔴 High-Risk Fiber Customer": {
        "customerID": "DEMO-HIGH-01",
        "gender": "Female", "senior": "No", "partner": "No", "dependents": "No",
        "tenure": 2,
        "phone": "Yes", "multi_lines": "No",
        "internet": "Fiber optic",
        "online_sec": "No", "online_bak": "No", "device_prot": "No",
        "tech_support": "No", "stream_tv": "Yes", "stream_movies": "Yes",
        "contract": "Month-to-month", "paperless": "Yes",
        "payment": "Electronic check",
        "monthly": 95.0, "total": 190.0,
    },
    "🟡 Borderline Customer": {
        "customerID": "DEMO-BORDER-01",
        "gender": "Male", "senior": "No", "partner": "Yes", "dependents": "No",
        "tenure": 18,
        "phone": "Yes", "multi_lines": "Yes",
        "internet": "Fiber optic",
        "online_sec": "No", "online_bak": "Yes", "device_prot": "No",
        "tech_support": "No", "stream_tv": "Yes", "stream_movies": "No",
        "contract": "Month-to-month", "paperless": "Yes",
        "payment": "Credit card (automatic)",
        "monthly": 78.0, "total": 1400.0,
    },
    "🟢 Loyal Long-Term Customer": {
        "customerID": "DEMO-LOYAL-01",
        "gender": "Female", "senior": "No", "partner": "Yes", "dependents": "Yes",
        "tenure": 65,
        "phone": "Yes", "multi_lines": "Yes",
        "internet": "DSL",
        "online_sec": "Yes", "online_bak": "Yes", "device_prot": "Yes",
        "tech_support": "Yes", "stream_tv": "No", "stream_movies": "No",
        "contract": "Two year", "paperless": "No",
        "payment": "Bank transfer (automatic)",
        "monthly": 65.0, "total": 4200.0,
    },
    "🟢 New Low-Risk Customer": {
        "customerID": "DEMO-NEW-01",
        "gender": "Male", "senior": "No", "partner": "Yes", "dependents": "No",
        "tenure": 6,
        "phone": "Yes", "multi_lines": "No",
        "internet": "DSL",
        "online_sec": "Yes", "online_bak": "Yes", "device_prot": "No",
        "tech_support": "Yes", "stream_tv": "No", "stream_movies": "No",
        "contract": "One year", "paperless": "No",
        "payment": "Mailed check",
        "monthly": 55.0, "total": 330.0,
    },
}


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_DIR / "final_model_lr.pkl")
    scaler = joblib.load(MODEL_DIR / "final_scaler.pkl")
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


try:
    model, scaler, metadata = load_model_artifacts()
    THRESHOLD = metadata["threshold_tuned"]
    FEATURE_COLS = metadata["feature_columns"]
except FileNotFoundError as e:
    st.error(
        f"Model files not found. Expected `model/` folder containing "
        f"final_model_lr.pkl, final_scaler.pkl, model_metadata.json. "
        f"Details: {e}"
    )
    st.stop()


# =============================================================================
# RAW CUSTOMER DATA (for the existing-customer dropdown)
# =============================================================================
@st.cache_data(ttl=300)
def load_raw_customers():
    for p in DATA_PATHS:
        if p.exists():
            df = pd.read_excel(p)
            df.columns = df.columns.str.strip()
            return df
    return pd.DataFrame()


# =============================================================================
# DATABASE LAYER (RDS or SQLite)
# =============================================================================
def get_rds_connection():
    import pymysql
    return pymysql.connect(
        host=RDS_HOST, user=RDS_USER, password=RDS_PASSWORD,
        database=RDS_DATABASE, connect_timeout=10,
    )


def init_sqlite():
    """Create predictions table and seed it from the dataset on first run."""
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            customerID      TEXT,
            churn_probability REAL,
            churn_prediction  INTEGER,
            actual_churn      INTEGER,
            threshold_used    REAL,
            timestamp         TIMESTAMP,
            source            TEXT
        )
    """)
    conn.commit()

    # If the table is empty, run a one-time batch scoring against the dataset
    cur.execute("SELECT COUNT(*) FROM predictions")
    if cur.fetchone()[0] == 0:
        raw = load_raw_customers()
        if len(raw) > 0:
            seed_sqlite_from_raw(conn, raw)
    conn.close()


def seed_sqlite_from_raw(conn, raw):
    """Run batch inference on the raw dataset and insert into SQLite."""
    df = raw.copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    ).fillna(0)

    customer_ids = df["customerID"].copy()
    df_model = df.drop("customerID", axis=1)
    if "Churn" in df_model.columns:
        df_model["Churn"] = df_model["Churn"].map({"Yes": 1, "No": 0})
        y_true = df_model["Churn"].values
        df_model = df_model.drop("Churn", axis=1)
    else:
        y_true = [-1] * len(df_model)

    X = pd.get_dummies(df_model, drop_first=True)
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    rows = [
        (str(cid), float(p), int(pr), int(yt),
         float(THRESHOLD), datetime.now(), "batch")
        for cid, p, pr, yt in zip(customer_ids, probs, preds, y_true)
    ]

    conn.executemany("""
        INSERT INTO predictions
        (customerID, churn_probability, churn_prediction, actual_churn,
         threshold_used, timestamp, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()


def get_db_connection():
    if USE_RDS:
        return get_rds_connection()
    return sqlite3.connect(SQLITE_PATH)


@st.cache_data(ttl=30)
def load_predictions():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    return df


def write_prediction(customer_id, prob, pred, threshold):
    conn = get_db_connection()
    cur = conn.cursor()
    if USE_RDS:
        cur.execute(
            """
            INSERT INTO predictions
            (customerID, churn_probability, churn_prediction, actual_churn,
             threshold_used, timestamp, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (customer_id, float(prob), int(pred), -1, float(threshold),
             datetime.now(), "live"),
        )
    else:
        cur.execute(
            """
            INSERT INTO predictions
            (customerID, churn_probability, churn_prediction, actual_churn,
             threshold_used, timestamp, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (customer_id, float(prob), int(pred), -1, float(threshold),
             datetime.now(), "live"),
        )
    conn.commit()
    conn.close()


# Initialize SQLite (no-op for RDS users)
if not USE_RDS:
    init_sqlite()


# =============================================================================
# SIDEBAR - profile selector + inputs
# =============================================================================
db_label = "RDS MySQL (cloud)" if USE_RDS else "SQLite (local)"
st.sidebar.caption(f"📡 Database: **{db_label}**")
st.sidebar.header("Live Customer Prediction")

raw_df = load_raw_customers()
has_raw = len(raw_df) > 0

st.sidebar.markdown("**Load profile from:**")
options = ["— Custom input —"]
options += [f"Demo: {name}" for name in DEMO_PERSONAS.keys()]
if has_raw:
    options += ["─── Existing customer ───"]

choice = st.sidebar.selectbox(
    "Select a profile", options, label_visibility="collapsed"
)

selected_existing_id = None
if choice == "─── Existing customer ───" and has_raw:
    selected_existing_id = st.sidebar.selectbox(
        "Pick a customerID (type to search)",
        options=sorted(raw_df["customerID"].tolist()),
        index=0,
    )


def row_to_inputs(row):
    return {
        "customerID": row["customerID"],
        "gender": row["gender"],
        "senior": "Yes" if int(row["SeniorCitizen"]) == 1 else "No",
        "partner": row["Partner"],
        "dependents": row["Dependents"],
        "tenure": int(row["tenure"]),
        "phone": row["PhoneService"],
        "multi_lines": row["MultipleLines"],
        "internet": row["InternetService"],
        "online_sec": row["OnlineSecurity"],
        "online_bak": row["OnlineBackup"],
        "device_prot": row["DeviceProtection"],
        "tech_support": row["TechSupport"],
        "stream_tv": row["StreamingTV"],
        "stream_movies": row["StreamingMovies"],
        "contract": row["Contract"],
        "paperless": row["PaperlessBilling"],
        "payment": row["PaymentMethod"],
        "monthly": float(row["MonthlyCharges"]),
        "total": float(pd.to_numeric(row["TotalCharges"], errors="coerce") or 0.0),
    }


prefill = None
if choice.startswith("Demo:"):
    persona_name = choice.replace("Demo: ", "")
    prefill = DEMO_PERSONAS[persona_name]
elif choice == "─── Existing customer ───" and selected_existing_id:
    row = raw_df[raw_df["customerID"] == selected_existing_id].iloc[0]
    prefill = row_to_inputs(row)


def pf(key, default):
    return prefill[key] if prefill else default


def idx(opts, value, fallback=0):
    try:
        return opts.index(value)
    except (ValueError, TypeError):
        return fallback


st.sidebar.divider()

# Hidden defaulted fields (low learned weight - kept out of UI for clarity)
customer_id = st.sidebar.text_input(
    "Customer ID", value=pf("customerID", "LIVE-0001")
)
gender = pf("gender", "Male")
senior = pf("senior", "No")
partner = pf("partner", "No")
dependents = pf("dependents", "No")
phone = pf("phone", "Yes")
multi_lines = pf("multi_lines", "No")

st.sidebar.markdown("**Account**")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, int(pf("tenure", 12)))

contract_opts = ["Month-to-month", "One year", "Two year"]
contract = st.sidebar.selectbox(
    "Contract", contract_opts,
    index=idx(contract_opts, pf("contract", "Month-to-month"))
)

yn = ["No", "Yes"]
paperless = st.sidebar.selectbox(
    "Paperless Billing", yn, index=idx(yn, pf("paperless", "No"))
)

payment_opts = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
payment = st.sidebar.selectbox(
    "Payment Method", payment_opts,
    index=idx(payment_opts, pf("payment", "Electronic check"))
)

monthly = st.sidebar.number_input(
    "Monthly Charges ($)", 0.0, 200.0, float(pf("monthly", 70.0)), step=5.0
)
total = st.sidebar.number_input(
    "Total Charges ($)", 0.0, 10000.0, float(pf("total", 800.0)), step=50.0
)

st.sidebar.markdown("**Internet & Services**")
net_opts = ["DSL", "Fiber optic", "No"]
internet = st.sidebar.selectbox(
    "Internet Service", net_opts, index=idx(net_opts, pf("internet", "DSL"))
)

svc_opts = ["No", "Yes", "No internet service"]
online_sec = st.sidebar.selectbox(
    "Online Security", svc_opts, index=idx(svc_opts, pf("online_sec", "No"))
)
online_bak = st.sidebar.selectbox(
    "Online Backup", svc_opts, index=idx(svc_opts, pf("online_bak", "No"))
)
device_prot = st.sidebar.selectbox(
    "Device Protection", svc_opts, index=idx(svc_opts, pf("device_prot", "No"))
)
tech_support = st.sidebar.selectbox(
    "Tech Support", svc_opts, index=idx(svc_opts, pf("tech_support", "No"))
)
stream_tv = st.sidebar.selectbox(
    "Streaming TV", svc_opts, index=idx(svc_opts, pf("stream_tv", "No"))
)
stream_movies = st.sidebar.selectbox(
    "Streaming Movies", svc_opts, index=idx(svc_opts, pf("stream_movies", "No"))
)

st.sidebar.divider()
save_to_db = st.sidebar.button(
    "💾 Save this prediction", type="primary"
)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def build_feature_row(inputs):
    raw = pd.DataFrame([{
        "gender": inputs["gender"],
        "SeniorCitizen": 1 if inputs["senior"] == "Yes" else 0,
        "Partner": inputs["partner"],
        "Dependents": inputs["dependents"],
        "tenure": inputs["tenure"],
        "PhoneService": inputs["phone"],
        "MultipleLines": inputs["multi_lines"],
        "InternetService": inputs["internet"],
        "OnlineSecurity": inputs["online_sec"],
        "OnlineBackup": inputs["online_bak"],
        "DeviceProtection": inputs["device_prot"],
        "TechSupport": inputs["tech_support"],
        "StreamingTV": inputs["stream_tv"],
        "StreamingMovies": inputs["stream_movies"],
        "Contract": inputs["contract"],
        "PaperlessBilling": inputs["paperless"],
        "PaymentMethod": inputs["payment"],
        "MonthlyCharges": inputs["monthly"],
        "TotalCharges": inputs["total"],
    }])
    encoded = pd.get_dummies(raw, drop_first=True)
    for col in FEATURE_COLS:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[FEATURE_COLS]
    return encoded


# =============================================================================
# MAIN
# =============================================================================
tab1, tab2 = st.tabs(["Live Prediction", "Batch Dashboard"])

# -------- Live Prediction --------
with tab1:
    st.subheader("Live Prediction Result")
    st.caption("Predictions update in real time as you change sidebar inputs.")

    inputs = {
        "gender": gender, "senior": senior, "partner": partner,
        "dependents": dependents, "tenure": tenure, "phone": phone,
        "multi_lines": multi_lines, "internet": internet,
        "online_sec": online_sec, "online_bak": online_bak,
        "device_prot": device_prot, "tech_support": tech_support,
        "stream_tv": stream_tv, "stream_movies": stream_movies,
        "contract": contract, "paperless": paperless,
        "payment": payment, "monthly": monthly, "total": total,
    }

    X_row = build_feature_row(inputs)
    X_scaled = scaler.transform(X_row)

    prob = float(model.predict_proba(X_scaled)[0, 1])
    pred = int(prob >= THRESHOLD)

    col1, col2, col3 = st.columns(3)
    with col1:
        if pred == 1:
            st.error("### HIGH RISK\n**Predicted to churn**")
        else:
            st.success("### LOW RISK\n**Predicted to stay**")
    with col2:
        st.metric(
            "Churn Probability",
            f"{prob:.1%}",
            delta=f"{(prob - THRESHOLD):+.1%} vs threshold",
        )
    with col3:
        st.metric("Threshold Used", f"{THRESHOLD:.3f}")
        st.caption("Business-driven (recall ≥ 0.75)")

    st.progress(prob)

    if save_to_db:
        try:
            write_prediction(customer_id, prob, pred, THRESHOLD)
            st.cache_data.clear()  # invalidate the dashboard cache
            st.success(f"✅ Prediction saved (customerID: {customer_id})")
        except Exception as e:
            st.warning(f"Could not save prediction: {e}")

    with st.expander("🧠 Which features are driving this prediction?"):
        coefs = model.coef_[0]
        feature_impact = X_scaled[0] * coefs
        impact_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "impact": feature_impact,
        }).sort_values("impact", key=abs, ascending=False).head(8)
        impact_df["direction"] = impact_df["impact"].apply(
            lambda x: "↑ pushes toward churn" if x > 0 else "↓ pushes toward stay"
        )
        st.caption(
            "Top 8 features ranked by their contribution to this specific "
            "prediction. Positive values push the probability up (toward "
            "churn); negative values push it down (toward stay)."
        )
        st.dataframe(
            impact_df[["feature", "impact", "direction"]].round(3),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.markdown("### Model Summary")
    metrics = metadata["test_metrics_tuned"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test AUC", f"{metrics['auc']:.3f}")
    c2.metric("Recall (churn)", f"{metrics['recall']:.1%}")
    c3.metric("Precision (churn)", f"{metrics['precision']:.1%}")
    c4.metric("F1 (churn)", f"{metrics['f1']:.3f}")

    st.caption(
        f"Model: Logistic Regression  |  Threshold: {THRESHOLD:.3f}  |  "
        f"Features: {len(FEATURE_COLS)}"
    )


# -------- Batch Dashboard --------
with tab2:
    st.subheader("Batch Predictions")
    st.caption(
        "Predictions stored in the database "
        f"({db_label.split(' ')[0]})."
    )

    try:
        df = load_predictions()

        if len(df) == 0:
            st.warning("No predictions in the database yet.")
        else:
            total_customers = len(df)
            flagged = int((df["churn_prediction"] == 1).sum())
            flag_rate = flagged / total_customers if total_customers else 0
            avg_prob = df["churn_probability"].mean()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Scored", f"{total_customers:,}")
            k2.metric("Flagged as High Risk", f"{flagged:,}")
            k3.metric("Flag Rate", f"{flag_rate:.1%}")
            k4.metric("Avg Churn Probability", f"{avg_prob:.3f}")

            st.divider()

            st.markdown("#### Churn Probability Distribution")
            hist_df = pd.DataFrame(
                {"probability": df["churn_probability"].round(2)}
            )
            hist_counts = (
                hist_df.groupby("probability").size().reset_index(name="count")
            )
            st.bar_chart(hist_counts.set_index("probability"))

            st.markdown("#### Top 20 Highest-Risk Customers")
            top20 = df.sort_values(
                "churn_probability", ascending=False
            ).head(20)[[
                "customerID", "churn_probability", "churn_prediction",
                "actual_churn", "timestamp", "source",
            ]]
            st.dataframe(top20, use_container_width=True, hide_index=True)

            live = df[df["source"] == "live"].sort_values(
                "timestamp", ascending=False
            ).head(10)
            if len(live):
                st.markdown("#### Recent Live Predictions")
                st.dataframe(
                    live[["customerID", "churn_probability",
                          "churn_prediction", "timestamp"]],
                    use_container_width=True, hide_index=True,
                )

    except Exception as e:
        st.error(f"Could not load predictions: {e}")


st.divider()
st.caption(
    f"Telco Churn POC  |  Logistic Regression  |  Threshold = {THRESHOLD:.3f}  "
    f"|  S3 → SageMaker → {db_label.split(' ')[0]} → Streamlit"
)
