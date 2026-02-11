import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from dataset_manager import (
    list_builtin_datasets,
    load_builtin_dataset,
    load_uploaded_dataset,
    basic_validation
)

from representation_engine import build_representation
from anomaly_engine import run_anomaly_engine
from explanation_engine import explain_row


# ================= SESSION STATE =================
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
    st.session_state.results = None
    st.session_state.z_threshold = 2.5


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Explainable Anomaly Detection",
    layout="wide"
)

st.title("🧠 Intelligent Explainable Anomaly Detection Platform")
st.caption("Automatic Detection | Cost vs Anomaly Highlighting | Human-Readable Explanations")


# ================= STEP 1: DATASET =================
st.header(" Dataset Selection")

source = st.radio(
    "Choose dataset source",
    ["Built-in datasets", "Upload your own CSV"]
)

df = None
meta = None

if source == "Built-in datasets":
    datasets = list_builtin_datasets()
    dataset_name = st.selectbox("Select dataset", datasets)
    if dataset_name:
        df, meta = load_builtin_dataset(dataset_name)
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df, meta = load_uploaded_dataset(uploaded_file)


# ================= VALIDATION =================
if df is not None:
    st.header(" Dataset Validation")

    issues = basic_validation(df)
    if issues:
        st.error("Dataset issues detected:")
        for issue in issues:
            st.write(f"- {issue}")
        st.stop()
    else:
        st.success("Dataset passed validation")


# ================= PREVIEW =================
if df is not None:
    st.header(" Dataset Preview")

    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.dataframe(df.head(20))


# ================= REPRESENTATION =================
###if df is not None:
   # st.header("4️⃣ Data Representation")

X, rep_info = build_representation(df)
   # st.success("Dataset converted to numeric representation")
   # st.write("Encoded feature matrix shape:", X.shape)


# ================= ANOMALY DETECTION =================
if df is not None:

    st.header(" Automatic Anomaly Detection")

    contamination = st.slider(
        "Expected anomaly proportion",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    z_threshold = st.slider(
        "Explanation Sensitivity (Z-score threshold)",
        min_value=2.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    if st.button("🚀 Detect Anomalies"):

        params = {"contamination": contamination}

        st.session_state.results = run_anomaly_engine(
            X,
            method="Isolation Forest",
            params=params
        )

        st.session_state.results_ready = True
        st.session_state.z_threshold = z_threshold


# ================= RESULTS =================
if df is not None and st.session_state.results_ready:

    results = st.session_state.results
    z_threshold = st.session_state.z_threshold

    df["anomaly"] = results["anomaly_mask"]
    df["anomaly_score"] = results["scores"]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cost_column = numeric_cols[0]

    reference_stats = {
        col: {"mean": df[col].mean(), "std": df[col].std()}
        for col in numeric_cols
    }

    df["explanation"] = df.apply(
        lambda row: explain_row(
            row,
            reference_stats,
            numeric_cols,
            z_thresh=z_threshold
        ) if row["anomaly"] else "Normal behavior",
        axis=1
    )

    # ================= SUMMARY =================
    st.markdown("---")
    st.subheader("📊 Detection Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    col2.metric("Anomalies Detected", int(df["anomaly"].sum()))
    col3.metric("Anomaly %", f"{100 * df['anomaly'].mean():.2f}%")

    st.write(f"🔎 Explanation Threshold (Z-score): {z_threshold}")

    # ================= GRAPH =================
    st.markdown("---")
    st.subheader("📈 Cost Trend with Anomalies Highlighted")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[cost_column],
        mode='markers',
        marker=dict(color='lightblue', size=6),
        name='Normal'
    ))

    fig.add_trace(go.Scatter(
        x=df[df["anomaly"]].index,
        y=df[df["anomaly"]][cost_column],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Anomaly'
    ))

    fig.update_layout(
        title=f"{cost_column} Over Time (Anomalies Highlighted)",
        xaxis_title="Index / Time",
        yaxis_title=cost_column,
        legend_title="Data Type",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= ANOMALY TABLE =================
    st.markdown("---")
    st.subheader("🔍 Detailed Anomalies")

    anomalies_df = (
        df[df["anomaly"] == True]
        .sort_values("anomaly_score", ascending=False)
    )

    st.dataframe(anomalies_df)

    # ================= DOWNLOAD REPORT =================
    st.markdown("---")
    st.subheader("📥 Download Anomaly Report")

    report_df = anomalies_df.copy()

    csv_data = report_df.to_csv(index=True).encode("utf-8")

    st.download_button(
        label="Download Anomaly Report (CSV)",
        data=csv_data,
        file_name="anomaly_report.csv",
        mime="text/csv"
    )
