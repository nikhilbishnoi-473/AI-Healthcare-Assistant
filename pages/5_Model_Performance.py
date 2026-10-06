import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))

from prediction import load_model_artifact
from preprocessing import load_dataset, inspect_dataset
from helpers import apply_custom_css, init_session_state, display_disclaimer

st.set_page_config(page_title="Model Performance | AI Healthcare", page_icon="📊", layout="wide")
apply_custom_css()
init_session_state()

st.title("📊 Model Performance & Data Analytics Dashboard")
st.write("Compare Machine Learning algorithms, inspect evaluation metrics, and explore dataset statistics.")

display_disclaimer()

disease_choice = st.radio("Select Disease Model Pipeline to Inspect", ["Heart Disease Models", "Diabetes Models"], horizontal=True)

dataset_key = "heart" if "Heart" in disease_choice else "diabetes"
artifact = load_model_artifact(dataset_key)

comp_df = pd.DataFrame(artifact["comparison_table"])
best_eval = artifact["evaluation"]

st.subheader(f"🏆 Best Model Selected: {artifact['model_name']}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Recall (Sensitivity)", best_eval["Recall"])
m2.metric("Accuracy", best_eval["Accuracy"])
m3.metric("Precision", best_eval["Precision"])
m4.metric("F1-Score", best_eval["F1-Score"])
m5.metric("ROC-AUC", best_eval["ROC-AUC"])

st.markdown("---")

perf_tab1, perf_tab2, perf_tab3, perf_tab4 = st.tabs([
    "📈 Model Comparison Table & Charts", "🔲 Confusion Matrix", "📉 ROC Curves", "📁 Exploratory Data Analysis (EDA)"
])

# ------------ TAB 1: COMPARISON TABLE & CHARTS ------------
with perf_tab1:
    st.markdown("### Model Benchmark Comparison")
    st.write("Evaluation results across Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.")
    st.dataframe(comp_df, use_container_width=True)
    
    # Plotly Bar Chart comparing Recall and Accuracy
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=comp_df["Model"], y=comp_df["Recall (Sensitivity)"],
        name="Recall (Sensitivity)", marker_color="#EF4444"
    ))
    fig_comp.add_trace(go.Bar(
        x=comp_df["Model"], y=comp_df["Accuracy"],
        name="Accuracy", marker_color="#3B82F6"
    ))
    fig_comp.update_layout(
        title="Model Recall vs Accuracy Comparison",
        barmode="group", yaxis=dict(range=[0, 1.0]), height=400
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ------------ TAB 2: CONFUSION MATRIX ------------
with perf_tab2:
    st.markdown("### Confusion Matrix Analysis")
    st.write(f"Displaying confusion matrix for the optimal selected model: **{artifact['model_name']}**")
    
    cm = np.array(best_eval["Confusion_Matrix"])
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted Label", y="Actual True Label", color="Sample Count"),
        x=["Negative (0)", "Positive (1)"],
        y=["Negative (0)", "Positive (1)"]
    )
    fig_cm.update_layout(title=f"Confusion Matrix ({artifact['model_name']})", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

# ------------ TAB 3: ROC CURVE ------------
with perf_tab3:
    st.markdown("### Receiver Operating Characteristic (ROC) Curve")
    st.write("Trade-off between True Positive Rate (Sensitivity) and False Positive Rate.")
    
    all_evals = artifact["all_evaluations"]
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, x1=1, y0=0, y1=1)
    
    for ev in all_evals:
        roc_data = ev["ROC_Curve"]
        fig_roc.add_trace(go.Scatter(
            x=roc_data["fpr"], y=roc_data["tpr"],
            mode="lines", name=f"{ev['Model']} (AUC = {ev['ROC-AUC']})"
        ))
        
    fig_roc.update_layout(
        title="Multi-Model ROC Curves",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR / Sensitivity)",
        height=450
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# ------------ TAB 4: EDA ------------
with perf_tab4:
    st.markdown("### Dataset Exploration & Distribution Analytics")
    df_raw = load_dataset(dataset_key)
    inspect = inspect_dataset(df_raw)
    
    st.write(f"**Dataset Dimensions**: {inspect['shape'][0]} Rows, {inspect['shape'][1]} Columns")
    
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        st.write("#### Raw Data Preview")
        st.dataframe(df_raw.head(8), use_container_width=True)
    with c_e2:
        st.write("#### Target Label Distribution")
        target_col = artifact["target_col"]
        counts = df_raw[target_col].value_counts().reset_index()
        counts.columns = ["Target_Value", "Count"]
        fig_pie = px.pie(
            counts, names="Target_Value", values="Count",
            title=f"Target Column ({target_col}) Class Balance",
            color_discrete_sequence=["#10B981", "#F59E0B"]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.write("#### Statistical Summary")
    st.dataframe(df_raw.describe(), use_container_width=True)
