import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))

from prediction import predict_heart_disease_risk, predict_diabetes_risk
from helpers import apply_custom_css, init_session_state, display_disclaimer, add_to_history, clear_history

st.set_page_config(page_title="Disease Prediction | AI Healthcare", page_icon="🔮", layout="wide")
apply_custom_css()
init_session_state()

st.title("🔮 AI Disease Risk Prediction & XAI Analysis")
st.write("Run trained Machine Learning models to evaluate disease risk estimates with Explainable AI.")

display_disclaimer()

# Retrieve saved assessment or use defaults
if "patient_assessment" in st.session_state and st.session_state.patient_assessment:
    patient = st.session_state.patient_assessment
    heart_inputs = patient["heart_inputs"]
    diabetes_inputs = patient["diabetes_inputs"]
    st.success("✅ Loaded active Patient Assessment metrics.")
else:
    st.info("ℹ️ No saved patient assessment found. Using default clinical baseline values. You can fill your metrics on the **1. Health Assessment** page.")
    heart_inputs = {
        'age': 55, 'sex': 1, 'cp': 1, 'trestbps': 130, 'chol': 240, 'fbs': 0,
        'restecg': 0, 'thalach': 140, 'exang': 0, 'oldpeak': 1.2, 'slope': 1, 'ca': 0, 'thal': 2
    }
    diabetes_inputs = {
        'Pregnancies': 2, 'Glucose': 120, 'BloodPressure': 78, 'SkinThickness': 25,
        'Insulin': 100, 'BMI': 28.5, 'DiabetesPedigreeFunction': 0.35, 'Age': 45
    }

# Tabs for Diseases
tab_heart, tab_diabetes, tab_history = st.tabs(["🫀 Heart Disease Risk", "🩸 Diabetes Risk", "📜 Prediction History"])

# ----------------- HEART DISEASE TAB -----------------
with tab_heart:
    st.subheader("Cardiovascular Risk Evaluation")
    
    if st.button("🚀 Calculate Heart Disease Risk", key="btn_run_heart"):
        with st.spinner("Analyzing cardiovascular features with ML model..."):
            res_heart = predict_heart_disease_risk(heart_inputs)
            add_to_history(res_heart, heart_inputs)
            
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.metric("Estimated Risk Probability", f"{res_heart['risk_percentage']}%")
                if res_heart['risk_level'] == "Low Risk":
                    st.markdown('<span class="badge-low">🟢 Low Risk Category</span>', unsafe_allow_html=True)
                elif res_heart['risk_level'] == "Moderate Risk":
                    st.markdown('<span class="badge-moderate">🟠 Moderate Risk Category</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-high">🔴 High Risk Category</span>', unsafe_allow_html=True)
                    
                st.progress(res_heart['probability'])
                st.info(f"**Medical Recommendation**: {res_heart['recommendation']}")
                
            with col_m2:
                st.markdown("### 💡 Explainable AI (XAI) Insight")
                st.write(res_heart["explanation"])
                
                # Plot Feature Contributions
                df_feat = res_heart["feature_contributions"].head(6)
                fig = px.bar(
                    df_feat, x="Importance_Pct", y="Feature", orientation="h",
                    title="Top Contributing Risk Factors (%)",
                    labels={"Importance_Pct": "Influence (%)", "Feature": "Clinical Parameter"},
                    color="Importance_Pct", color_continuous_scale="Reds"
                )
                fig.update_layout(yaxis=dict(autorange="reversed"), height=300)
                st.plotly_chart(fig, use_container_width=True)

# ----------------- DIABETES TAB -----------------
with tab_diabetes:
    st.subheader("Type 2 Diabetes Risk Evaluation")
    
    if st.button("🚀 Calculate Diabetes Risk", key="btn_run_diab"):
        with st.spinner("Analyzing metabolic metrics with ML model..."):
            res_diab = predict_diabetes_risk(diabetes_inputs)
            add_to_history(res_diab, diabetes_inputs)
            
            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                st.metric("Estimated Risk Probability", f"{res_diab['risk_percentage']}%")
                if res_diab['risk_level'] == "Low Risk":
                    st.markdown('<span class="badge-low">🟢 Low Risk Category</span>', unsafe_allow_html=True)
                elif res_diab['risk_level'] == "Moderate Risk":
                    st.markdown('<span class="badge-moderate">🟠 Moderate Risk Category</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-high">🔴 High Risk Category</span>', unsafe_allow_html=True)
                    
                st.progress(res_diab['probability'])
                st.info(f"**Medical Recommendation**: {res_diab['recommendation']}")
                
            with col_d2:
                st.markdown("### 💡 Explainable AI (XAI) Insight")
                st.write(res_diab["explanation"])
                
                df_feat_d = res_diab["feature_contributions"].head(6)
                fig_d = px.bar(
                    df_feat_d, x="Importance_Pct", y="Feature", orientation="h",
                    title="Top Contributing Risk Factors (%)",
                    labels={"Importance_Pct": "Influence (%)", "Feature": "Clinical Parameter"},
                    color="Importance_Pct", color_continuous_scale="Oranges"
                )
                fig_d.update_layout(yaxis=dict(autorange="reversed"), height=300)
                st.plotly_chart(fig_d, use_container_width=True)

# ----------------- HISTORY TAB -----------------
with tab_history:
    st.subheader("📜 Session Risk Evaluation Log")
    if not st.session_state.prediction_history:
        st.write("No predictions conducted in this session yet.")
    else:
        df_hist = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_hist, use_container_width=True)
        
        if st.button("🗑️ Clear Session History"):
            clear_history()
            st.rerun()
