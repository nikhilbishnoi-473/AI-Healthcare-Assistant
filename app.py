import streamlit as st
import sys
import os

# Add src and utils to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from helpers import apply_custom_css, init_session_state, display_disclaimer

st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
init_session_state()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-heart.png", width=70)
    st.title("AI Healthcare Assistant")
    st.caption("Disease Risk Prediction & Health Analysis")
    st.markdown("---")
    
    st.subheader("📊 Session Statistics")
    history_count = len(st.session_state.prediction_history)
    st.metric("Total Assessments Conducted", history_count)
    
    st.markdown("---")
    st.markdown("### 📌 Navigation Quick Links")
    st.markdown("Use the left menu to navigate between pages:")
    st.markdown("- **1. Health Assessment**")
    st.markdown("- **2. Disease Prediction**")
    st.markdown("- **3. Health Calculators**")
    st.markdown("- **4. AI Health Assistant**")
    st.markdown("- **5. Model Performance**")

# Main Header
st.markdown('<h1 class="header-title">🏥 AI Healthcare Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">An Educational Machine Learning System for Disease Risk Prediction & Health Analytics</p>', unsafe_allow_html=True)

display_disclaimer()

# App Overview Grid
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💡 Welcome & Project Overview")
    st.write("""
    The **AI Healthcare Assistant** is an intelligent web application designed for AIML students, healthcare researchers, and medical enthusiasts. 
    It combines trained **Machine Learning models (Scikit-Learn, Random Forest, Logistic Regression, XGBoost)** with interactive visual analytics to estimate risk probabilities for chronic diseases based on clinical health parameters.
    """)
    
    st.markdown("### 🚀 Core Platform Features")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("""
        * 📋 **Comprehensive Health Assessment**: Standardized health metrics form.
        * 🫀 **Heart Disease Risk Estimation**: Trained on Cleveland UCI benchmarks.
        * 🩸 **Diabetes Risk Estimation**: Trained on Pima Indians clinical data.
        """)
    with f_col2:
        st.markdown("""
        * 🔍 **Explainable AI (XAI)**: Feature contribution breakdown for full transparency.
        * 🧮 **Educational Calculators**: BMI, Blood Pressure, and Heart Rate analyzers.
        * 💬 **AI Health Chatbot**: 24/7 educational health assistant with API/rule fallback.
        """)

with col2:
    st.subheader("🤖 Machine Learning Architecture")
    st.info("""
    **Models Benchmarked & Evaluated:**
    * 🔹 **Logistic Regression** (Baseline & Linear Decision Boundary)
    * 🔹 **Decision Trees** (Interpretable Rules)
    * 🔹 **Random Forest Classifier** (Ensemble Bagging)
    * 🔹 **XGBoost / Gradient Boosting** (Ensemble Boosting)
    
    **Optimization Metric:**  
    Models are evaluated primarily on **Recall (Sensitivity)** to minimize false negative risk in medical estimates.
    """)

st.markdown("---")

# Quick Action Cards
st.subheader("🛠️ Quick Access Modules")
q1, q2, q3, q4 = st.columns(4)

with q1:
    st.markdown("""
    #### 📋 Assessment
    Enter your physical metrics and clinical parameters.
    """)
    if st.button("Go to Assessment", key="btn_assess"):
        st.switch_page("pages/1_Health_Assessment.py")

with q2:
    st.markdown("""
    #### 🔮 Prediction
    Run ML disease risk models and inspect XAI explanations.
    """)
    if st.button("Go to Prediction", key="btn_pred"):
        st.switch_page("pages/2_Disease_Prediction.py")

with q3:
    st.markdown("""
    #### 🧮 Calculators
    Compute BMI, Blood Pressure stage, and target heart rates.
    """)
    if st.button("Go to Calculators", key="btn_calc"):
        st.switch_page("pages/3_Health_Calculators.py")

with q4:
    st.markdown("""
    #### 💬 AI Chatbot
    Ask educational questions about symptoms and health habits.
    """)
    if st.button("Chat with AI", key="btn_chat"):
        st.switch_page("pages/4_AI_Health_Assistant.py")

st.markdown("---")
st.caption("Developed with ❤️ for AIML College Project & Portfolio | Powered by Streamlit, Scikit-Learn & Plotly")
