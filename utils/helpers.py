import streamlit as st
import pandas as pd
from datetime import datetime

MEDICAL_DISCLAIMER = (
    "⚠️ **Medical Disclaimer**: This application is an educational AI/ML project that provides health-risk "
    "estimates and general health information. It is **NOT** a medical diagnosis or treatment system. The predictions "
    "may be inaccurate. Always consult a qualified healthcare professional or medical doctor for professional advice and diagnosis."
)

def apply_custom_css():
    """Applies modern healthcare theme custom CSS styling to Streamlit app."""
    st.markdown("""
        <style>
        /* Main background and font */
        .main {
            background-color: #FAFCFF;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Header styling */
        .header-title {
            color: #1E3A8A;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        
        .header-subtitle {
            color: #4B5563;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        
        /* Custom card styling */
        .health-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border: 1px solid #E5E7EB;
            margin-bottom: 1rem;
        }
        
        /* Disclaimer callout box */
        .disclaimer-box {
            background-color: #FEF3C7;
            border-left: 5px solid #F59E0B;
            padding: 1rem;
            border-radius: 6px;
            color: #92400E;
            font-size: 0.92rem;
            margin-bottom: 1.5rem;
        }
        
        /* Risk Badges */
        .badge-low {
            background-color: #D1FAE5;
            color: #065F46;
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-weight: 600;
            display: inline-block;
        }
        
        .badge-moderate {
            background-color: #FFEDD5;
            color: #9A3412;
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-weight: 600;
            display: inline-block;
        }
        
        .badge-high {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-weight: 600;
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initializes global Streamlit session state keys."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "patient_assessment" not in st.session_state:
        st.session_state.patient_assessment = {}

def add_to_history(prediction_result, patient_inputs):
    """Adds a prediction event to session history log (no personal identifying info stored)."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disease": prediction_result["disease"],
        "risk_level": prediction_result["risk_level"],
        "risk_percentage": prediction_result["risk_percentage"],
        "key_inputs": {
            "Age": patient_inputs.get("age", patient_inputs.get("Age", "N/A")),
            "BMI": patient_inputs.get("bmi", patient_inputs.get("BMI", "N/A")),
            "Blood Pressure": patient_inputs.get("trestbps", patient_inputs.get("BloodPressure", "N/A")),
            "Glucose": patient_inputs.get("Glucose", patient_inputs.get("fbs", "N/A"))
        }
    }
    st.session_state.prediction_history.append(entry)

def clear_history():
    """Clears all session prediction history."""
    st.session_state.prediction_history = []

def display_disclaimer():
    """Renders prominent Medical Disclaimer banner."""
    st.markdown(f'<div class="disclaimer-box">{MEDICAL_DISCLAIMER}</div>', unsafe_allow_html=True)
