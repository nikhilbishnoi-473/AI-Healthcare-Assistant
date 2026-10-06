import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
from helpers import apply_custom_css, init_session_state, display_disclaimer

st.set_page_config(page_title="Health Calculators | AI Healthcare", page_icon="🧮", layout="wide")
apply_custom_css()
init_session_state()

st.title("🧮 Interactive Health Calculators & Reference Guide")
st.write("Educational reference utilities for calculating body mass index, target heart rate zones, and blood pressure categories.")

display_disclaimer()

calc_tab1, calc_tab2, calc_tab3 = st.tabs(["⚖️ Body Mass Index (BMI)", "💓 Heart Rate & Target Zones", "🩸 Blood Pressure Categories"])

# ----------------- BMI CALCULATOR -----------------
with calc_tab1:
    st.subheader("Body Mass Index (BMI) Calculator")
    st.write("BMI measures body fat based on height and weight applicable to adult men and women.")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        h_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5, key="bmi_h")
        w_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5, key="bmi_w")
        
        bmi_val = round(w_kg / ((h_cm / 100) ** 2), 2)
        
        if bmi_val < 18.5:
            cat = "Underweight"
            color = "blue"
        elif 18.5 <= bmi_val < 25:
            cat = "Normal Weight"
            color = "green"
        elif 25 <= bmi_val < 30:
            cat = "Overweight"
            color = "orange"
        else:
            cat = "Obese"
            color = "red"
            
        st.metric("Your Calculated BMI", f"{bmi_val} kg/m²")
        st.markdown(f"**BMI Category**: :{color}[**{cat}**]")

    with col_b2:
        st.markdown("""
        #### 📌 Standard BMI Categories (WHO Classification)
        * 🔵 **Underweight**: < 18.5 kg/m²
        * 🟢 **Normal weight**: 18.5 – 24.9 kg/m²
        * 🟠 **Overweight**: 25.0 – 29.9 kg/m²
        * 🔴 **Obesity**: 30.0 kg/m² or higher
        """)

# ----------------- HEART RATE CALCULATOR -----------------
with calc_tab2:
    st.subheader("Heart Rate & Target Zone Analyzer")
    st.write("Calculate maximum estimated heart rate and exercise training target zones.")
    
    col_hr1, col_hr2 = st.columns(2)
    with col_hr1:
        hr_age = st.number_input("Age (Years)", min_value=1, max_value=120, value=45, key="hr_age")
        resting_hr = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=150, value=72, key="r_hr")
        
        max_hr = 220 - hr_age
        target_low = int(max_hr * 0.50)
        target_high = int(max_hr * 0.85)
        
        st.metric("Estimated Maximum Heart Rate", f"{max_hr} bpm")
        st.info(f"🎯 **Target Exercise Zone (50%-85% intensity)**: **{target_low} – {target_high} bpm**")

    with col_hr2:
        st.markdown("""
        #### 💓 Resting Heart Rate Guidelines (Adults)
        * 🟢 **Athlete / High Fitness**: 40 – 60 bpm
        * 🟢 **Normal Adult Resting**: 60 – 100 bpm
        * 🟠 **Tachycardia (Elevated)**: > 100 bpm
        
        *Note: Target heart rate zones vary by physical conditioning and medical background.*
        """)

# ----------------- BLOOD PRESSURE CALCULATOR -----------------
with calc_tab3:
    st.subheader("Blood Pressure Stage Classifier")
    st.write("Check American Heart Association (AHA) blood pressure categories based on systolic and diastolic readings.")
    
    col_bp1, col_bp2 = st.columns(2)
    with col_bp1:
        sys_val = st.number_input("Systolic Pressure (mm Hg)", min_value=70, max_value=250, value=120, key="sys_bp")
        dia_val = st.number_input("Diastolic Pressure (mm Hg)", min_value=40, max_value=150, value=80, key="dia_bp")
        
        if sys_val < 120 and dia_val < 80:
            bp_cat = "Normal Blood Pressure"
            bp_col = "green"
        elif 120 <= sys_val <= 129 and dia_val < 80:
            bp_cat = "Elevated Blood Pressure"
            bp_col = "orange"
        elif (130 <= sys_val <= 139) or (80 <= dia_val <= 89):
            bp_cat = "Stage 1 Hypertension"
            bp_col = "orange"
        elif (140 <= sys_val < 180) or (90 <= dia_val < 120):
            bp_cat = "Stage 2 Hypertension"
            bp_col = "red"
        else:
            bp_cat = "Hypertensive Crisis (Seek Immediate Medical Care!)"
            bp_col = "red"
            
        st.markdown(f"### Result: :{bp_col}[**{bp_cat}**]")

    with col_bp2:
        st.markdown("""
        #### 🩸 AHA Blood Pressure Category Reference
        * 🟢 **Normal**: Systolic < 120 AND Diastolic < 80
        * 🟡 **Elevated**: Systolic 120–129 AND Diastolic < 80
        * 🟠 **Stage 1 Hypertension**: Systolic 130–139 OR Diastolic 80–89
        * 🔴 **Stage 2 Hypertension**: Systolic 140+ OR Diastolic 90+
        * 🚨 **Hypertensive Crisis**: Systolic > 180 AND/OR Diastolic > 120
        """)
