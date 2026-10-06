import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
from helpers import apply_custom_css, init_session_state, display_disclaimer

st.set_page_config(page_title="Health Assessment | AI Healthcare", page_icon="📋", layout="wide")
apply_custom_css()
init_session_state()

st.title("📋 Patient Health Assessment Form")
st.write("Enter clinical parameters below to prepare data for Disease Risk Analysis.")

display_disclaimer()

with st.form("health_assessment_form"):
    st.subheader("1. General Demographic & Body Metrics")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=45)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c3:
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
    with c4:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)
        
    # Auto-calculate BMI
    bmi = round(weight / ((height / 100) ** 2), 2)
    st.info(f"💡 **Calculated Body Mass Index (BMI)**: **{bmi} kg/m²**")
    
    st.markdown("---")
    st.subheader("2. Vital Signs & Lab Results")
    v1, v2, v3, v4 = st.columns(4)
    
    with v1:
        systolic_bp = st.number_input("Systolic BP (mm Hg)", min_value=70, max_value=240, value=120)
    with v2:
        diastolic_bp = st.number_input("Diastolic BP (mm Hg)", min_value=40, max_value=140, value=80)
    with v3:
        glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=400, value=99)
    with v4:
        cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
        
    v5, v6, v7 = st.columns(3)
    with v5:
        max_heart_rate = st.number_input("Max Heart Rate (bpm)", min_value=50, max_value=220, value=150)
    with v6:
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            ["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"]
        )
    with v7:
        st_depression = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    st.markdown("---")
    st.subheader("3. Lifestyle & Medical History")
    l1, l2, l3, l4 = st.columns(4)
    
    with l1:
        pregnancies = st.number_input("Pregnancies (if female)", min_value=0, max_value=20, value=0 if gender=="Male" else 1)
    with l2:
        physical_activity = st.selectbox("Physical Activity Level", ["Low (Sedentary)", "Moderate (1-3 days/wk)", "High (4+ days/wk)"])
    with l3:
        smoking_status = st.selectbox("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker"])
    with l4:
        family_history = st.selectbox("Family History of Heart/Diabetes", ["No", "Yes"])

    symptoms = st.multiselect(
        "Current Symptoms (Check all that apply)",
        ["Chest Pain/Tightness", "Shortness of Breath", "Fatigue/Weakness", "Excessive Thirst", "Frequent Urination", "Dizziness", "None"]
    )
    
    submit_btn = st.form_submit_button("💾 Save Health Assessment & Proceed to Prediction", use_container_width=True)

if submit_btn:
    # Save formatted dictionary into session state
    sex_binary = 1 if gender == "Male" else 0
    cp_val = int(chest_pain_type.split(":")[0])
    fbs_binary = 1 if glucose > 120 else 0
    
    assessment_data = {
        "demographics": {
            "age": age, "gender": gender, "height": height, "weight": weight, "bmi": bmi
        },
        "vitals": {
            "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp, "glucose": glucose,
            "cholesterol": cholesterol, "max_heart_rate": max_heart_rate,
            "chest_pain_type": cp_val, "st_depression": st_depression
        },
        "lifestyle": {
            "pregnancies": pregnancies, "physical_activity": physical_activity,
            "smoking_status": smoking_status, "family_history": family_history,
            "symptoms": symptoms
        },
        # Model-ready dictionary for Heart Disease
        "heart_inputs": {
            "age": age, "sex": sex_binary, "cp": cp_val, "trestbps": systolic_bp,
            "chol": cholesterol, "fbs": fbs_binary, "restecg": 0, "thalach": max_heart_rate,
            "exang": 1 if "Chest Pain/Tightness" in symptoms else 0,
            "oldpeak": st_depression, "slope": 1, "ca": 0, "thal": 2
        },
        # Model-ready dictionary for Diabetes
        "diabetes_inputs": {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": diastolic_bp,
            "SkinThickness": 20, "Insulin": 80, "BMI": bmi,
            "DiabetesPedigreeFunction": 0.5 if family_history == "Yes" else 0.2, "Age": age
        }
    }
    
    st.session_state.patient_assessment = assessment_data
    st.success("✅ Patient health assessment saved successfully!")
    st.info("👈 Click on **2. Disease Prediction** in the sidebar or below to run the ML Risk Evaluation!")
    
    if st.button("🚀 Continue to Disease Risk Prediction Now"):
        st.switch_page("pages/2_Disease_Prediction.py")
