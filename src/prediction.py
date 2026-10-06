import os
import joblib
import pandas as pd
import numpy as np
from explainability import explain_single_prediction

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_LOADED_MODELS = {}

def load_model_artifact(dataset_name):
    """
    Loads saved model artifact from models/ directory.
    Uses caching to avoid reloading disk files repeatedly.
    """
    filename = f"{dataset_name}_disease_model.pkl" if dataset_name == "heart" else "diabetes_model.pkl"
    path = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Run src/train_models.py first.")
        
    if dataset_name not in _LOADED_MODELS:
        _LOADED_MODELS[dataset_name] = joblib.load(path)
        
    return _LOADED_MODELS[dataset_name]

def predict_heart_disease_risk(user_inputs):
    """
    Predicts heart disease risk probability from user input dict.
    Expected keys: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    """
    artifact = load_model_artifact("heart")
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    
    # Input validation and formatting
    input_dict = {}
    for feat in feature_names:
        val = user_inputs.get(feat, 0)
        input_dict[feat] = float(val)
        
    df_input = pd.DataFrame([input_dict], columns=feature_names)
    scaled_array = scaler.transform(df_input)
    scaled_input = pd.DataFrame(scaled_array, columns=feature_names)
    
    # Get probability
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(scaled_input)[0][1])
    else:
        prob = float(model.predict(scaled_input)[0])
        
    # Categorize Risk Level
    if prob < 0.35:
        risk_level = "Low Risk"
        risk_color = "green"
        recommendation = "Maintain your current healthy lifestyle habits with regular physical activity and annual health checkups."
    elif prob < 0.65:
        risk_level = "Moderate Risk"
        risk_color = "orange"
        recommendation = "Consider reviewing your physical activity, diet, and blood pressure with your primary healthcare provider."
    else:
        risk_level = "High Risk"
        risk_color = "red"
        recommendation = "We recommend consulting a medical doctor or cardiologist for a comprehensive cardiovascular assessment."
        
    # Explainability
    xai = explain_single_prediction(model, df_input, feature_names, scaler=scaler)
    
    return {
        "disease": "Heart Disease",
        "probability": prob,
        "risk_percentage": round(prob * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "disclaimer": "⚠️ The model estimates a risk probability based on provided data. This is an educational tool, NOT a medical diagnosis.",
        "explanation": xai["explanation_text"],
        "top_features": xai["top_features"],
        "feature_contributions": xai["feature_contributions"]
    }

def predict_diabetes_risk(user_inputs):
    """
    Predicts diabetes risk probability from user input dict.
    Expected keys: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    """
    artifact = load_model_artifact("diabetes")
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    
    input_dict = {}
    for feat in feature_names:
        val = user_inputs.get(feat, 0)
        input_dict[feat] = float(val)
        
    df_input = pd.DataFrame([input_dict], columns=feature_names)
    scaled_array = scaler.transform(df_input)
    scaled_input = pd.DataFrame(scaled_array, columns=feature_names)
    
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(scaled_input)[0][1])
    else:
        prob = float(model.predict(scaled_input)[0])
        
    if prob < 0.35:
        risk_level = "Low Risk"
        risk_color = "green"
        recommendation = "Keep up a balanced nutrition plan, stay active, and get periodic fasting glucose screenings."
    elif prob < 0.65:
        risk_level = "Moderate Risk"
        risk_color = "orange"
        recommendation = "Pay attention to daily carbohydrate intake, physical activity, and discuss fasting blood sugar with a doctor."
    else:
        risk_level = "High Risk"
        risk_color = "red"
        recommendation = "We advise consulting a qualified physician or endocrinologist to perform HbA1c and oral glucose testing."
        
    xai = explain_single_prediction(model, df_input, feature_names, scaler=scaler)
    
    return {
        "disease": "Diabetes",
        "probability": prob,
        "risk_percentage": round(prob * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "disclaimer": "⚠️ The model estimates a risk probability based on provided data. This is an educational tool, NOT a medical diagnosis.",
        "explanation": xai["explanation_text"],
        "top_features": xai["top_features"],
        "feature_contributions": xai["feature_contributions"]
    }

if __name__ == "__main__":
    print("--- TESTING PREDICTION PIPELINE ---")
    sample_heart = {
        'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233, 'fbs': 1,
        'restecg': 0, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    }
    res_h = predict_heart_disease_risk(sample_heart)
    print("Heart Prediction Result:", res_h["risk_level"], f"({res_h['risk_percentage']}%)")
    print("  Explanation:", res_h["explanation"])
    
    sample_diab = {
        'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72, 'SkinThickness': 35,
        'Insulin': 0, 'BMI': 33.6, 'DiabetesPedigreeFunction': 0.627, 'Age': 50
    }
    res_d = predict_diabetes_risk(sample_diab)
    print("Diabetes Prediction Result:", res_d["risk_level"], f"({res_d['risk_percentage']}%)")
    print("  Explanation:", res_d["explanation"])
