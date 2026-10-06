# 🏥 AI Healthcare Assistant – Disease Risk Prediction & Health Analysis System

A modern, full-stack Machine Learning web application built with **Python, Scikit-learn, XGBoost, Streamlit, Plotly, and Joblib** to evaluate cardiovascular and metabolic disease risk probabilities using clinical health parameters.

> **⚠️ Medical Disclaimer**: This system is strictly an **educational risk-prediction tool** for college projects, research demonstration, and portfolio showcases. It is **NOT** a clinical diagnosis system. Always consult a certified medical professional for diagnosis or advice.

---

## 📌 1. Project Title
**AI Healthcare Assistant – Disease Risk Prediction & Health Analysis System**

---

## 📝 2. Project Overview
Chronic illnesses such as **Heart Disease** and **Type 2 Diabetes** represent leading causes of global mortality. Early detection through pattern recognition in physical vitals and lab results can encourage timely lifestyle interventions. 

This project provides an interactive multi-page web platform where users can assess physical health metrics, calculate standard health indicators (BMI, BP stages, target heart rates), query an AI health assistant, and inspect Machine Learning model predictions with **Explainable AI (XAI)** feature importances.

---

## 🎯 3. Problem Statement
Many individuals lack accessible tools to understand how their daily physical health metrics (such as fasting glucose, blood pressure, cholesterol, and BMI) contribute to overall disease risk. Existing tools either present complex statistical raw outputs or lack transparency into *why* a particular risk estimate was generated.

---

## 🚩 4. Key Objectives
1. Build a complete, modular, end-to-end Python & Streamlit healthcare application.
2. Train and benchmark 4 Machine Learning classifiers (**Logistic Regression, Decision Tree, Random Forest, XGBoost/Gradient Boosting**) on benchmark datasets.
3. Optimize model selection with an emphasis on **Recall (Sensitivity)** to minimize dangerous false negative predictions.
4. Provide transparent **Explainable AI (XAI)** insights explaining which clinical features drove the risk estimate.
5. Provide educational health utilities (BMI, Blood Pressure, Heart Rate calculators) and an interactive AI Healthcare Chatbot with offline fallback support.

---

## ✨ 5. Core Features
* 📋 **Patient Health Assessment**: Form validating age, height, weight, BP, glucose, cholesterol, max heart rate, and symptoms.
* 🫀 **Heart Disease Risk Estimator**: Benchmark risk evaluation based on UCI Cleveland clinical data.
* 🩸 **Diabetes Risk Estimator**: Metabolic risk evaluation based on Pima Indians Diabetes data.
* 💡 **Explainable AI (XAI)**: Visual and narrative breakdown of feature contributions for each prediction.
* 🧮 **Interactive Health Calculators**: BMI category calculator, AHA Blood Pressure stage classifier, and target exercise heart rate range analyzer.
* 💬 **AI Health Assistant Chatbot**: Conversational AI assistant (Gemini/LLM API integration with offline FAQ fallback system).
* 📊 **Analytics Dashboard**: Interactive Plotly charts for model benchmarks, confusion matrices, ROC curves, and dataset EDA.
* 📜 **Session Prediction History**: Logged prediction events with privacy preservation (no personal identifying info stored) and one-click history clearing.

---

## 🛠️ 6. Technologies Used
* **Programming Language**: Python 3.14+
* **Machine Learning**: Scikit-Learn, XGBoost, Joblib
* **Data Processing**: Pandas, NumPy
* **Visualization**: Plotly Express, Matplotlib, Seaborn
* **Web Framework**: Streamlit
* **Explainability**: Custom XAI Feature Contribution Engine & SHAP
* **Environment & Config**: Python-Dotenv

---

## 🏗️ 7. System Architecture & Folder Structure
```text
AI-Healthcare-Assistant/
├── app.py                          # Streamlit Main Dashboard & Home Page
├── requirements.txt                # Python Dependencies
├── README.md                       # Comprehensive Documentation
├── .gitignore                      # Git Ignore Rules
├── .env.example                    # Environment Template for API Keys
│
├── data/
│   ├── README.md                   # Dataset Documentation & Provenance
│   ├── heart_disease.csv           # UCI Heart Disease CSV
│   └── diabetes.csv                # Pima Indians Diabetes CSV
│
├── models/
│   ├── README.md                   # Model Artifact Documentation
│   ├── heart_disease_model.pkl     # Best Trained Heart Model & Scaler
│   └── diabetes_model.pkl          # Best Trained Diabetes Model & Scaler
│
├── notebooks/
│   ├── heart_disease_analysis.ipynb # Jupyter Analysis Notebook
│   └── diabetes_analysis.ipynb      # Jupyter Analysis Notebook
│
├── src/
│   ├── download_data.py            # Automated Dataset Downloader
│   ├── preprocessing.py           # Inspection, Zero Imputation, Scaling & Stratified Split
│   ├── train_models.py            # Model Training & Model Selection Pipeline
│   ├── prediction.py             # Inference Engine & Input Validation
│   ├── evaluation.py             # Accuracy, Recall, Precision, F1 & ROC-AUC Metrics
│   └── explainability.py         # Feature Importance XAI Engine
│
├── pages/
│   ├── 1_Health_Assessment.py     # Patient Input Form Page
│   ├── 2_Disease_Prediction.py    # Risk Prediction & XAI Page
│   ├── 3_Health_Calculators.py    # Educational Calculators Page
│   ├── 4_AI_Health_Assistant.py   # AI Healthcare Chatbot
│   └── 5_Model_Performance.py    # Model Benchmark & EDA Dashboard
│
└── utils/
    └── helpers.py                 # UI Styling, Disclaimers & Session History
```

---

## 🔄 8. ML Workflow
```
[ Raw Public Datasets ] ──> [ Data Cleaning & Zero Imputation ] ──> [ Stratified Train-Test Split (80/20) ]
                                                                                  │
[ Saved Joblib Artifacts ] <── [ Model Selection (Recall Priority) ] <── [ Model Training & Evaluation (4 Algorithms) ]
```

---

## 📊 9. Dataset Information
1. **UCI Heart Disease Dataset (Cleveland)**: 303 rows, 13 clinical features (Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, ECG, Max HR, ST Depression, etc.) + `target` (0 = No Disease, 1 = Disease Risk).
2. **Pima Indians Diabetes Dataset**: 768 rows, 8 clinical features (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age) + `Outcome` (0 = No Diabetes, 1 = Diabetes Risk).

---

## 🔍 10. Exploratory Data Analysis (EDA)
- **Zero-Value Treatment**: In the Diabetes dataset, biologically impossible zeros in Glucose, BP, BMI, Insulin, and Skin Thickness were treated as missing values and imputed using target-stratified medians.
- **Class Balance**: Target stratification ensured balanced distribution across train (80%) and test (20%) sets.

---

## 🤖 11. Trained Models & Benchmark Comparisons
Four algorithms were trained and evaluated per disease:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost / Gradient Boosting Classifier**

---

## 📈 12. Model Evaluation Metrics
Model selection prioritizes **Recall (Sensitivity)**:
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
*In medical risk screening, missing a high-risk patient (False Negative) carries far greater severity than a false alarm (False Positive).*

---

## 💡 13. Explainable AI (XAI)
The system extracts feature importances and computes percentage contributions for individual patient inputs. For instance:
> *"Among the entered values, **Glucose** (contributing ~24.6%) and **Age** (contributing ~18.0%) had the strongest influence on the model's risk estimation."*

---

## 🖼️ 14. Screenshots Section
*(Add screenshots of your running Streamlit application here for your GitHub portfolio!)*

---

## 💻 15. Installation & Setup Instructions

### Step 1: Clone the Repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Healthcare-Assistant
```

### Step 2: Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Datasets & Train Models
```bash
python3 src/download_data.py
python3 src/train_models.py
```

### Step 5: Launch the Streamlit Application
```bash
streamlit run app.py
```

---

## 🚀 16. How to Upload to GitHub
```bash
git init
git add .
git commit -m "Initial complete healthcare AI project release"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

---

## ⚠️ 17. Limitations
* Model training is based on public benchmark datasets with specific demographic distributions.
* Predictions are probabilistic risk estimates and cannot replace clinical lab tests (e.g. HbA1c, Coronary Angiography).

---

## 🔮 18. Future Improvements
* Integrate multi-disease risk profiling (e.g., Kidney Disease, Stroke Risk).
* Integrate wearable device API connections (e.g., Apple HealthKit / Google Fit).
* Deploy to Streamlit Community Cloud or AWS/GCP with automated CI/CD pipelines.

---

## ⚕️ 19. Medical Disclaimer
> **⚠️ Medical Disclaimer**: This application is an educational AI/ML project that provides health-risk estimates and general health information. It is not a medical diagnosis or treatment system. The predictions may be incorrect. Always consult a qualified healthcare professional for medical advice.
