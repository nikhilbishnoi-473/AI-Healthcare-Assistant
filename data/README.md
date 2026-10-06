# Datasets Directory

This directory stores the datasets used for training disease risk prediction models.

## 1. Heart Disease Dataset (`heart_disease.csv`)
* **Source**: UCI Machine Learning Repository / Cleveland Heart Disease Dataset
* **Samples**: 303 patient records
* **Features (13 clinical inputs)**:
  * `age`: Age in years
  * `sex`: Gender (1 = male, 0 = female)
  * `cp`: Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
  * `trestbps`: Resting blood pressure in mm Hg
  * `chol`: Serum cholesterol in mg/dl
  * `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
  * `restecg`: Resting ECG results (0, 1, 2)
  * `thalach`: Maximum heart rate achieved
  * `exang`: Exercise induced angina (1 = yes, 0 = no)
  * `oldpeak`: ST depression induced by exercise relative to rest
  * `slope`: Slope of the peak exercise ST segment
  * `ca`: Number of major vessels (0-3) colored by flourosopy
  * `thal`: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
* **Target**: `target` (0 = No Heart Disease, 1 = Heart Disease Risk)

---

## 2. Diabetes Dataset (`diabetes.csv`)
* **Source**: National Institute of Diabetes and Digestive and Kidney Diseases / Pima Indians Diabetes Dataset
* **Samples**: 768 patient records
* **Features (8 clinical inputs)**:
  * `Pregnancies`: Number of times pregnant
  * `Glucose`: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
  * `BloodPressure`: Diastolic blood pressure (mm Hg)
  * `SkinThickness`: Triceps skin fold thickness (mm)
  * `Insulin`: 2-Hour serum insulin (mu U/ml)
  * `BMI`: Body mass index (weight in kg / (height in m)^2)
  * `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic score)
  * `Age`: Age in years
* **Target**: `Outcome` (0 = No Diabetes, 1 = Diabetes Risk)
