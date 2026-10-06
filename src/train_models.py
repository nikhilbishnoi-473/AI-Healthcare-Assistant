import os
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Try importing XGBoost; fall back to Scikit-Learn GradientBoosting if OpenMP missing on macOS
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as e:
    HAS_XGBOOST = False

from preprocessing import load_dataset, clean_dataset, preprocess_and_split
from evaluation import evaluate_model, compare_models

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_disease_models(dataset_name, target_col):
    """
    Loads, cleans, preprocesses dataset, trains candidate classifiers, evaluates them,
    selects best model based on Recall/Sensitivity, and saves model artifact.
    """
    print(f"\n==================================================")
    print(f" TRAINING MODELS FOR: {dataset_name.upper()} DISEASE PREDICTION ")
    print(f"==================================================")
    
    # Step 1: Load dataset
    df_raw = load_dataset(dataset_name)
    print(f"1. Dataset loaded. Initial shape: {df_raw.shape}")
    
    # Step 2: Clean dataset
    df_clean = clean_dataset(df_raw, dataset_name)
    print(f"2. Dataset cleaned. Post-cleaning shape: {df_clean.shape}")
    
    # Step 3: Preprocess and Stratified Split
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_and_split(
        df_clean, target_col=target_col, test_size=0.2, random_state=42
    )
    print(f"3. Stratified split complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # Step 4: Define candidate models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6),
    }
    
    if HAS_XGBOOST:
        models["XGBoost Classifier"] = XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=100, max_depth=4)
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=4)
    
    evaluations = []
    trained_models = {}
    
    # Step 5: Train and evaluate each model
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        ev = evaluate_model(model, X_test, y_test, model_name=name)
        evaluations.append(ev)
        trained_models[name] = model
        
    # Step 6: Compare models
    comparison_df = compare_models(evaluations)
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(comparison_df.to_string(index=False))
    
    # Step 7: Select best model (sorted primarily by Recall/Sensitivity for medical safety)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_eval = [ev for ev in evaluations if ev["Model"] == best_model_name][0]
    
    print(f"\n🏆 BEST MODEL SELECTED: '{best_model_name}'")
    print(f"   Recall (Sensitivity): {best_eval['Recall']}")
    print(f"   Accuracy: {best_eval['Accuracy']}, ROC-AUC: {best_eval['ROC-AUC']}")
    
    # Step 8: Package artifact and save using Joblib
    save_filename = f"{dataset_name}_disease_model.pkl" if dataset_name == "heart" else "diabetes_model.pkl"
    save_path = os.path.join(MODELS_DIR, save_filename)
    
    artifact = {
        "dataset_name": dataset_name,
        "target_col": target_col,
        "model_name": best_model_name,
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "evaluation": best_eval,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "all_evaluations": evaluations
    }
    
    joblib.dump(artifact, save_path)
    print(f"✅ Saved trained model artifact to {save_path}")
    
    return artifact

def run_all_training():
    train_disease_models("heart", target_col="target")
    train_disease_models("diabetes", target_col="Outcome")

if __name__ == "__main__":
    run_all_training()
