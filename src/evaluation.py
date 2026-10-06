import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained classifier on test data.
    Returns a dictionary of metrics, confusion matrix, and ROC curve coordinates.
    """
    y_pred = model.predict(X_test)
    
    # Predict probabilities if supported by the model
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred.astype(float)
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    except Exception:
        auc = 0.5
        fpr, tpr, thresholds = [0, 1], [0, 1], [0, 1]
        
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1-Score": round(float(f1), 4),
        "ROC-AUC": round(float(auc), 4),
        "Confusion_Matrix": cm,
        "ROC_Curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    }
    
    return metrics

def compare_models(evaluations_list):
    """
    Compares multiple model evaluation results.
    Returns a pandas DataFrame sorted by Recall (highest sensitivity first) and F1-score.
    """
    df_compare = pd.DataFrame([{
        "Model": ev["Model"],
        "Accuracy": ev["Accuracy"],
        "Precision": ev["Precision"],
        "Recall (Sensitivity)": ev["Recall"],
        "F1-Score": ev["F1-Score"],
        "ROC-AUC": ev["ROC-AUC"]
    } for ev in evaluations_list])
    
    # Sort by Recall primarily (medical priority) and F1-Score secondarily
    df_compare = df_compare.sort_values(by=["Recall (Sensitivity)", "F1-Score", "ROC-AUC"], ascending=False).reset_index(drop=True)
    return df_compare

if __name__ == "__main__":
    print("--- TESTING MODEL EVALUATION MODULE ---")
    from sklearn.dummy import DummyClassifier
    X_dummy = np.random.randn(100, 5)
    y_dummy = np.random.choice([0, 1], 100)
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_dummy, y_dummy)
    res = evaluate_model(clf, X_dummy, y_dummy, "DummyBaseline")
    print("Dummy Evaluation Metrics:", {k: v for k, v in res.items() if k not in ["Confusion_Matrix", "ROC_Curve"]})
