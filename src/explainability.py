import numpy as np
import pandas as pd

def get_feature_importances(model, feature_names):
    """
    Extracts global feature importances from a trained model.
    Supports Random Forest, Decision Tree, XGBoost, and Logistic Regression (coefficients).
    """
    importances = None
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        
    if importances is not None:
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        # Normalize to percentage
        total = df_imp["Importance"].sum()
        if total > 0:
            df_imp["Importance_Pct"] = (df_imp["Importance"] / total) * 100
        else:
            df_imp["Importance_Pct"] = 100.0 / len(feature_names)
        return df_imp
    else:
        # Equal importance fallback
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": [1.0 / len(feature_names)] * len(feature_names),
            "Importance_Pct": [100.0 / len(feature_names)] * len(feature_names)
        })

def explain_single_prediction(model, input_series, feature_names, scaler=None):
    """
    Generates a beginner-friendly explanation for a single prediction instance.
    Identifies top contributing risk features and generates natural language narrative.
    """
    # Convert input to DataFrame
    if isinstance(input_series, dict):
        input_df = pd.DataFrame([input_series])
    elif isinstance(input_series, pd.Series):
        input_df = pd.DataFrame([input_series.to_dict()])
    else:
        input_df = pd.DataFrame(input_series, columns=feature_names)
        
    # Scale input if scaler provided
    if scaler is not None:
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
    else:
        input_scaled = input_df
        
    global_imp = get_feature_importances(model, feature_names)
    
    # Calculate feature contribution score (scaled feature value * global importance)
    contributions = []
    for _, row in global_imp.iterrows():
        feat = row["Feature"]
        val = input_df[feat].values[0]
        imp_pct = row["Importance_Pct"]
        contributions.append({
            "Feature": feat,
            "Value": val,
            "Importance_Pct": round(imp_pct, 2),
            "Impact_Score": round(imp_pct * (abs(val) + 0.1), 2)
        })
        
    df_contrib = pd.DataFrame(contributions).sort_values(by="Importance_Pct", ascending=False)
    
    top_3 = df_contrib.head(3)["Feature"].tolist()
    top_3_pct = df_contrib.head(3)["Importance_Pct"].tolist()
    
    explanation_text = (
        f"Among the entered values, **{top_3[0].replace('_', ' ').title()}** (contributing ~{top_3_pct[0]:.1f}%) "
        f"and **{top_3[1].replace('_', ' ').title()}** (contributing ~{top_3_pct[1]:.1f}%) "
        f"had the strongest influence on the model's risk estimation."
    )
    
    return {
        "top_features": top_3,
        "feature_contributions": df_contrib,
        "explanation_text": explanation_text
    }

def get_shap_explanation(model, X_sample):
    """
    Computes SHAP values if shap library is available.
    """
    try:
        import shap
        if hasattr(model, "predict_proba") and "XGB" in type(model).__name__ or "Forest" in type(model).__name__ or "Tree" in type(model).__name__:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            return explainer, shap_values
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            return explainer, shap_values
    except Exception as e:
        return None, None
