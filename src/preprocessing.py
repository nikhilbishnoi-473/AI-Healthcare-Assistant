import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def load_dataset(dataset_name):
    """Loads dataset by name ('heart' or 'diabetes')."""
    filename = "heart_disease.csv" if dataset_name == "heart" else "diabetes.csv"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}. Run src/download_data.py first.")
    return pd.read_csv(path)

def inspect_dataset(df):
    """Returns dataset exploration summary dictionary."""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum()),
        "describe": df.describe().to_dict()
    }
    return summary

def clean_dataset(df, dataset_name):
    """
    Cleans dataset by handling duplicate rows, invalid 0-values for medical metrics,
    and imputing missing values with median values.
    """
    df_clean = df.copy()
    
    # 1. Remove duplicate records
    df_clean = df_clean.drop_duplicates()
    
    if dataset_name == "diabetes":
        # In Pima Diabetes dataset, 0 in Glucose, BloodPressure, SkinThickness, Insulin, BMI are physically impossible
        invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in invalid_zero_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace(0, np.nan)
                # Median imputation per target class if target exists
                if 'Outcome' in df_clean.columns:
                    df_clean[col] = df_clean.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    
    elif dataset_name == "heart":
        # Ensure numerical values are bounded correctly
        if 'ca' in df_clean.columns:
            df_clean['ca'] = df_clean['ca'].fillna(df_clean['ca'].mode()[0])
        if 'thal' in df_clean.columns:
            df_clean['thal'] = df_clean['thal'].fillna(df_clean['thal'].mode()[0])
            
    return df_clean

def preprocess_and_split(df, target_col, test_size=0.2, random_state=42):
    """
    Splits data into train & test sets using stratified sampling and standardizes features.
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)
    
    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames for feature name retention
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

if __name__ == "__main__":
    print("--- TESTING DATA PREPROCESSING PIPELINE ---")
    for d_name, target in [("heart", "target"), ("diabetes", "Outcome")]:
        df_raw = load_dataset(d_name)
        info = inspect_dataset(df_raw)
        print(f"\nDataset: {d_name.upper()}")
        print(f"  Shape: {info['shape']}")
        print(f"  Duplicates: {info['duplicate_count']}")
        
        df_clean = clean_dataset(df_raw, d_name)
        X_tr, X_te, y_tr, y_te, scaler, f_names = preprocess_and_split(df_clean, target)
        print(f"  Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
        print(f"  Target Stratification - Train: {np.bincount(y_tr)}, Test: {np.bincount(y_te)}")
