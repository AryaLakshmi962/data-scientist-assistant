"""
data_scientist_assistant.py

A compact pipeline used by the Flask app. It performs:
- Load CSV
- Basic cleaning (drop high-missing cols)
- Datetime parsing & simple feature engineering
- Imputation and encoding
- Trains a simple model if target detected
- Saves cleaned CSV and JSON report into the output directory
"""

import os
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def run_pipeline(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)
    report = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_types": df.dtypes.astype(str).to_dict(),
        "missing_counts": df.isnull().sum().to_dict()
    }

    # Drop columns with >40% missing
    missing_pct = df.isnull().mean() * 100
    drop_cols = missing_pct[missing_pct > 40].index.tolist()
    report["dropped_columns"] = drop_cols
    df = df.drop(columns=drop_cols)

    # Parse datetime-like columns & add features
    datetime_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if parsed.notna().sum() / max(1, len(parsed)) > 0.5:
                    df[col] = parsed
                    datetime_cols.append(col)
            except Exception:
                continue

    report["datetime_columns"] = datetime_cols

    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_hour"] = df[col].dt.hour
        df[f"{col}_weekday"] = df[col].dt.weekday

    # Detect target column heuristically
    target = None
    for c in df.columns:
        if c.lower() in ("target", "label", "severity"):
            target = c
            break

    if target is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in reversed(numeric):
            nunique = int(df[c].nunique(dropna=True))
            if 2 <= nunique <= 20:
                target = c
                break

    report["detected_target"] = target

    # Separate features and target
    if target:
        X = df.drop(columns=[target])
        y = df[target]
    else:
        X = df.copy()
        y = None

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Transformers
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ], remainder="drop")

    # Transform data
    try:
        X_trans = preprocessor.fit_transform(X)
    except Exception:
        X_trans = preprocessor.fit_transform(X.fillna("Missing"))

    # Create cleaned DataFrame
    feature_names = num_cols.copy()
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_names)
    except Exception:
        pass

    X_clean = pd.DataFrame(X_trans, columns=feature_names, index=X.index)

    # Save cleaned dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_path = os.path.join(output_dir, f"cleaned_{timestamp}.csv")
    X_clean.to_csv(cleaned_path, index=False)
    report["cleaned_path"] = cleaned_path

    # Train simple model if target present
    model_metrics = {}
    if y is not None:
        try:
            is_class = (y.dtype == "object") or (y.nunique() <= 20)
            model = RandomForestClassifier(n_estimators=50, random_state=42) if is_class else RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_clean.fillna(0), y)

            try:
                importances = model.feature_importances_
                fi = sorted(zip(X_clean.columns.tolist(), importances), key=lambda x: x[1], reverse=True)[:20]
                model_metrics["feature_importances"] = [{"feature": f, "importance": float(im)} for f, im in fi]
            except Exception:
                model_metrics["feature_importances"] = []

            model_path = os.path.join(output_dir, f"model_{timestamp}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            report["model_path"] = model_path
        except Exception as e:
            report["model_error"] = str(e)

    report["model_metrics"] = model_metrics

    # Save report JSON
    report_path = os.path.join(output_dir, f"report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report
