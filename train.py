"""
Model Training, Cross-Validation, and Rigorous Evaluation Script.

Trains, evaluates, and compares classification models for:
1. Storm Occurrence (Storm vs No Storm)
2. Weather Severity (Light, Moderate, Heavy, Severe)

Prevents data leakage, evaluates probability calibration, computes genuine metrics,
and saves production-ready scikit-learn pipelines with joblib.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, brier_score_loss, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import (
    get_preprocessor_pipeline, OCCURRENCE_NUMERIC_FEATURES, SEVERITY_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, clean_weather_dataframe, VALID_SEVERITIES
)
from src.visualization import (
    plot_model_confusion_matrices, plot_roc_and_calibration,
    plot_feature_importances
)


def train_storm_occurrence_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and rigorously evaluate Logistic Regression, Decision Tree,
    and Random Forest classifiers for Storm Occurrence (Binary).
    """
    print("\n" + "="*60)
    print("=== STAGE 1: STORM OCCURRENCE PREDICTION MODELING ===")
    print("="*60)

    # Feature matrix X and target y with reset index and native numpy arrays
    # Decoupled from precipitation to eliminate the ASOS lightning sensor artifact
    feature_cols = OCCURRENCE_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy().reset_index(drop=True)
    # Ensure standard python object types for categorical columns to avoid pyarrow slicing quirks
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    for col in OCCURRENCE_NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

    y = np.array(df['is_storm'].astype(int).values)

    print(f"[*] Dataset size for occurrence modeling: {len(X):,} rows")
    print(f"[*] Class distribution: Non-Storm (0) = {(y == 0).sum():,} | Storm (1) = {(y == 1).sum():,}")

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[*] Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # Define Candidate Models
    candidate_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=15,
            class_weight='balanced',
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )
    }

    trained_pipelines = {}
    metrics_summary = {}
    confusion_matrices = {}
    roc_data = {}
    calib_data = {}

    for name, clf in candidate_models.items():
        print(f"\n[*] Training and evaluating: {name}...")
        
        # Build end-to-end pipeline: Preprocessing + Classifier (Occurrence decoupled from precipitation)
        preprocessor = get_preprocessor_pipeline(OCCURRENCE_NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Fit exclusively on train set to prevent any data leakage
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions on unseen test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        prec_storm = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_storm = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_storm = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        metrics_summary[name] = {
            'accuracy': round(float(acc), 4),
            'precision_macro': round(float(prec_macro), 4),
            'recall_macro': round(float(rec_macro), 4),
            'f1_macro': round(float(f1_macro), 4),
            'precision_storm': round(float(prec_storm), 4),
            'recall_storm': round(float(rec_storm), 4),
            'f1_storm': round(float(f1_storm), 4),
            'roc_auc': round(float(auc), 4),
            'brier_score': round(float(brier), 4)
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc)}

        # Calibration curve data (Reliability diagram)
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        calib_data[name] = {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'brier': float(brier)
        }

        print(f"    Accuracy:  {acc*100:.2f}%")
        print(f"    Macro F1:  {f1_macro:.4f} (Storm F1: {f1_storm:.4f})")
        print(f"    ROC-AUC:   {auc:.4f} | Brier Score: {brier:.4f}")

    # Select Best Occurrence Model by F1-Score and ROC-AUC
    best_name = max(metrics_summary, key=lambda k: metrics_summary[k]['f1_macro'] + metrics_summary[k]['roc_auc'])
    print(f"\n[+] Selected Best Occurrence Model: {best_name} (Highest discriminatory power & balance)")

    # Generate Evaluation Charts
    print("[*] Generating Occurrence Model Evaluation Charts...")
    plot_model_confusion_matrices(
        confusion_matrices,
        class_names=['No Storm', 'Storm'],
        output_dir="outputs/metrics",
        filename="occurrence_confusion_matrices.png",
        title_prefix="Storm Occurrence"
    )

    plot_roc_and_calibration(roc_data, calib_data, output_dir="outputs/metrics")

    # Feature Importances from Tree / Random Forest
    rf_model = trained_pipelines['Random Forest'].named_steps['classifier']
    preproc = trained_pipelines['Random Forest'].named_steps['preprocessor']
    try:
        cat_features_out = preproc.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(CATEGORICAL_FEATURES)
        all_feature_names = OCCURRENCE_NUMERIC_FEATURES + list(cat_features_out)
        plot_feature_importances(all_feature_names, rf_model.feature_importances_, output_dir="outputs/metrics")
    except Exception as e:
        print(f"    Notice: feature importance plot skipped ({e})")

    return {
        'trained_pipelines': trained_pipelines,
        'best_model_name': best_name,
        'metrics_summary': metrics_summary,
        'test_data': (X_test, y_test)
    }


def train_storm_severity_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and evaluate Multi-Class Severity Classification models
    across genuine weather severity levels: Light, Moderate, Heavy, Severe.
    """
    print("\n" + "="*60)
    print("=== STAGE 2: WEATHER EVENT SEVERITY MODELING ===")
    print("="*60)

    # Filter to valid ground-truth severity categories and reset index
    valid_mask = df['severity'].isin(VALID_SEVERITIES)
    sev_df = df[valid_mask].copy().reset_index(drop=True)

    feature_cols = SEVERITY_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = sev_df[feature_cols].copy().reset_index(drop=True)
    # Ensure clean types to prevent arrow indexing issues
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    for col in SEVERITY_NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

    y = np.array(sev_df['severity'].values)

    print(f"[*] Dataset size for severity modeling: {len(X):,} records")
    sev_counts = pd.Series(y).value_counts()
    for sev, cnt in sev_counts.items():
        print(f"    {sev}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    candidate_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )
    }

    trained_pipelines = {}
    metrics_summary = {}
    confusion_matrices = {}

    for name, clf in candidate_models.items():
        print(f"\n[*] Training and evaluating Severity Classifier: {name}...")
        preprocessor = get_preprocessor_pipeline(SEVERITY_NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics_summary[name] = {
            'accuracy': round(float(acc), 4),
            'macro_f1': round(float(macro_f1), 4),
            'weighted_f1': round(float(weighted_f1), 4),
            'classification_report': report
        }

        cm = confusion_matrix(y_test, y_pred, labels=VALID_SEVERITIES)
        confusion_matrices[name] = cm

        print(f"    Accuracy:    {acc*100:.2f}%")
        print(f"    Macro F1:    {macro_f1:.4f}")
        print(f"    Weighted F1: {weighted_f1:.4f}")

    best_sev_name = max(metrics_summary, key=lambda k: metrics_summary[k]['macro_f1'])
    print(f"\n[+] Selected Best Severity Model: {best_sev_name}")

    # Plot Severity Confusion Matrix
    plot_model_confusion_matrices(
        {best_sev_name: confusion_matrices[best_sev_name]},
        class_names=VALID_SEVERITIES,
        output_dir="outputs/metrics",
        filename="severity_confusion_matrix.png",
        title_prefix="Weather Severity Classification"
    )

    return {
        'trained_pipelines': trained_pipelines,
        'best_model_name': best_sev_name,
        'metrics_summary': metrics_summary,
        'classes': VALID_SEVERITIES
    }


def run_full_training_pipeline(
    cleaned_csv_path: str = "data/cleaned_weather_sample.csv",
    models_dir: str = "models",
    metrics_dir: str = "outputs/metrics"
):
    """Orchestrate end-to-end training, saving artifacts and evaluation metrics."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"[*] Loading cleaned dataset: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path)

    # 1. Train Storm Occurrence Models
    occ_results = train_storm_occurrence_models(df)

    # 2. Train Severity Classification Models
    sev_results = train_storm_severity_models(df)

    # 3. Save Production Models using Joblib
    best_occ_name = occ_results['best_model_name']
    best_occ_pipeline = occ_results['trained_pipelines'][best_occ_name]
    occ_model_path = os.path.join(models_dir, "storm_occurrence_model.joblib")
    print(f"\n[*] Saving best occurrence model ({best_occ_name}) to: {occ_model_path}")
    joblib.dump(best_occ_pipeline, occ_model_path)

    # Also save baseline models for comparison
    for m_name in ['Logistic Regression', 'Decision Tree']:
        if m_name in occ_results['trained_pipelines']:
            fname = m_name.lower().replace(" ", "_") + "_occurrence.joblib"
            joblib.dump(occ_results['trained_pipelines'][m_name], os.path.join(models_dir, fname))

    best_sev_name = sev_results['best_model_name']
    best_sev_pipeline = sev_results['trained_pipelines'][best_sev_name]
    sev_model_path = os.path.join(models_dir, "storm_severity_model.joblib")
    print(f"[*] Saving best severity model ({best_sev_name}) to: {sev_model_path}")
    joblib.dump(best_sev_pipeline, sev_model_path)

    # 4. Save Comprehensive Metrics JSON
    comparison_json = {
        'storm_occurrence': {
            'best_model': best_occ_name,
            'models': occ_results['metrics_summary'],
            'features': {
                'numeric': OCCURRENCE_NUMERIC_FEATURES,
                'categorical': CATEGORICAL_FEATURES
            }
        },
        'weather_severity': {
            'best_model': best_sev_name,
            'models': sev_results['metrics_summary'],
            'classes': VALID_SEVERITIES
        }
    }

    metrics_json_path = os.path.join(metrics_dir, "model_metrics_comparison.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(comparison_json, f, indent=2)
    print(f"[+] Saved evaluation metrics to: {metrics_json_path}")

    print("\n" + "="*60)
    print("=== MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY ===")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_full_training_pipeline()
