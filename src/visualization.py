"""
Visualization utilities for Exploratory Data Analysis (EDA) and Model Evaluation.
Generates publication-quality charts using Matplotlib and Seaborn.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


# Set global aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 14


def plot_missing_values(df: pd.DataFrame, output_dir: str = "outputs/eda") -> str:
    """Plot missing value counts and percentages across all columns."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eda_missing_values.png")
    
    missing = df.isnull().sum()
    pct_missing = (missing / len(df)) * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(pct_missing.index, pct_missing.values, color='#3498db', edgecolor='black', alpha=0.85)
    ax.set_ylabel('Missing Percentage (%)')
    ax.set_title('Dataset Missing-Value Analysis by Feature', fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim(0, max(max(pct_missing.values) * 1.3, 5))
    
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.annotate('0%',
                        xy=(bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color='#7f8c8d')
                        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_class_distributions(df: pd.DataFrame, output_dir: str = "outputs/eda") -> str:
    """Plot distribution of weather event types and binary storm target."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eda_class_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Event types distribution
    event_counts = df['event_type'].value_counts()
    palette = sns.color_palette("muted", len(event_counts))
    axes[0].bar(event_counts.index, event_counts.values, color=palette, edgecolor='black', alpha=0.85)
    axes[0].set_title('Weather Event Type Distribution', fontweight='bold')
    axes[0].set_ylabel('Frequency')
    axes[0].tick_params(axis='x', rotation=30)
    for i, v in enumerate(event_counts.values):
        axes[0].text(i, v + (max(event_counts.values)*0.01), f"{v:,}\n({v/len(df)*100:.1f}%)", 
                     ha='center', fontsize=8, fontweight='bold')

    # Binary Storm vs Non-Storm
    storm_counts = df['is_storm'].value_counts().rename({0: 'No Storm (Non-convective)', 1: 'Storm (Thunderstorm)'})
    colors = ['#2ecc71', '#e74c3c']
    axes[1].pie(storm_counts.values, labels=storm_counts.index, autopct='%1.1f%%',
                startangle=140, colors=colors, explode=(0.04, 0.04),
                wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    axes[1].set_title('Target 1: Storm Occurrence Proportion', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_severity_breakdown(df: pd.DataFrame, output_dir: str = "outputs/eda") -> str:
    """Plot cross-tabulation of Severity across Weather Event Types."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eda_severity_distribution.png")

    ct = pd.crosstab(df['event_type'], df['severity'])
    
    fig, ax = plt.subplots(figsize=(11, 6))
    ct.plot(kind='bar', stacked=True, ax=ax, colormap='Spectral', edgecolor='black', alpha=0.85)
    ax.set_title('Severity Category Breakdown by Weather Event Type', fontweight='bold', pad=15)
    ax.set_xlabel('Event Type')
    ax.set_ylabel('Record Count')
    plt.xticks(rotation=30, ha='right')
    ax.legend(title='Reported Severity', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_temporal_patterns(df: pd.DataFrame, output_dir: str = "outputs/eda") -> str:
    """Plot storm occurrences across months (seasonal) and hours (diurnal)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eda_temporal_patterns.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Storm occurrence by month
    monthly_storm = df[df['is_storm'] == 1].groupby('month').size()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months_idx = range(1, 13)
    storm_vals = [monthly_storm.get(m, 0) for m in months_idx]

    axes[0].plot(month_names, storm_vals, marker='o', linewidth=2.5, color='#e67e22', markersize=7)
    axes[0].fill_between(month_names, storm_vals, color='#e67e22', alpha=0.2)
    axes[0].set_title('Seasonal Storm Distribution (by Month)', fontweight='bold')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Storm Count')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Storm occurrence by hour (diurnal cycle)
    hourly_storm = df[df['is_storm'] == 1].groupby('hour').size()
    hours_idx = range(24)
    hour_vals = [hourly_storm.get(h, 0) for h in hours_idx]

    axes[1].bar(hours_idx, hour_vals, color='#9b59b6', edgecolor='black', alpha=0.85)
    axes[1].set_title('Diurnal Storm Distribution (by Hour UTC)', fontweight='bold')
    axes[1].set_xlabel('Hour of Day (UTC)')
    axes[1].set_ylabel('Storm Count')
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_geographic_scatter(df: pd.DataFrame, output_dir: str = "outputs/eda", sample_size: int = 5000) -> str:
    """Plot geographic distribution of storm vs non-storm airport stations across US coordinates."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eda_geographic_distribution.png")

    # Filter contiguous US bounding box
    geo_df = df[(df['latitude'] >= 24) & (df['latitude'] <= 50) & 
                (df['longitude'] >= -125) & (df['longitude'] <= -65)]
    if len(geo_df) > sample_size:
        geo_df = geo_df.sample(sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=geo_df,
        x='longitude', y='latitude',
        hue='event_type', style='is_storm',
        palette='tab10', alpha=0.6, s=35, ax=ax
    )
    ax.set_title('Geographic Weather Station Events across Contiguous US', fontweight='bold', pad=15)
    ax.set_xlabel('Longitude (°W)')
    ax.set_ylabel('Latitude (°N)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Event Type')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "outputs/eda") -> str:
    """Plot Pearson correlation heatmap between numerical features and storm indicator."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eda_correlation_heatmap.png")

    num_cols = ['precipitation_in', 'latitude', 'longitude', 'month', 'day', 'hour', 'is_storm']
    available_cols = [c for c in num_cols if c in df.columns]
    corr = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, linecolor='white', square=True, ax=ax)
    ax.set_title('Correlation Heatmap (Meteorological & Spatio-Temporal Features)', fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_model_confusion_matrices(
    confusion_matrices: dict,
    class_names: list,
    output_dir: str = "outputs/metrics",
    filename: str = "occurrence_confusion_matrices.png",
    title_prefix: str = "Storm Occurrence"
) -> str:
    """Plot side-by-side normalized confusion matrices for model comparison."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    n_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, confusion_matrices.items()):
        # Normalize by true label row
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        ax.set_title(f"{model_name}\n({title_prefix})", fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_roc_and_calibration(
    models_roc: dict,
    models_calib: dict,
    output_dir: str = "outputs/metrics"
) -> Tuple[str, str]:
    """Plot ROC curves and Calibration reliability curves for Occurrence models."""
    os.makedirs(output_dir, exist_ok=True)
    roc_path = os.path.join(output_dir, "occurrence_roc_curves.png")
    calib_path = os.path.join(output_dir, "occurrence_calibration_curves.png")

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in models_roc.items():
        fpr, tpr, roc_auc = data['fpr'], data['tpr'], data['auc']
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close(fig)

    # 2. Calibration Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in models_calib.items():
        prob_true, prob_pred, brier = data['prob_true'], data['prob_pred'], data['brier']
        ax.plot(prob_pred, prob_true, marker='s', label=f"{name} (Brier = {brier:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives (Empirical Probability)')
    ax.set_title('Probability Calibration Reliability Diagram', fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(calib_path, dpi=300)
    plt.close(fig)

    return roc_path, calib_path


def plot_feature_importances(
    feature_names: list,
    importances: np.ndarray,
    output_dir: str = "outputs/metrics",
    top_n: int = 15
) -> str:
    """Plot top N feature importances from tree-based model."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "feature_importances.png")

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_scores[::-1], color='#1abc9c', edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel('Gini Feature Importance')
    ax.set_title(f'Top {top_n} Features Driving Storm Occurrence Prediction', fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_all_eda_plots(csv_path: str = "data/cleaned_weather_sample.csv", output_dir: str = "outputs/eda"):
    """Run full suite of Phase 1 EDA visualizations."""
    print(f"[*] Loading cleaned dataset for EDA generation from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df):,} rows.")

    print("[*] Generating Missing Values Plot...")
    p1 = plot_missing_values(df, output_dir)
    print(f"  -> {p1}")

    print("[*] Generating Class Distributions Plot...")
    p2 = plot_class_distributions(df, output_dir)
    print(f"  -> {p2}")

    print("[*] Generating Severity Breakdown Plot...")
    p3 = plot_severity_breakdown(df, output_dir)
    print(f"  -> {p3}")

    print("[*] Generating Temporal Patterns Plot...")
    p4 = plot_temporal_patterns(df, output_dir)
    print(f"  -> {p4}")

    print("[*] Generating Geographic Station Plot...")
    p5 = plot_geographic_scatter(df, output_dir)
    print(f"  -> {p5}")

    print("[*] Generating Correlation Heatmap...")
    p6 = plot_correlation_heatmap(df, output_dir)
    print(f"  -> {p6}")

    print("[+] All Phase 1 EDA plots generated successfully in:", output_dir)


if __name__ == "__main__":
    generate_all_eda_plots()
