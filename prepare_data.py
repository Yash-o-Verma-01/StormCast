"""
Dataset Preparation Script for Storm Prediction & Severity Analysis.

Extracts a statistically representative, stratified analytical dataset
from the raw 8.6M-row US Weather Events dataset (2016-2022) without modifying the original file.
Applies feature engineering, leakage prevention, and validation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import clean_weather_dataframe, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def prepare_dataset(
    raw_csv_path: str = "WeatherEvents_Jan2016-Dec2022.csv",
    output_csv_path: str = "data/cleaned_weather_sample.csv",
    summary_path: str = "outputs/data_summary.json",
    n_storms_target: int = 40000,
    n_non_storms_target: int = 60000,
    chunk_size: int = 500000,
    random_state: int = 42
):
    print(f"[*] Starting dataset preparation from: {raw_csv_path}")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_csv_path}. Please verify path.")

    np.random.seed(random_state)
    
    storms_collected = []
    non_storms_collected = []
    total_raw_processed = 0
    total_storms_seen = 0
    total_non_storms_seen = 0
    raw_event_counts = {}

    print(f"[*] Reading raw dataset in streaming chunks of {chunk_size:,} rows...")
    
    # Process chunks iteratively to prevent memory exhaustion
    for i, chunk in enumerate(pd.read_csv(raw_csv_path, chunksize=chunk_size)):
        total_raw_processed += len(chunk)
        
        # Track raw event counts
        for event, count in chunk['Type'].value_counts().items():
            raw_event_counts[event] = raw_event_counts.get(event, 0) + count
            
        storms_mask = chunk['Type'] == 'Storm'
        chunk_storms = chunk[storms_mask]
        chunk_non_storms = chunk[~storms_mask]
        
        total_storms_seen += len(chunk_storms)
        total_non_storms_seen += len(chunk_non_storms)
        
        # Sample storms with high retention
        if len(chunk_storms) > 0:
            storms_collected.append(chunk_storms)
            
        # Stratified sampling of non-storms proportionally across chunk
        # With 8.6M rows and 60k non-storms needed, sample rate is ~ 60k / 8.5M ≈ 0.007
        sampled_non_storms = chunk_non_storms.sample(frac=0.008, random_state=random_state + i)
        non_storms_collected.append(sampled_non_storms)
        
        print(f"  Processed {total_raw_processed:>9,} rows | Collected {sum(len(c) for c in storms_collected):>6,} storms, {sum(len(c) for c in non_storms_collected):>6,} non-storms", end='\r')

    print(f"\n[+] Raw pass complete. Total processed: {total_raw_processed:,} rows.")
    
    # Concatenate and sample to final targets
    print("[*] Consolidating and subsampling to target distribution...")
    df_storms = pd.concat(storms_collected, ignore_index=True)
    df_non_storms = pd.concat(non_storms_collected, ignore_index=True)
    
    if len(df_storms) > n_storms_target:
        df_storms = df_storms.sample(n=n_storms_target, random_state=random_state)
        
    if len(df_non_storms) > n_non_storms_target:
        df_non_storms = df_non_storms.sample(n=n_non_storms_target, random_state=random_state)

    combined_df = pd.concat([df_storms, df_non_storms], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[*] Cleaned sample size: {len(combined_df):,} records ({len(df_storms):,} storms, {len(df_non_storms):,} non-storms)")

    # Apply cleaning and feature engineering
    print("[*] Applying feature engineering and cleaning...")
    cleaned_df = clean_weather_dataframe(combined_df)
    
    # Retain analytical columns
    keep_cols = [
        'event_type', 'severity', 'is_storm',
        'precipitation_in', 'latitude', 'longitude',
        'month', 'day', 'hour', 'day_of_week', 'season',
        'timezone', 'state', 'city', 'county'
    ]
    cleaned_df = cleaned_df[[c for c in keep_cols if c in cleaned_df.columns]]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    print(f"[*] Saving cleaned dataset to: {output_csv_path}")
    cleaned_df.to_csv(output_csv_path, index=False)

    # Generate summary statistics
    summary = {
        "raw_dataset_total_rows": int(total_raw_processed),
        "raw_event_type_distribution": {k: int(v) for k, v in raw_event_counts.items()},
        "cleaned_dataset_rows": int(len(cleaned_df)),
        "storm_events_count": int(cleaned_df['is_storm'].sum()),
        "non_storm_events_count": int((cleaned_df['is_storm'] == 0).sum()),
        "event_type_distribution": cleaned_df['event_type'].value_counts().to_dict(),
        "storm_severity_distribution": cleaned_df[cleaned_df['is_storm'] == 1]['severity'].value_counts().to_dict(),
        "all_severity_distribution": cleaned_df['severity'].value_counts().to_dict(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "precipitation_stats": {
            "mean": float(cleaned_df['precipitation_in'].mean()),
            "std": float(cleaned_df['precipitation_in'].std()),
            "median": float(cleaned_df['precipitation_in'].median()),
            "max": float(cleaned_df['precipitation_in'].max())
        },
        "states_count": int(cleaned_df['state'].nunique()),
        "seasons_distribution": cleaned_df['season'].value_counts().to_dict()
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Summary saved to: {summary_path}")
    print("\n=== Dataset Summary ===")
    print(f"Total rows: {len(cleaned_df):,}")
    print(f"Storm occurrences: {summary['storm_events_count']:,} ({summary['storm_events_count']/len(cleaned_df)*100:.1f}%)")
    print(f"Non-storm occurrences: {summary['non_storm_events_count']:,} ({summary['non_storm_events_count']/len(cleaned_df)*100:.1f}%)")
    print("\nStorm Severity Breakdown:")
    for sev, cnt in summary['storm_severity_distribution'].items():
        print(f"  {sev}: {cnt:,}")
    print("========================\n")
    return cleaned_df


if __name__ == "__main__":
    prepare_dataset()
