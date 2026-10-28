"""
Data preprocessing, feature engineering, and pipeline utilities
for the Storm Prediction & Severity Analysis Machine Learning Project.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Feature definitions
# Storm Occurrence features decouple precipitation to avoid the ASOS lightning sensor artifact
OCCURRENCE_NUMERIC_FEATURES = [
    'latitude',
    'longitude',
    'month',
    'day',
    'hour'
]

# Weather Severity features include precipitation as an intensity driver
SEVERITY_NUMERIC_FEATURES = [
    'precipitation_in',
    'latitude',
    'longitude',
    'month',
    'day',
    'hour'
]

NUMERIC_FEATURES = OCCURRENCE_NUMERIC_FEATURES

CATEGORICAL_FEATURES = [
    'timezone',
    'season',
    'state'
]

# Valid severity levels for storm classification
VALID_SEVERITIES = ['Light', 'Moderate', 'Heavy', 'Severe']


def get_season(month: int) -> str:
    """Map month integer (1-12) to meteorological season."""
    if month in (12, 1, 2):
        return 'Winter'
    elif month in (3, 4, 5):
        return 'Spring'
    elif month in (6, 7, 8):
        return 'Summer'
    else:
        return 'Fall'


def extract_temporal_features(df: pd.DataFrame, time_col: str = 'StartTime(UTC)') -> pd.DataFrame:
    """
    Extract calendar and diurnal features from a UTC timestamp column.
    Prevents lookahead leakage by strictly utilizing event start time.
    """
    dt_series = pd.to_datetime(df[time_col], errors='coerce')
    df = df.copy()
    df['month'] = dt_series.dt.month.fillna(1).astype(int)
    df['day'] = dt_series.dt.day.fillna(1).astype(int)
    df['hour'] = dt_series.dt.hour.fillna(12).astype(int)
    df['day_of_week'] = dt_series.dt.day_name().fillna('Unknown')
    df['season'] = df['month'].apply(get_season)
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize raw dataset column names to clean, snake_case conventions."""
    rename_dict = {
        'Type': 'event_type',
        'Severity': 'severity',
        'Precipitation(in)': 'precipitation_in',
        'TimeZone': 'timezone',
        'LocationLat': 'latitude',
        'LocationLng': 'longitude',
        'State': 'state',
        'City': 'city',
        'County': 'county',
        'AirportCode': 'airport_code',
        'ZipCode': 'zip_code'
    }
    return df.rename(columns=rename_dict)


def clean_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply end-to-end cleaning and feature derivation to raw or sampled data.
    """
    df = df.copy()
    
    # Standardize column naming if original names present
    if 'Precipitation(in)' in df.columns or 'Type' in df.columns:
        df = standardize_column_names(df)
        
    # Extract temporal features if StartTime(UTC) present
    if 'StartTime(UTC)' in df.columns:
        df = extract_temporal_features(df, time_col='StartTime(UTC)')
    elif 'start_time' in df.columns:
        df = extract_temporal_features(df, time_col='start_time')
        
    # Clean numeric fields
    if 'precipitation_in' in df.columns:
        df['precipitation_in'] = pd.to_numeric(df['precipitation_in'], errors='coerce').fillna(0.0)
        # Cap unreasonable gauge outliers (> 20 inches in a single reading)
        df['precipitation_in'] = df['precipitation_in'].clip(lower=0.0, upper=20.0)
        
    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
    # Standardize string fields
    for col in ['event_type', 'severity', 'timezone', 'state']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Derive binary storm occurrence target
    if 'event_type' in df.columns:
        df['is_storm'] = (df['event_type'] == 'Storm').astype(int)
        
    return df


def get_preprocessor_pipeline(
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Construct a Scikit-Learn ColumnTransformer pipeline.
    - Numeric features: Median imputation + StandardScaler
    - Categorical features: Most frequent imputation + OneHotEncoder(handle_unknown='ignore')
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor