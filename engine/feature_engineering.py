"""
feature_engineering.py
-----------------------
Verbatim extraction of Phase 4 feature engineering from the notebook.
No Streamlit imports allowed.
"""

import pandas as pd
import numpy as np

TARGETS     = ['rainfall', 'discharge']
LAG_VARS    = ['rainfall', 'discharge']
LAG_PERIODS = [1, 2, 3, 6, 12]
ROLL_VARS   = ['rainfall', 'discharge', 'temperature', 'pressure', 'wind_speed', 'radiation']
ROLL_WINDOWS = [3, 6, 12]
EXCLUDE     = ['season', 'is_outlier', 'time']


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for var in LAG_VARS:
        if var in df.columns:
            for lag in LAG_PERIODS:
                df[f'{var}_lag{lag}'] = df[var].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for var in ROLL_VARS:
        if var in df.columns:
            for window in ROLL_WINDOWS:
                df[f'{var}_roll{window}_mean'] = (
                    df[var].rolling(window=window, min_periods=1).mean().shift(1)
                )
    for window in ROLL_WINDOWS:
        if 'rainfall' in df.columns:
            df[f'rainfall_roll{window}_std'] = (
                df['rainfall'].rolling(window=window, min_periods=1).std().shift(1)
            )
    return df


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def add_season_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['season'] = df['month'].apply(
        lambda m: 'wet' if m in [4, 5, 6, 7, 8, 9] else 'dry'
    )
    df['is_wet_season'] = (df['season'] == 'wet').astype(int)
    return df


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['time_index'] = np.arange(len(df))
    return df


def add_yoy_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'rainfall' in df.columns:
        df['rainfall_yoy_change'] = df['rainfall'] - df['rainfall'].shift(12)
    if 'discharge' in df.columns:
        df['discharge_yoy_change'] = df['discharge'] - df['discharge'].shift(12)
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'pressure' in df.columns and 'wind_speed' in df.columns:
        df['pressure_x_wind'] = df['pressure'] * df['wind_speed']
    if 'radiation' in df.columns and 'relative_humidity' in df.columns:
        df['radiation_x_humidity'] = df['radiation'] * df['relative_humidity']
    return df


def drop_lag_nans(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = [c for c in df.columns if '_lag' in c or '_yoy_' in c]
    return df.dropna(subset=lag_cols).copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_cyclical_encoding(df)
    df = add_season_flags(df)
    df = add_trend(df)
    df = add_yoy_change(df)
    df = add_interactions(df)
    df = drop_lag_nans(df)
    return df


def get_feature_cols(df: pd.DataFrame, target: str) -> list:
    """Return ML-safe feature columns for a given target."""
    safe = [
        c for c in df.columns
        if c not in TARGETS + EXCLUDE
        and df[c].dtype in [np.float64, np.int64, float, int]
        and (target not in c or 'lag' in c or 'roll' in c)
    ]
    return safe
