"""
data_loader.py
--------------
Loads and cleans raw data from a CSV file.
No Streamlit imports allowed in this module.
"""

import pandas as pd
import numpy as np
import io


NUMERIC_COLS = [
    'relative_humidity', 'pressure', 'wind_speed', 'wind_direction',
    'rainfall', 'radiation', 'temperature', 'earth_skin_temp', 'discharge'
]

RENAME_MAP = {
    'Date': 'date', 'UT': 'time',
    'Relative humidity': 'relative_humidity',
    'Pressure': 'pressure',
    'Wind_speed': 'wind_speed',
    'Wind_dir': 'wind_direction',
    'Rainfall': 'rainfall',
    'Radiation': 'radiation',
    'Temp_C': 'temperature',
    'Earth-skin- Temp': 'earth_skin_temp',
    'Discharge': 'discharge',
}


def load_csv(file_obj) -> pd.DataFrame:
    """
    Load a cleaned CSV (already in the processed form with a 'date' index).
    Accepts a file path (str) or a file-like object.
    """
    if isinstance(file_obj, (str,)):
        df = pd.read_csv(file_obj, index_col='date', parse_dates=True)
    else:
        content = file_obj.read()
        df = pd.read_csv(io.BytesIO(content), index_col='date', parse_dates=True)

    df.sort_index(inplace=True)
    df['year'] = df.index.year
    df['month'] = df.index.month
    return df


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a raw DataFrame extracted from PDF / CSV and applies
    the same cleaning pipeline used in the notebook.
    """
    df.columns = df.columns.str.replace('\n', '').str.strip()
    df.rename(columns=RENAME_MAP, inplace=True)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date'] = pd.to_datetime(df.get('date', df.index), errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df['year'] = df.index.year
    df['month'] = df.index.month

    # Interpolate rainfall; ffill/bfill the rest
    if 'rainfall' in df.columns:
        df['rainfall'] = df['rainfall'].interpolate(method='linear')
    df = df.ffill().bfill()

    # Outlier flag
    if 'rainfall' in df.columns:
        Q1 = df['rainfall'].quantile(0.25)
        Q3 = df['rainfall'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df['is_outlier'] = ((df['rainfall'] < lower) | (df['rainfall'] > upper)).astype(int)

    return df


def train_test_split(df: pd.DataFrame, split_ratio: float = 0.8):
    """Temporal train/test split. Returns (train, test)."""
    split_idx = int(len(df) * split_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def get_available_vars(df: pd.DataFrame):
    """Return the variables present in the DataFrame."""
    all_vars = ['rainfall', 'temperature', 'pressure', 'wind_speed',
                'radiation', 'discharge', 'earth_skin_temp', 'relative_humidity']
    return [v for v in all_vars if v in df.columns]
