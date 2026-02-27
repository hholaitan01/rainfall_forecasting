"""
metrics.py
----------
Compute validation metrics. No Streamlit imports allowed.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae  = float(mean_absolute_error(actual, predicted))
    r2   = float(r2_score(actual, predicted))
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def compute_residual_stats(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residuals = actual - predicted
    _, p_norm  = stats.shapiro(residuals)
    return {
        'residuals':  residuals,
        'mean':       float(residuals.mean()),
        'std':        float(residuals.std()),
        'skew':       float(stats.skew(residuals)),
        'shapiro_p':  float(p_norm),
        'is_normal':  p_norm > 0.05,
    }
