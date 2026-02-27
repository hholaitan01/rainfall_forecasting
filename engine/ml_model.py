"""
ml_model.py
-----------
XGBoost model training and loading.
Preserves the notebook's tuned XGBoost pipeline exactly.
No Streamlit imports allowed.
"""

import numpy as np
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform

from engine.feature_engineering import get_feature_cols, TARGETS

XGB_PARAM_DIST = {
    'n_estimators':     randint(100, 500),
    'max_depth':        randint(3, 8),
    'learning_rate':    uniform(0.01, 0.15),
    'subsample':        uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'reg_alpha':        uniform(0, 1),
    'reg_lambda':       uniform(0.5, 2),
}


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame,
                  target: str, n_iter: int = 20, n_splits: int = 5):
    """
    Trains XGBoost with RandomizedSearchCV + TimeSeriesSplit.
    Returns (fitted_model, feature_cols, metrics_dict).
    """
    features = get_feature_cols(train, target)
    X_train  = train[features].fillna(0)
    y_train  = train[target]
    X_test   = test[features].fillna(0)
    y_test   = test[target]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rnd_search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
    )
    rnd_search.fit(X_train, y_train)
    best_model = rnd_search.best_estimator_
    best_model.fit(X_train, y_train,
                   eval_set=[(X_test, y_test)],
                   verbose=False)

    preds = np.maximum(best_model.predict(X_test), 0)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)

    train_preds = np.maximum(best_model.predict(X_train), 0)

    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mae': float(np.mean(np.abs(y_test.values - preds))),
        'test_actual': y_test.values,
        'test_predicted': preds,
        'train_actual': y_train.values,
        'train_predicted': train_preds,
        'best_params': rnd_search.best_params_,
        'cv_rmse': -rnd_search.best_score_,
        'feature_importances': dict(zip(features, best_model.feature_importances_)),
    }
    return best_model, features, metrics


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)


def predict(model, df_features: pd.DataFrame, features: list) -> np.ndarray:
    X = df_features[features].fillna(0)
    return np.maximum(model.predict(X), 0)
