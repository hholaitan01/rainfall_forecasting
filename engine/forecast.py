"""
forecast.py
-----------
Recursive 12-month XGBoost forecasting.
Verbatim extraction from Phase 6 of the notebook.
No Streamlit imports allowed.
"""

import pandas as pd
import numpy as np
from engine.feature_engineering import LAG_PERIODS


def recursive_forecast(model, df: pd.DataFrame, features: list,
                        target: str, n_months: int = 12,
                        start_date: str = None) -> pd.Series:
    """
    Recursively forecast `n_months` beyond the end of `df`.

    Parameters
    ----------
    model     : trained XGBoost model
    df        : full featured DataFrame (train + test concatenated)
    features  : list of feature column names used by the model
    target    : 'rainfall' or 'discharge'
    n_months  : forecast horizon (default 12)
    start_date: 'YYYY-MM-DD' override for first forecast month

    Returns
    -------
    pd.Series with forecast values indexed by date
    """
    if start_date is None:
        last_date   = df.index[-1]
        # Move to next month
        if last_date.month == 12:
            first_future = pd.Timestamp(last_date.year + 1, 1, 1)
        else:
            first_future = pd.Timestamp(last_date.year, last_date.month + 1, 1)
    else:
        first_future = pd.Timestamp(start_date)

    forecast_dates  = pd.date_range(start=first_future, periods=n_months, freq='MS')
    forecast_values = []
    last_known      = df.copy()

    for step, future_date in enumerate(forecast_dates):
        row = last_known[features].iloc[-1].copy()

        # Update time features
        row['month']        = future_date.month
        row['year']         = future_date.year
        row['time_index']   = last_known['time_index'].iloc[-1] + 1
        row['month_sin']    = np.sin(2 * np.pi * future_date.month / 12)
        row['month_cos']    = np.cos(2 * np.pi * future_date.month / 12)
        row['is_wet_season'] = 1 if future_date.month in [4, 5, 6, 7, 8, 9] else 0

        # Update lag features using previously forecasted values
        for lag in LAG_PERIODS:
            lag_col = f'{target}_lag{lag}'
            if lag_col in features:
                if lag <= step:
                    row[lag_col] = forecast_values[step - lag]
                else:
                    row[lag_col] = last_known[target].iloc[-(lag - step)]

        # Predict
        X_future = pd.DataFrame([row[features]])
        pred     = float(np.maximum(model.predict(X_future), 0))
        forecast_values.append(pred)

        # Append to last_known so next iteration can use it
        new_row          = last_known.iloc[-1].copy()
        new_row[target]  = pred
        new_row['time_index'] += 1
        last_known = pd.concat(
            [last_known, pd.DataFrame([new_row], index=[future_date])]
        )

    return pd.Series(forecast_values, index=forecast_dates, name=target)


def build_forecast_dataframe(rainfall_forecast: pd.Series,
                              discharge_forecast: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({
        'rainfall_forecast':  rainfall_forecast,
        'discharge_forecast': discharge_forecast,
    })
    df.index.name = 'date'
    return df
