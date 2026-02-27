"""
plots.py
--------
All matplotlib visualisation functions.
Returns matplotlib Figure objects — no plt.show() calls.
No Streamlit imports allowed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    'rainfall':  '#1f77b4',
    'discharge': '#d62728',
    'train':     '#1f77b4',
    'test':      '#ff7f0e',
    'pred':      '#2ca02c',
    'band':      '#aec7e8',
    'neutral':   '#7f7f7f',
}
FONT_TITLE   = {'fontsize': 13, 'fontweight': 'bold'}
FONT_LABEL   = {'fontsize': 10}
FONT_TICK    = {'labelsize': 9}
GRID_KW      = {'alpha': 0.3}


# ─────────────────────────────────────────────
# 1. TIME SERIES OVERVIEW
# ─────────────────────────────────────────────
def plot_timeseries_overview(df: pd.DataFrame, vars_present: list) -> plt.Figure:
    units = {
        'rainfall': 'mm', 'temperature': '°C', 'pressure': 'hPa',
        'wind_speed': 'm/s', 'radiation': 'MJ/m²', 'discharge': 'm³/s',
        'earth_skin_temp': '°C', 'relative_humidity': '%',
    }
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    n = len(vars_present)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, var, color in zip(axes, vars_present, palette):
        ax.plot(df.index, df[var], linewidth=1, color=color)
        unit = units.get(var, '')
        ax.set_ylabel(f"{var.replace('_', ' ').title()}\n({unit})", fontsize=9)
        ax.set_xlabel('Date', fontsize=8)
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(5))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.autoscale(axis='y')
        ax.margins(y=0.1)
        ax.grid(True, alpha=0.3)
        ax.axhline(df[var].mean(), color='red', linestyle='--', linewidth=0.8,
                   alpha=0.6, label=f'Mean: {df[var].mean():.2f} {unit}')
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('Monthly Time Series — All Variables (Independent Axes)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(h_pad=1.5)
    return fig


# ─────────────────────────────────────────────
# 2. ACTUAL VS PREDICTED (with ±1 RMSE band)
# ─────────────────────────────────────────────
def plot_actual_vs_predicted(test_index, y_actual: np.ndarray,
                              y_pred: np.ndarray, rmse: float,
                              r2: float, target: str) -> plt.Figure:
    color = COLORS.get(target, '#1f77b4')
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_index, y_actual, color='black', linewidth=1.5,
            label='Actual', zorder=3)
    ax.plot(test_index, y_pred, color=color, linewidth=1.5,
            linestyle='--', label='XGBoost Predicted', zorder=3)
    ax.fill_between(test_index, y_pred - rmse, y_pred + rmse,
                    alpha=0.15, color=color, label=f'±1 RMSE band')
    unit = 'mm' if target == 'rainfall' else 'm³/s'
    ax.set_ylabel(f'{target.title()} ({unit})', **FONT_LABEL)
    ax.set_title(f'{target.title()} — Actual vs XGBoost  |  RMSE: {rmse:.3f}  |  R²: {r2:.3f}',
                 **FONT_TITLE)
    ax.set_xlabel('Date', **FONT_LABEL)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 3. SCATTER: ACTUAL vs PREDICTED
# ─────────────────────────────────────────────
def plot_scatter(y_actual: np.ndarray, y_pred: np.ndarray,
                 r2: float, target: str) -> plt.Figure:
    color = COLORS.get(target, '#1f77b4')
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_actual, y_pred, alpha=0.7, color=color,
               edgecolors='k', linewidths=0.3, s=60)
    lims = [min(y_actual.min(), y_pred.min()),
            max(y_actual.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect fit')
    z = np.polyfit(y_actual, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(sorted(y_actual), p(sorted(y_actual)), color='red',
            linewidth=1.5, linestyle='-', label='Trend')
    ax.set_xlabel(f'Actual {target.title()}', **FONT_LABEL)
    ax.set_ylabel(f'Predicted {target.title()}', **FONT_LABEL)
    ax.set_title(f'{target.title()} | R² = {r2:.3f}', **FONT_TITLE)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 4. RESIDUAL ANALYSIS (6-panel)
# ─────────────────────────────────────────────
def plot_residuals(test_index, y_actual: np.ndarray, y_pred: np.ndarray,
                   train_actual: np.ndarray, train_pred: np.ndarray,
                   target: str) -> plt.Figure:
    residuals       = y_actual - y_pred
    train_residuals = train_actual - train_pred

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Residual Analysis — {target.title()} (XGBoost)',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Residuals over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(test_index, residuals, color='#1f77b4', linewidth=1.2)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.fill_between(test_index, residuals, 0,
                     where=(residuals > 0), alpha=0.3, color='green',
                     label='Over-predicted')
    ax1.fill_between(test_index, residuals, 0,
                     where=(residuals < 0), alpha=0.3, color='red',
                     label='Under-predicted')
    ax1.set_title('Residuals Over Time (Test Set)')
    ax1.set_ylabel('Actual − Predicted')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Residuals vs Predicted
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(y_pred, residuals, alpha=0.6, color='#ff7f0e',
                edgecolors='k', linewidths=0.3)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)

    # 3. Residual Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(residuals, bins=20, color='#2ca02c', edgecolor='black', alpha=0.8)
    ax3.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax3.axvline(residuals.mean(), color='blue', linestyle='--',
                linewidth=1.5, label=f'Mean: {residuals.mean():.2f}')
    ax3.set_title('Residual Distribution')
    ax3.set_xlabel('Residual Value')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Q-Q Plot
    ax4 = fig.add_subplot(gs[2, 0])
    stats.probplot(residuals, dist='norm', plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)')
    ax4.grid(True, alpha=0.3)

    # 5. Train vs Test residuals
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.boxplot([train_residuals, residuals],
                tick_labels=['Train', 'Test'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax5.axhline(0, color='red', linestyle='--')
    ax5.set_title('Train vs Test Residuals')
    ax5.set_ylabel('Residual')
    ax5.grid(True, alpha=0.3)

    return fig


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(importances: dict, target: str, top_n: int = 15) -> plt.Figure:
    import pandas as pd
    imp_series = pd.Series(importances).nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    imp_series.plot(kind='barh', ax=ax, color='#1f77b4', edgecolor='black')
    ax.set_title(f'XGBoost — Top {top_n} Features ({target.title()})', **FONT_TITLE)
    ax.set_xlabel('Importance', **FONT_LABEL)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 6. FUTURE FORECAST
# ─────────────────────────────────────────────
def plot_forecast(history: pd.Series, forecast: pd.Series,
                  rmse: float, target: str) -> plt.Figure:
    color = COLORS.get(target, '#1f77b4')
    unit  = 'mm' if target == 'rainfall' else 'm³/s'
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(history.index, history.values, color='black', linewidth=1.5,
            label='Historical (last 3 years)')
    ax.plot(forecast.index, forecast.values, color=color, linewidth=2,
            linestyle='--', marker='o', markersize=5,
            label=f'Forecast ({forecast.index[0].year})')
    ax.fill_between(forecast.index,
                    forecast.values - rmse,
                    forecast.values + rmse,
                    alpha=0.2, color=color, label='±1 RMSE uncertainty')
    ax.axvline(history.index[-1], color='gray', linestyle=':', linewidth=1.5,
               label='Forecast start')
    ax.set_ylabel(f'{target.title()} ({unit})', **FONT_LABEL)
    ax.set_title(f'{target.title()} — 12-Month Forecast', **FONT_TITLE)
    ax.set_xlabel('Date', **FONT_LABEL)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 7. FLOOD RISK COMPARISON
# ─────────────────────────────────────────────
def plot_flood_risk(Q_ml: float, Q_rational: float,
                    safety_factor: float, risk_level: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ['Q_rational\n(IDF / Rational Method)\n[Design Discharge]',
              f'Q_ML\n(XGBoost 95th pct)\n[Basin-scale Risk]']
    values = [Q_rational, Q_ml]
    colors = ['#2ca02c', '#d62728']
    bars   = ax.bar(labels, values, color=colors, edgecolor='black', width=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + Q_ml * 0.01,
                f'{val:.4f} m³/s', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.axhline(Q_rational, color='#2ca02c', linestyle='--',
               linewidth=1, alpha=0.7)
    ax.set_ylabel('Discharge (m³/s)', **FONT_LABEL)
    ax.set_title(
        f'Design vs ML Discharge Comparison\n'
        f'Risk Level: {risk_level}  |  Safety Factor: {safety_factor:.1f}×',
        **FONT_TITLE)
    ax.grid(True, alpha=0.3, axis='y')
    # Annotation arrow
    ax.annotate(
        f'{safety_factor:.1f}× gap',
        xy=(1, Q_ml), xytext=(0.55, (Q_ml + Q_rational) / 2),
        fontsize=11, color='#d62728', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
    )
    plt.tight_layout()
    return fig
