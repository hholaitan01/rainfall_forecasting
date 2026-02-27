"""
risk_assessment.py
------------------
Flood risk assessment: compare ML peak discharge vs rational design discharge.
ML discharge is NEVER used to size channels — only for risk indication.
No Streamlit imports allowed.
"""

import numpy as np


def compute_flood_risk(Q_ml: float, Q_rational: float) -> dict:
    """
    Compare ML-derived peak discharge with rational design discharge.

    Parameters
    ----------
    Q_ml       : peak discharge from ML model (m³/s) — flood risk indicator
    Q_rational : design discharge from IDF / rational method (m³/s)

    Returns
    -------
    dict with risk interpretation and overdesign factor
    """
    safety_factor = Q_ml / Q_rational if Q_rational > 0 else float('inf')

    if safety_factor >= 10:
        risk_level = 'EXTREME'
        risk_color = 'red'
        recommendation = (
            "The ML-derived discharge exceeds the design capacity by more than 10×. "
            "The basin is highly vulnerable to catastrophic flooding during extreme events. "
            "Retention ponds, diversion channels, and flood early-warning systems are "
            "strongly recommended in addition to the designed drain."
        )
    elif safety_factor >= 5:
        risk_level = 'VERY HIGH'
        risk_color = 'red'
        recommendation = (
            "ML discharge exceeds channel capacity by 5–10×. Significant overtopping "
            "risk exists during extreme events. Supplementary flood mitigation measures "
            "(detention basins, overflow spillways) are recommended."
        )
    elif safety_factor >= 2:
        risk_level = 'HIGH'
        risk_color = 'orange'
        recommendation = (
            "ML discharge exceeds design discharge by 2–5×. This reflects the larger "
            "basin-scale hydrology captured by ML. Consider increasing channel freeboard "
            "or providing overflow capacity for rare extreme events."
        )
    else:
        risk_level = 'MODERATE'
        risk_color = 'goldenrod'
        recommendation = (
            "ML discharge is within 2× of design discharge. The channel design should "
            "provide adequate capacity under most conditions. Standard maintenance and "
            "monitoring practices are sufficient."
        )

    interpretation = (
        f"The XGBoost model predicts a peak discharge of {Q_ml:.2f} m³/s, derived from "
        f"historical river gauge records covering the entire upstream catchment. "
        f"This represents BASIN-SCALE flood risk, not the local drainage design scenario. "
        f"The rational method gives Q_design = {Q_rational:.4f} m³/s for the "
        f"local 0.5 km² catchment using 10-year IDF rainfall. "
        f"The ML discharge is {safety_factor:.1f}× larger than the design discharge, "
        f"which is physically expected given the scale difference between the river "
        f"gauge catchment and the local design catchment."
    )

    return {
        'Q_ml':           Q_ml,
        'Q_rational':     Q_rational,
        'safety_factor':  safety_factor,
        'risk_level':     risk_level,
        'risk_color':     risk_color,
        'recommendation': recommendation,
        'interpretation': interpretation,
    }


def get_ml_peak_discharge(test_actual: np.ndarray,
                           test_predicted: np.ndarray,
                           percentile: float = 95.0) -> float:
    """
    Compute the ML-derived peak discharge as the 95th percentile
    of predicted discharge on the test set.
    """
    return float(np.percentile(test_predicted, percentile))
