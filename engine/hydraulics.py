"""
hydraulics.py
-------------
Rational method + IDF computations.
Q_rational is the ONLY discharge used for channel sizing.
No Streamlit imports allowed.
"""

import numpy as np


def compute_time_of_concentration(L: float, S: float) -> dict:
    """
    Kirpich formula: Tc (minutes) for a catchment.
    L  = longest flow path (m)
    S  = average slope (m/m)
    Returns dict with Tc_min and Tc_hr.
    """
    Tc_min = 0.0195 * (L ** 0.77) * (S ** -0.385)
    return {'Tc_min': Tc_min, 'Tc_hr': Tc_min / 60}


def compute_idf_intensity(Tc_min: float, A_idf: float = 8266,
                          B_idf: float = 63.1) -> float:
    """
    IDF equation: i = A / (Tc + B)   [mm/hr]
    Coefficients from Ikeja, Nigeria (10-year return period).
    """
    return A_idf / (Tc_min + B_idf)


def compute_rational_discharge(C: float, i_mm_hr: float,
                                A_km2: float) -> float:
    """
    Rational method: Q = 0.278 * C * i * A
    C     = runoff coefficient
    i     = rainfall intensity (mm/hr)
    A_km2 = catchment area (km²)
    Returns Q in m³/s.
    """
    return 0.278 * C * i_mm_hr * A_km2


def rational_method_summary(L: float, S_catch: float, C: float,
                             A_km2: float,
                             A_idf: float = 8266,
                             B_idf: float = 63.1) -> dict:
    """
    Full rational method pipeline.
    Returns a summary dict of intermediate and final values.
    """
    tc    = compute_time_of_concentration(L, S_catch)
    i_idf = compute_idf_intensity(tc['Tc_min'], A_idf, B_idf)
    Q     = compute_rational_discharge(C, i_idf, A_km2)
    return {
        'L_m':       L,
        'S_catch':   S_catch,
        'C':         C,
        'A_km2':     A_km2,
        'Tc_min':    tc['Tc_min'],
        'Tc_hr':     tc['Tc_hr'],
        'A_idf':     A_idf,
        'B_idf':     B_idf,
        'i_idf':     i_idf,
        'Q_rational': Q,
    }
