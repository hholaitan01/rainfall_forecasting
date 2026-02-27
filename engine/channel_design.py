"""
channel_design.py
-----------------
Manning's equation based channel design.
Best hydraulic section assumptions.
Q_design MUST come from rational method — never from ML.
No Streamlit imports allowed.
"""

import numpy as np
from scipy.optimize import fsolve

V_MIN = 0.6   # m/s  minimum velocity (self-cleansing)
V_MAX = 6.0   # m/s  maximum velocity (erosion limit)


def _freeboard(y: float) -> float:
    """Freeboard formula: F = 0.3 + 0.25*y  (standard practice)."""
    return 0.3 + 0.25 * y


def design_rectangular(Q: float, n: float, S: float) -> dict:
    """
    Best hydraulic section rectangular channel (b = 2y).
    Uses Manning's equation.
    Q  = design discharge (m³/s)
    n  = Manning's roughness
    S  = channel bed slope (m/m)
    Returns dict of hydraulic properties.
    """
    def equation(y):
        if y <= 0:
            return 1e10
        b = 2 * y
        A = b * y
        P = b + 2 * y
        R = A / P
        return (1 / n) * A * (R ** (2 / 3)) * (S ** 0.5) - Q

    y0    = fsolve(equation, 1.0)[0]
    b0    = 2 * y0
    A0    = b0 * y0
    P0    = b0 + 2 * y0
    R0    = A0 / P0
    V0    = Q / A0
    F0    = _freeboard(y0)
    Yt0   = y0 + F0
    ok    = V_MIN <= V0 <= V_MAX

    return {
        'type':            'Rectangular',
        'Q_design':        Q,
        'n':               n,
        'S':               S,
        'y':               y0,
        'b':               b0,
        'z':               0,
        'A':               A0,
        'P':               P0,
        'R':               R0,
        'V':               V0,
        'F':               F0,
        'Y_total':         Yt0,
        'velocity_ok':     ok,
        'velocity_status': '✅ OK' if ok else '⚠️ CHECK',
    }


def design_trapezoidal(Q: float, n: float, S: float, z: float = 1.0) -> dict:
    """
    Best hydraulic section trapezoidal channel.
    z  = side slope (H:V). Default z=1 gives 45° sides.
    Q  = design discharge (m³/s)
    n  = Manning's roughness
    S  = channel bed slope (m/m)
    Returns dict of hydraulic properties.
    """
    def equation(y):
        if y <= 0:
            return 1e10
        b_t = 2 * y * (np.sqrt(1 + z ** 2) - z)
        A_t = (b_t + z * y) * y
        P_t = b_t + 2 * y * np.sqrt(1 + z ** 2)
        R_t = A_t / P_t
        return (1 / n) * A_t * (R_t ** (2 / 3)) * (S ** 0.5) - Q

    y0   = fsolve(equation, 1.0)[0]
    b0   = 2 * y0 * (np.sqrt(1 + z ** 2) - z)
    A0   = (b0 + z * y0) * y0
    P0   = b0 + 2 * y0 * np.sqrt(1 + z ** 2)
    R0   = A0 / P0
    V0   = Q / A0
    F0   = _freeboard(y0)
    Yt0  = y0 + F0
    ok   = V_MIN <= V0 <= V_MAX

    return {
        'type':            'Trapezoidal',
        'Q_design':        Q,
        'n':               n,
        'S':               S,
        'y':               y0,
        'b':               b0,
        'z':               z,
        'A':               A0,
        'P':               P0,
        'R':               R0,
        'V':               V0,
        'F':               F0,
        'Y_total':         Yt0,
        'velocity_ok':     ok,
        'velocity_status': '✅ OK' if ok else '⚠️ CHECK',
    }
