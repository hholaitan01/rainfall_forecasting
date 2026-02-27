"""
channel_sections.py
-------------------
True-scale engineering cross-section plots for rectangular
and trapezoidal channels.
No Streamlit imports allowed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FONT_TITLE = {'fontsize': 11, 'fontweight': 'bold'}
FONT_LABEL = {'fontsize': 10}


def plot_channel_sections(rect: dict, trap: dict, Q_design: float,
                           n: float, S: float, C: float, A_km2: float) -> plt.Figure:
    """
    Plot both channel cross-sections side by side (true scale).
    rect, trap : dicts from channel_design.design_rectangular / design_trapezoidal
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Final Channel Design  |  Q_design = {Q_design:.4f} m³/s  '
        f'(IDF Method, 10-yr, Ikeja)\n'
        f'n = {n} (Concrete)  |  S = {S}  |  C = {C}  |  A_catch = {A_km2} km²',
        fontsize=12, fontweight='bold',
    )

    _draw_rectangular(axes[0], rect)
    _draw_trapezoidal(axes[1], trap)

    plt.tight_layout()
    return fig


def _draw_rectangular(ax, r: dict):
    y = r['y']
    b = r['b']
    F = r['F']
    Yt = r['Y_total']
    ok = r['velocity_ok']

    hb = b / 2
    # Channel walls
    ax.plot([-hb - 0.2, -hb, -hb, hb, hb, hb + 0.2],
            [0, 0, Yt, Yt, 0, 0], 'k-', linewidth=2.5)
    # Water fill
    ax.fill([-hb, -hb, hb, hb], [0, y, y, 0],
            color='#AED6F1', alpha=0.85, label=f'Water  y = {y:.3f} m')
    # Freeboard fill
    ax.fill([-hb, -hb, hb, hb], [y, Yt, Yt, y],
            color='#FDEBD0', alpha=0.85, label=f'Freeboard  F = {F:.3f} m')
    ax.axhline(y, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.7)

    # Width arrow
    ax.annotate('', xy=(hb, -0.05), xytext=(-hb, -0.05),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(0, -0.1, f'b = {b:.3f} m',
            ha='center', va='top', fontsize=10, fontweight='bold')
    # Water depth arrow
    ax.annotate('', xy=(hb + 0.15, y), xytext=(hb + 0.15, 0),
                arrowprops=dict(arrowstyle='<->', color='#1f77b4', lw=1.5))
    ax.text(hb + 0.22, y / 2, f'y = {y:.3f} m',
            va='center', fontsize=9, color='#1f77b4')
    # Total depth arrow
    ax.annotate('', xy=(hb + 0.32, Yt), xytext=(hb + 0.32, 0),
                arrowprops=dict(arrowstyle='<->', color='#d62728', lw=1.5))
    ax.text(hb + 0.39, Yt / 2, f'Y = {Yt:.3f} m',
            va='center', fontsize=9, color='#d62728')

    # Info box
    info = (f"V = {r['V']:.3f} m/s  {'✅' if ok else '⚠️'}\n"
            f"A = {r['A']:.3f} m²\nR = {r['R']:.3f} m")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=9,
            va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlim(-hb - 0.5, hb + 0.6)
    ax.set_ylim(-0.25, Yt + 0.25)
    ax.set_xlabel('Width (m)', **FONT_LABEL)
    ax.set_ylabel('Depth (m)', **FONT_LABEL)
    ax.set_title(f'(A) Rectangular Channel\nb = {b:.3f} m  |  y = {y:.3f} m  |  Y_total = {Yt:.3f} m',
                 **FONT_TITLE)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def _draw_trapezoidal(ax, t: dict):
    y   = t['y']
    b   = t['b']
    z   = t['z']
    F   = t['F']
    Yt  = t['Y_total']
    ok  = t['velocity_ok']

    hbt  = b / 2
    so_w = z * y
    so_t = z * Yt

    # Channel walls (outer shape)
    ax.plot([-(hbt + so_t) - 0.2, -(hbt + so_t), -hbt, hbt, hbt + so_t, hbt + so_t + 0.2],
            [Yt, Yt, 0, 0, Yt, Yt], 'k-', linewidth=2.5)
    # Water fill
    ax.fill([-(hbt + so_w), -hbt, hbt, hbt + so_w],
            [y, 0, 0, y],
            color='#AED6F1', alpha=0.85, label=f'Water  y = {y:.3f} m')
    # Freeboard fill
    ax.fill([-(hbt + so_w), -(hbt + so_t), hbt + so_t, hbt + so_w],
            [y, Yt, Yt, y],
            color='#FDEBD0', alpha=0.85, label=f'Freeboard  F = {F:.3f} m')
    ax.axhline(y, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0, y + 0.04, 'Water Level', ha='center', fontsize=8, color='#1f77b4')

    # Base width arrow
    ax.annotate('', xy=(hbt, -0.05), xytext=(-hbt, -0.05),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(0, -0.1, f'b = {b:.3f} m',
            ha='center', va='top', fontsize=10, fontweight='bold')
    # Water depth
    ax.annotate('', xy=(hbt + so_t + 0.15, y), xytext=(hbt + so_t + 0.15, 0),
                arrowprops=dict(arrowstyle='<->', color='#1f77b4', lw=1.5))
    ax.text(hbt + so_t + 0.22, y / 2, f'y = {y:.3f} m',
            va='center', fontsize=9, color='#1f77b4')
    # Total depth
    ax.annotate('', xy=(hbt + so_t + 0.32, Yt), xytext=(hbt + so_t + 0.32, 0),
                arrowprops=dict(arrowstyle='<->', color='#d62728', lw=1.5))
    ax.text(hbt + so_t + 0.39, Yt / 2, f'Y = {Yt:.3f} m',
            va='center', fontsize=9, color='#d62728')
    ax.text(hbt + so_w * 0.6, y * 0.4, f'z={z}\n(45°)',
            fontsize=8, color='gray', ha='center', fontstyle='italic')

    # Info box
    info = (f"V = {t['V']:.3f} m/s  {'✅' if ok else '⚠️'}\n"
            f"A = {t['A']:.3f} m²\nR = {t['R']:.3f} m")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=9,
            va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlim(-(hbt + so_t) - 0.5, (hbt + so_t) + 0.6)
    ax.set_ylim(-0.25, Yt + 0.25)
    ax.set_xlabel('Width (m)', **FONT_LABEL)
    ax.set_ylabel('Depth (m)', **FONT_LABEL)
    ax.set_title(f'(B) Trapezoidal Channel  (z = {z}, 45°)\nb = {b:.3f} m  |  y = {y:.3f} m  |  Y_total = {Yt:.3f} m',
                 **FONT_TITLE)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
