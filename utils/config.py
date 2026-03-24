"""
config.py
=========
Central configuration for the PINN heat transfer project.
All domain parameters, thermal conditions, and material properties
are defined here. Import from this module rather than hardcoding
values in other scripts.
"""

# ── Domain ────────────────────────────────────────────────────────────────────
XMAX = 0.05     # [m] domain width  (meat thickness)
YMAX = 0.150    # [m] domain height (meat length)
NX   = 26       # grid points in x
NY   = 76       # grid points in y
NT   = 150      # number of time steps
DT   = 6        # [s] time step size
T_MAX = (NT - 1) * DT  # [s] total simulation time = 894 s

# ── Thermal conditions ────────────────────────────────────────────────────────
T_BOUNDARY = 473.15  # [K] Weber grill surface temperature (~200 °C)
T_INITIAL  = 288.15  # [K] fridge temperature              (~ 15 °C)
DELTA_T    = T_BOUNDARY - T_INITIAL  # 185 K — reference scale for loss normalisation

# ── Training meat properties ──────────────────────────────────────────────────
#    rho [kg/m³] | cp [J/kg·K] | k [W/m·K]
MEAT_PROPERTIES = {
    "beef": {
        "rho": 1090.0,
        "cp":  2720.0,
        "k":   0.471,
    },
    "chicken": {
        "rho": 1050.0,
        "cp":  3300.0,
        "k":   0.412,
    },
    "pork": {
        "rho": 1060.0,
        "cp":  2800.0,
        "k":   0.480,
    },
}

# ── Test meat (unseen during training) ───────────────────────────────────────
TEST_MEAT = {
    "lamb": {
        "rho": 1070.0,
        "cp":  2900.0,
        "k":   0.450,
    }
}

# ── Normalisation bounds (min-max over training meats) ────────────────────────
RHO_MIN, RHO_MAX = 1050.0, 1090.0
CP_MIN,  CP_MAX  = 2720.0, 3300.0
K_MIN,   K_MAX   = 0.412,  0.480

# ── Fourier series truncation ─────────────────────────────────────────────────
FOURIER_M = 20  # number of x-mode terms
FOURIER_N = 20  # number of y-mode terms

# ── Centre point (for time-history plots) ────────────────────────────────────
X_CENTRE = 0.025  # [m]
Y_CENTRE = YMAX / 2  # [m]

# ── Safe internal temperature ─────────────────────────────────────────────────
T_SAFE = 343.15  # [K] 70 °C — minimum safe centre temperature for cooked meat
