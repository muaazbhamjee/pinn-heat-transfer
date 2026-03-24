"""
fourier.py
==========
Exact Fourier series solution for 2D unsteady heat conduction on a
rectangular domain with uniform Dirichlet BCs and uniform initial condition.

Reference
---------
Cengel, Y. A. & Ghajar, A. J. (2015). Heat and Mass Transfer (5th ed.).
McGraw-Hill. Chapter 4 — Two-Dimensional Steady-State Heat Conduction.
"""

import numpy as np
from .config import (
    XMAX, YMAX, T_BOUNDARY, T_INITIAL,
    FOURIER_M, FOURIER_N,
)


def T_fourier_point(x, y, time, alpha,
                    T_s=T_BOUNDARY, T_i=T_INITIAL,
                    a=XMAX, b=YMAX,
                    m_end=FOURIER_M, n_end=FOURIER_N):
    """
    Exact Fourier series temperature at a single point (x, y) and time t.

    Parameters
    ----------
    x, y   : float — spatial coordinates [m]
    time   : float — evaluation time [s]
    alpha  : float — thermal diffusivity [m²/s]
    T_s    : float — surface (boundary) temperature [K]
    T_i    : float — initial temperature [K]
    a, b   : float — domain dimensions [m]
    m_end, n_end : int — series truncation (even modes contribute zero)

    Returns
    -------
    T : float — temperature [K]
    """
    total = 0.0
    for n in range(1, n_end):
        for m in range(1, m_end):
            A_mn = (
                4.0 * (T_i - T_s) / (m * n * np.pi**2)
                * (1.0 - np.cos(m * np.pi))
                * (1.0 - np.cos(n * np.pi))
            )
            spatial  = np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)
            temporal = np.exp(
                -alpha * ((m * np.pi / a)**2 + (n * np.pi / b)**2) * time
            )
            total += A_mn * spatial * temporal
    return T_s + total


def T_fourier_grid(x_arr, y_arr, time, alpha,
                   T_s=T_BOUNDARY, T_i=T_INITIAL,
                   a=XMAX, b=YMAX,
                   m_end=FOURIER_M, n_end=FOURIER_N):
    """
    Exact Fourier series temperature over a full 2D grid at a single time.

    Parameters
    ----------
    x_arr : ndarray (nx,) — x-coordinates [m]
    y_arr : ndarray (ny,) — y-coordinates [m]
    time  : float         — evaluation time [s]
    alpha : float         — thermal diffusivity [m²/s]

    Returns
    -------
    T : ndarray (nx, ny) — temperature field [K]
    """
    X, Y  = np.meshgrid(x_arr, y_arr, indexing="ij")
    T_out = np.zeros_like(X, dtype=float)

    for n in range(1, n_end):
        for m in range(1, m_end):
            A_mn = (
                4.0 * (T_i - T_s) / (m * n * np.pi**2)
                * (1.0 - np.cos(m * np.pi))
                * (1.0 - np.cos(n * np.pi))
            )
            T_out += (
                A_mn
                * np.sin(m * np.pi * X / a)
                * np.sin(n * np.pi * Y / b)
                * np.exp(-alpha * ((m * np.pi / a)**2 + (n * np.pi / b)**2) * time)
            )

    return T_s + T_out


def T_fourier_centre_history(t_arr, alpha,
                              T_s=T_BOUNDARY, T_i=T_INITIAL,
                              a=XMAX, b=YMAX,
                              x_c=None, y_c=None,
                              m_end=FOURIER_M, n_end=FOURIER_N):
    """
    Exact centre temperature over a time array.

    Parameters
    ----------
    t_arr : ndarray (nt,) — time array [s]
    alpha : float         — thermal diffusivity [m²/s]
    x_c, y_c : float      — centre coordinates [m] (defaults to domain centre)

    Returns
    -------
    T_c : ndarray (nt,) — centre temperature history [K]
    """
    if x_c is None:
        x_c = a / 2.0
    if y_c is None:
        y_c = b / 2.0

    return np.array([
        T_fourier_point(x_c, y_c, ti, alpha,
                        T_s=T_s, T_i=T_i, a=a, b=b,
                        m_end=m_end, n_end=n_end)
        for ti in t_arr
    ])
