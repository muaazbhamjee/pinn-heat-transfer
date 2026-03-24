"""
generate_dataset.py
===================
INSTRUCTOR TOOL — run this script once to generate the training and test
datasets from the FDM solver. The resulting .npz files are committed to
the repository. Students do not need to run this script.

Usage
-----
    python generate_dataset.py

Output
------
    data/train_beef.npz
    data/train_chicken.npz
    data/train_pork.npz
    data/test_lamb.npz

Each .npz file contains the following arrays:
    x    : (N,) — x-coordinates of interior points [m]
    y    : (N,) — y-coordinates of interior points [m]
    t    : (N,) — time values [s]
    rho  : (N,) — density (constant per meat) [kg/m³]
    cp   : (N,) — specific heat (constant per meat) [J/kg·K]
    k    : (N,) — thermal conductivity (constant per meat) [W/m·K]
    T    : (N,) — temperature at (x, y, t) from FDM solution [K]

The FDM uses an explicit second-order central difference scheme.
Stability criterion: 2r ≤ 0.5, where r = alpha * dt / dx².

Note: Interior points only. Boundary values are enforced analytically
in the PINN loss function and are not included in the dataset.
"""

import numpy as np
from pathlib import Path
from utils.config import (
    NX, NY, NT, DT,
    XMAX, YMAX,
    T_BOUNDARY, T_INITIAL,
    MEAT_PROPERTIES, TEST_MEAT,
)


# ── FDM Solver ────────────────────────────────────────────────────────────────

def solve_fdm(rho, cp, k,
              nx=NX, ny=NY, nt=NT, dt=DT,
              xmax=XMAX, ymax=YMAX,
              T_initial=T_INITIAL, T_boundary=T_BOUNDARY):
    """
    Solve the 2D unsteady heat equation using explicit finite differences.

    Governing PDE
    -------------
    ρ c_p ∂T/∂t = k (∂²T/∂x² + ∂²T/∂y²)

    Discretisation (explicit, 2nd-order central difference)
    --------------------------------------------------------
    T[i,j]^{n+1} = T[i,j]^n
        + α Δt/Δx² (T[i+1,j]^n - 2T[i,j]^n + T[i-1,j]^n)
        + α Δt/Δy² (T[i,j+1]^n - 2T[i,j]^n + T[i,j-1]^n)

    Stability: 2r ≤ 0.5 where r = α Δt / Δx²

    Parameters
    ----------
    rho, cp, k : float — material properties [kg/m³, J/kg·K, W/m·K]

    Returns
    -------
    u    : ndarray (nx, ny, nt) — full temperature field [K]
    x    : ndarray (nx,)        — x-coordinates [m]
    y    : ndarray (ny,)        — y-coordinates [m]
    t    : ndarray (nt,)        — time array [s]
    """
    alpha = k / (rho * cp)
    dx    = xmax / (nx - 1)
    dy    = ymax / (ny - 1)
    r     = alpha * dt / dx**2

    if 2 * r > 0.5:
        raise ValueError(
            f"Stability violated: 2r = {2*r:.4f} > 0.5. "
            f"Reduce dt or increase spatial resolution."
        )

    x = np.linspace(0, xmax, nx)
    y = np.linspace(0, ymax, ny)
    t = np.arange(nt) * dt

    u = np.full((nx, ny, nt), T_initial, dtype=np.float64)

    # Dirichlet BCs — all four walls at T_boundary for all time
    u[0,  :, :] = T_boundary
    u[-1, :, :] = T_boundary
    u[:,  0, :] = T_boundary
    u[:, -1, :] = T_boundary

    for it in range(nt - 1):
        un = u[:, :, it]
        u[1:-1, 1:-1, it + 1] = (
            un[1:-1, 1:-1]
            + alpha * dt / dx**2 * (un[2:,   1:-1] - 2 * un[1:-1, 1:-1] + un[:-2,  1:-1])
            + alpha * dt / dy**2 * (un[1:-1, 2:]   - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
        )

    return u, x, y, t


# ── Dataset extraction ────────────────────────────────────────────────────────

def extract_dataset(u, x, y, t, rho, cp, k):
    """
    Extract all interior (x, y, t, rho, cp, k, T) points from an FDM solution.

    Boundary nodes are excluded — BCs are enforced analytically in the PINN.

    Returns flat 1D arrays, one entry per spatiotemporal point.
    """
    x_int = x[1:-1]
    y_int = y[1:-1]
    X_g, Y_g = np.meshgrid(x_int, y_int, indexing="ij")
    x_flat = X_g.flatten()
    y_flat = Y_g.flatten()
    n_spatial = len(x_flat)

    x_all   = np.tile(x_flat, len(t))
    y_all   = np.tile(y_flat, len(t))
    t_all   = np.repeat(t, n_spatial)
    rho_all = np.full_like(x_all, rho)
    cp_all  = np.full_like(x_all, cp)
    k_all   = np.full_like(x_all, k)
    T_all   = np.vstack([u[1:-1, 1:-1, it].flatten() for it in range(len(t))]).flatten()

    return x_all, y_all, t_all, rho_all, cp_all, k_all, T_all


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    all_meats = {**MEAT_PROPERTIES, **TEST_MEAT}

    for meat, props in all_meats.items():
        rho, cp, k = props["rho"], props["cp"], props["k"]
        alpha = k / (rho * cp)

        print(f"\nProcessing: {meat.capitalize()}")
        print(f"  rho={rho} kg/m³ | cp={cp} J/kg·K | k={k} W/m·K")
        print(f"  alpha = {alpha:.3e} m²/s")

        u, x, y, t = solve_fdm(rho, cp, k)

        x_d, y_d, t_d, rho_d, cp_d, k_d, T_d = extract_dataset(
            u, x, y, t, rho, cp, k
        )

        prefix = "test" if meat in TEST_MEAT else "train"
        out_path = data_dir / f"{prefix}_{meat}.npz"

        np.savez_compressed(
            out_path,
            x=x_d.astype(np.float32),
            y=y_d.astype(np.float32),
            t=t_d.astype(np.float32),
            rho=rho_d.astype(np.float32),
            cp=cp_d.astype(np.float32),
            k=k_d.astype(np.float32),
            T=T_d.astype(np.float32),
        )

        size_kb = out_path.stat().st_size / 1024
        print(f"  Saved {out_path} | {len(x_d):,} points | {size_kb:.1f} KB")

    print("\nDataset generation complete.")
    print("Commit the data/ directory to the repository.")


if __name__ == "__main__":
    main()
