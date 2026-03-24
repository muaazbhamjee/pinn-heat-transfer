"""
dataset.py
==========
Data loading, input normalisation, and collocation point sampling
for the ANN and PINN training pipelines.

The ANN is trained on FDM-generated data loaded from .npz files.
The PINN is trained purely on physics — no FDM data is used.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from .config import (
    XMAX, YMAX, T_MAX,
    RHO_MIN, RHO_MAX,
    CP_MIN,  CP_MAX,
    K_MIN,   K_MAX,
    T_BOUNDARY, T_INITIAL,
)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise_inputs(x, y, t, rho, cp, k):
    """
    Normalise all six inputs to [0, 1].

    Spatial and temporal inputs are scaled by domain extents.
    Material properties are min-max normalised over training meat range.

    Parameters (all ndarray or float)
    ----------
    x, y, t     : spatial [m] and temporal [s] coordinates
    rho, cp, k  : material properties [kg/m³, J/kg·K, W/m·K]

    Returns
    -------
    Six normalised arrays of the same shape as inputs.
    """
    x_n   = x   / XMAX
    y_n   = y   / YMAX
    t_n   = t   / T_MAX
    rho_n = (rho - RHO_MIN) / (RHO_MAX - RHO_MIN + 1e-10)
    cp_n  = (cp  - CP_MIN)  / (CP_MAX  - CP_MIN  + 1e-10)
    k_n   = (k   - K_MIN)   / (K_MAX   - K_MIN   + 1e-10)
    return x_n, y_n, t_n, rho_n, cp_n, k_n


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_meat_data(data_dir, meat_name):
    """
    Load a pre-generated .npz dataset for a single meat type.

    Parameters
    ----------
    data_dir  : str or Path — directory containing .npz files
    meat_name : str         — e.g. 'beef', 'chicken', 'pork', 'lamb'

    Returns
    -------
    data : dict with keys 'x', 'y', 't', 'rho', 'cp', 'k', 'T'
           All values are 1D ndarrays of the same length (one row per point).
    """
    path = Path(data_dir) / f"train_{meat_name}.npz"
    if not path.exists():
        path = Path(data_dir) / f"test_{meat_name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset for '{meat_name}' in {data_dir}.\n"
            f"Expected: train_{meat_name}.npz or test_{meat_name}.npz"
        )
    raw = np.load(path)
    return {key: raw[key] for key in raw.files}


def build_ann_dataloader(data_dir, meat_names,
                          n_per_meat=5000, batch_size=512,
                          device="cpu", seed=42):
    """
    Build a PyTorch DataLoader for ANN training from .npz files.

    Randomly samples n_per_meat points from each meat's dataset,
    normalises inputs, and returns a DataLoader.

    Parameters
    ----------
    data_dir    : str or Path — directory containing .npz files
    meat_names  : list[str]  — training meats to include
    n_per_meat  : int        — number of points sampled per meat
    batch_size  : int        — DataLoader batch size
    device      : str        — 'cpu' or 'cuda'
    seed        : int        — random seed for reproducibility

    Returns
    -------
    loader : DataLoader  — yields (X_batch, T_batch) tensors
    n_total : int        — total number of training points
    """
    rng = np.random.default_rng(seed)
    X_list, T_list = [], []

    for meat in meat_names:
        data = load_meat_data(data_dir, meat)
        n_available = len(data["x"])
        n_sample    = min(n_per_meat, n_available)
        idx         = rng.choice(n_available, n_sample, replace=False)

        xn, yn, tn, rn, cn, kn = normalise_inputs(
            data["x"][idx], data["y"][idx], data["t"][idx],
            data["rho"][idx], data["cp"][idx], data["k"][idx],
        )
        X = np.column_stack([xn, yn, tn, rn, cn, kn])
        T = data["T"][idx]

        X_list.append(X)
        T_list.append(T)
        print(f"  Loaded {meat:8s}: {n_sample:,} points")

    X_all = np.vstack(X_list).astype(np.float32)
    T_all = np.concatenate(T_list).astype(np.float32)

    X_t = torch.tensor(X_all).to(device)
    T_t = torch.tensor(T_all).unsqueeze(1).to(device)

    loader = DataLoader(TensorDataset(X_t, T_t),
                        batch_size=batch_size, shuffle=True)
    print(f"  Total ANN training points: {len(X_all):,}")
    return loader, len(X_all)


# ── PINN collocation points ───────────────────────────────────────────────────

def sample_collocation_points(N_col=8000, N_bc=600, N_ic=600, seed=42):
    """
    Sample physics collocation points for PINN training.

    The PINN does not use FDM data. Instead it is trained by enforcing:
      - The PDE residual at random interior points  (col['pde'])
      - Dirichlet BCs on all four walls             (col['bc'])
      - The initial condition at t = 0              (col['ic'])

    Material properties are sampled uniformly across the training meat
    range, making the PINN a parametric surrogate.

    Parameters
    ----------
    N_col : int — number of PDE collocation points
    N_bc  : int — number of boundary condition points
    N_ic  : int — number of initial condition points
    seed  : int — random seed

    Returns
    -------
    col : dict
        col['pde'] = (x, y, t, rho, cp, k)  — raw (unnormalised) arrays
        col['bc']  = (x, y, t, rho, cp, k)
        col['ic']  = (x, y, t, rho, cp, k)
    """
    rng = np.random.default_rng(seed)

    def rand_props(N):
        return (
            rng.uniform(RHO_MIN, RHO_MAX, N),
            rng.uniform(CP_MIN,  CP_MAX,  N),
            rng.uniform(K_MIN,   K_MAX,   N),
        )

    # ── PDE interior points ───────────────────────────────────────────────────
    x_c = rng.uniform(0,    XMAX,  N_col)
    y_c = rng.uniform(0,    YMAX,  N_col)
    t_c = rng.uniform(0,    T_MAX, N_col)
    rho_c, cp_c, k_c = rand_props(N_col)

    # ── Boundary condition points (four walls) ────────────────────────────────
    Nw = N_bc // 4
    x_bc = np.concatenate([
        np.zeros(Nw),                         # left wall   x = 0
        np.full(Nw, XMAX),                    # right wall  x = XMAX
        rng.uniform(0, XMAX, Nw),             # bottom wall y = 0
        rng.uniform(0, XMAX, Nw),             # top wall    y = YMAX
    ])
    y_bc = np.concatenate([
        rng.uniform(0, YMAX, Nw),
        rng.uniform(0, YMAX, Nw),
        np.zeros(Nw),
        np.full(Nw, YMAX),
    ])
    t_bc = rng.uniform(0, T_MAX, len(x_bc))
    rho_bc, cp_bc, k_bc = rand_props(len(x_bc))

    # ── Initial condition points (t = 0) ──────────────────────────────────────
    x_ic  = rng.uniform(0, XMAX, N_ic)
    y_ic  = rng.uniform(0, YMAX, N_ic)
    t_ic  = np.zeros(N_ic)
    rho_ic, cp_ic, k_ic = rand_props(N_ic)

    col = {
        "pde": (x_c,  y_c,  t_c,  rho_c,  cp_c,  k_c),
        "bc":  (x_bc, y_bc, t_bc, rho_bc, cp_bc, k_bc),
        "ic":  (x_ic, y_ic, t_ic, rho_ic, cp_ic, k_ic),
    }

    print(f"Collocation points — PDE: {N_col} | BC: {len(x_bc)} | IC: {N_ic}")
    return col
