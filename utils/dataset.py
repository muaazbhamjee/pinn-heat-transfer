"""
dataset.py
==========
Data loading, input normalisation, train/validation splitting,
and collocation point sampling for the ANN and PINN pipelines.

Data pipeline
-------------
Training meats (beef, chicken, pork) → combined dataset → train/val split
Test meat (lamb)                      → held out entirely, never seen during training

The train/validation split is controlled by VAL_SPLIT in the notebook
hyperparameter block. A typical value is 0.2 (80% train, 20% validation).

The ANN is trained on the training split and monitored on the validation split.
The PINN has no data splits — it is trained purely on physics. However, its
PDE residual is also evaluated on separate validation collocation points.

The test meat (lamb) is only loaded in Section 8 for final evaluation.
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


# ── Combined dataset with train/val split ─────────────────────────────────────

def build_ann_dataloaders(data_dir, meat_names,
                           n_per_meat=5000,
                           val_split=0.2,
                           batch_size=512,
                           device="cpu",
                           seed=42):
    """
    Build train and validation DataLoaders for ANN training.

    Pipeline
    --------
    1. Load .npz data for each training meat
    2. Sample n_per_meat points from each
    3. Combine into one shuffled dataset
    4. Split into train (1 - val_split) and validation (val_split)
    5. Return as two DataLoaders

    Parameters
    ----------
    data_dir    : str or Path — directory containing .npz files
    meat_names  : list[str]  — training meats to combine
    n_per_meat  : int        — number of points sampled per meat before split
    val_split   : float      — fraction of combined data for validation
                               e.g. 0.2 → 80% train, 20% validation
    batch_size  : int        — batch size for training loader
    device      : str        — 'cpu' or 'cuda'
    seed        : int        — random seed for reproducibility

    Returns
    -------
    train_loader : DataLoader — shuffled mini-batch loader for training
    val_loader   : DataLoader — full validation set in a single batch
    n_train      : int        — number of training points
    n_val        : int        — number of validation points
    """
    rng = np.random.default_rng(seed)
    X_list, T_list = [], []

    print("Loading and combining training meat datasets...")
    for meat in meat_names:
        data        = load_meat_data(data_dir, meat)
        n_available = len(data["x"])
        n_sample    = min(n_per_meat, n_available)
        idx         = rng.choice(n_available, n_sample, replace=False)

        xn, yn, tn, rn, cn, kn = normalise_inputs(
            data["x"][idx], data["y"][idx], data["t"][idx],
            data["rho"][idx], data["cp"][idx], data["k"][idx],
        )
        X_list.append(np.column_stack([xn, yn, tn, rn, cn, kn]))
        T_list.append(data["T"][idx])
        print(f"  {meat:8s}: {n_sample:,} points sampled")

    # ── Combine and shuffle ───────────────────────────────────────────────────
    X_all = np.vstack(X_list).astype(np.float32)
    T_all = np.concatenate(T_list).astype(np.float32)

    shuffle_idx = rng.permutation(len(X_all))
    X_all = X_all[shuffle_idx]
    T_all = T_all[shuffle_idx]

    # ── Train / validation split ──────────────────────────────────────────────
    n_total = len(X_all)
    n_val   = int(np.floor(val_split * n_total))
    n_train = n_total - n_val

    X_train, T_train = X_all[:n_train], T_all[:n_train]
    X_val,   T_val   = X_all[n_train:], T_all[n_train:]

    # ── Tensors and DataLoaders ───────────────────────────────────────────────
    def to_tensors(X, T):
        return (torch.tensor(X).to(device),
                torch.tensor(T).unsqueeze(1).to(device))

    X_tr_t, T_tr_t = to_tensors(X_train, T_train)
    X_vl_t, T_vl_t = to_tensors(X_val,   T_val)

    train_loader = DataLoader(TensorDataset(X_tr_t, T_tr_t),
                               batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl_t, T_vl_t),
                               batch_size=len(X_val), shuffle=False)

    print(f"\n  Combined : {n_total:,} points total")
    print(f"  Train    : {n_train:,}  ({100*(1-val_split):.0f}%)")
    print(f"  Val      : {n_val:,}  ({100*val_split:.0f}%)")
    print(f"  Test     : lamb — held out, loaded only in Section 8")

    return train_loader, val_loader, n_train, n_val


# ── PINN collocation points ───────────────────────────────────────────────────

def sample_collocation_points(N_col=8000, N_bc=600, N_ic=600,
                               N_col_val=2000, N_bc_val=200, N_ic_val=200,
                               seed=42):
    """
    Sample training and validation physics collocation points for the PINN.

    The PINN does not use FDM data. It is trained by enforcing:
      - The PDE residual at interior collocation points  (col['pde'])
      - Dirichlet BCs on all four walls                  (col['bc'])
      - The initial condition at t = 0                   (col['ic'])

    A separate, non-overlapping validation set is returned so the PDE
    residual can be monitored on unseen points during training — this is
    the PINN equivalent of a validation loss.

    Material properties are sampled uniformly across the training meat
    range, making the PINN a parametric surrogate.

    Parameters
    ----------
    N_col, N_bc, N_ic             : int — training point counts
    N_col_val, N_bc_val, N_ic_val : int — validation point counts
    seed                          : int — random seed

    Returns
    -------
    col     : dict — training collocation points
    col_val : dict — validation collocation points
        Each dict has keys 'pde', 'bc', 'ic' — raw (unnormalised)
        tuples of (x, y, t, rho, cp, k) arrays.
    """
    rng = np.random.default_rng(seed)

    def rand_props(N):
        return (
            rng.uniform(RHO_MIN, RHO_MAX, N),
            rng.uniform(CP_MIN,  CP_MAX,  N),
            rng.uniform(K_MIN,   K_MAX,   N),
        )

    def _sample(n_col, n_bc, n_ic):
        x_c = rng.uniform(0, XMAX,  n_col)
        y_c = rng.uniform(0, YMAX,  n_col)
        t_c = rng.uniform(0, T_MAX, n_col)
        rho_c, cp_c, k_c = rand_props(n_col)

        Nw = n_bc // 4
        x_bc = np.concatenate([np.zeros(Nw), np.full(Nw, XMAX),
                                rng.uniform(0, XMAX, Nw),
                                rng.uniform(0, XMAX, Nw)])
        y_bc = np.concatenate([rng.uniform(0, YMAX, Nw),
                                rng.uniform(0, YMAX, Nw),
                                np.zeros(Nw), np.full(Nw, YMAX)])
        t_bc = rng.uniform(0, T_MAX, len(x_bc))
        rho_bc, cp_bc, k_bc = rand_props(len(x_bc))

        x_ic  = rng.uniform(0, XMAX, n_ic)
        y_ic  = rng.uniform(0, YMAX, n_ic)
        t_ic  = np.zeros(n_ic)
        rho_ic, cp_ic, k_ic = rand_props(n_ic)

        return {
            "pde": (x_c,  y_c,  t_c,  rho_c,  cp_c,  k_c),
            "bc":  (x_bc, y_bc, t_bc, rho_bc, cp_bc, k_bc),
            "ic":  (x_ic, y_ic, t_ic, rho_ic, cp_ic, k_ic),
        }

    col     = _sample(N_col,     N_bc,     N_ic)
    col_val = _sample(N_col_val, N_bc_val, N_ic_val)

    print(f"PINN collocation points:")
    print(f"  Train — PDE: {N_col:,} | BC: {N_bc} | IC: {N_ic}")
    print(f"  Val   — PDE: {N_col_val:,} | BC: {N_bc_val} | IC: {N_ic_val}")

    return col, col_val
