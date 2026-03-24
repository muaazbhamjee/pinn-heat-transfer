"""
training.py
===========
Training loop functions for the ANN and PINN models.

PINN training strategy (two-phase, following Raissi et al.)
------------------------------------------------------------
Phase 1 — Adam optimiser
    Fast, robust convergence from random initialisation toward the
    solution manifold. Typically 10,000–15,000 epochs.

Phase 2 — L-BFGS optimiser
    Second-order quasi-Newton method. Converges precisely once Adam
    has brought the solution into a reasonable neighbourhood.
    This is the phase that closes the gap between PINN and ANN accuracy.

Collocation resampling
----------------------
Collocation points are resampled every `resample_every` epochs during
Adam training. This prevents the PINN from overfitting to a fixed set
of random points — the diverging train/val loss seen with fixed points
is a symptom of this overfitting.
"""

import time
import numpy as np
import torch
import torch.optim as optim

from .dataset import sample_collocation_points


# ── ANN ───────────────────────────────────────────────────────────────────────

def train_ann(model, train_loader, val_loader,
              epochs=5000, lr=1e-3,
              lr_step=2000, lr_gamma=0.5,
              print_every=500, device="cpu"):
    """
    Train the ANN on FDM reference data with validation monitoring.

    Parameters
    ----------
    model        : ANN instance
    train_loader : DataLoader — training split
    val_loader   : DataLoader — validation split
    epochs       : int   — number of training epochs
    lr           : float — initial Adam learning rate
    lr_step      : int   — epoch interval for LR decay
    lr_gamma     : float — multiplicative LR decay factor
    print_every  : int   — logging interval
    device       : str

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss'
    """
    model.to(device).train()
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimiser,
                                           step_size=lr_step,
                                           gamma=lr_gamma)
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    print(f"Training ANN | epochs={epochs} | lr={lr} | params={model.n_params():,}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_b, T_b in train_loader:
            X_b, T_b = X_b.to(device), T_b.to(device)
            optimiser.zero_grad()
            loss = model.compute_loss(X_b, T_b)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        train_avg = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = sum(model.compute_loss(X_v.to(device), T_v.to(device)).item()
                           for X_v, T_v in val_loader) / len(val_loader)

        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_loss)

        if epoch % print_every == 0:
            print(f"  epoch {epoch:6d} | train={train_avg:.4e} | val={val_loss:.4e} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f} s")

    print(f"\nANN done | train={history['train_loss'][-1]:.4e} | "
          f"val={history['val_loss'][-1]:.4e} | time={time.time()-t0:.1f} s")
    return history


# ── PINN ──────────────────────────────────────────────────────────────────────

def train_pinn(model, col, col_val,
               epochs_adam=10000, lr=1e-3,
               epochs_lbfgs=500,
               lr_step=3000, lr_gamma=0.5,
               resample_every=2000,
               print_every=1000, device="cpu"):
    """
    Train the PINN using a two-phase strategy (Adam → L-BFGS).

    Phase 1 — Adam with periodic collocation resampling
        Robust global convergence. Collocation points are resampled every
        `resample_every` epochs to prevent overfitting to fixed points.

    Phase 2 — L-BFGS
        Precise local convergence. Follows Raissi et al. (2019) training
        protocol. Runs after Adam has found a good neighbourhood.

    Parameters
    ----------
    model          : PINN instance
    col            : dict — initial training collocation points
    col_val        : dict — validation collocation points (fixed throughout)
    epochs_adam    : int   — Adam training epochs
    lr             : float — Adam learning rate
    epochs_lbfgs   : int   — L-BFGS iterations (set 0 to skip)
    lr_step        : int   — Adam LR decay interval
    lr_gamma       : float — Adam LR decay factor
    resample_every : int   — resample collocation points every N Adam epochs
                             (set 0 to disable resampling)
    print_every    : int   — logging interval
    device         : str

    Returns
    -------
    history : dict
        history['train_total'], ['train_pde'], ['train_bc'], ['train_ic']
        history['val_total'],   ['val_pde'],   ['val_bc'],   ['val_ic']
        history['phase']       : list[str] — 'adam' or 'lbfgs' per epoch
    """
    model.to(device).train()
    history = {
        "train_total": [], "train_pde": [], "train_bc": [], "train_ic": [],
        "val_total":   [], "val_pde":   [], "val_bc":   [], "val_ic":   [],
        "phase":       [],
    }
    t0 = time.time()

    print(f"Training PINN | Adam={epochs_adam} epochs + L-BFGS={epochs_lbfgs} iters | "
          f"params={model.n_params():,}")
    print(f"  λ_pde={model.lambda_pde} | λ_bc={model.lambda_bc} | λ_ic={model.lambda_ic}")
    print(f"  Collocation resampling every {resample_every} epochs")
    print("-" * 65)

    # ── Phase 1: Adam ─────────────────────────────────────────────────────────
    adam = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(adam, step_size=lr_step, gamma=lr_gamma)

    # Track collocation seed so resampling is deterministic and reproducible.
    # Derived from PyTorch's current RNG state so it respects set_seed() calls.
    col_seed = int(torch.randint(0, 10000, (1,)).item())
    resample_count = 0

    for epoch in range(1, epochs_adam + 1):

        # Resample collocation points periodically
        if resample_every > 0 and epoch > 1 and (epoch - 1) % resample_every == 0:
            resample_count += 1
            col_seed = col_seed + resample_count  # deterministic progression
            col, _ = sample_collocation_points(
                N_col=len(col["pde"][0]),
                N_bc=len(col["bc"][0]),
                N_ic=len(col["ic"][0]),
                N_col_val=1,   # dummy — we keep col_val fixed
                N_bc_val=1,
                N_ic_val=1,
                seed=col_seed,
            )

        model.train()
        adam.zero_grad()
        total, L_pde, L_bc, L_ic = model.compute_loss(col, device)
        total.backward()
        adam.step()
        scheduler.step()

        model.eval()
        v_total, v_pde, v_bc, v_ic = model.compute_loss(col_val, device)

        history["train_total"].append(total.item())
        history["train_pde"].append(L_pde)
        history["train_bc"].append(L_bc)
        history["train_ic"].append(L_ic)
        history["val_total"].append(v_total.item())
        history["val_pde"].append(v_pde)
        history["val_bc"].append(v_bc)
        history["val_ic"].append(v_ic)
        history["phase"].append("adam")

        if epoch % print_every == 0:
            print(f"  [Adam] epoch {epoch:6d} | "
                  f"train={total.item():.4e} val={v_total.item():.4e} | "
                  f"pde={L_pde:.4e} bc={L_bc:.4e} ic={L_ic:.4e} | "
                  f"{time.time()-t0:.1f} s")

    print(f"\n  Adam done | train={history['train_total'][-1]:.4e} | "
          f"val={history['val_total'][-1]:.4e}")

    # ── Phase 2: L-BFGS ───────────────────────────────────────────────────────
    if epochs_lbfgs > 0:
        print(f"\n  Starting L-BFGS ({epochs_lbfgs} iterations)...")
        lbfgs = optim.LBFGS(model.parameters(),
                              max_iter=epochs_lbfgs,
                              tolerance_grad=1e-9,
                              tolerance_change=1e-11,
                              history_size=50,
                              line_search_fn="strong_wolfe")

        lbfgs_history = []

        def closure():
            lbfgs.zero_grad()
            total, L_pde, L_bc, L_ic = model.compute_loss(col, device)
            total.backward()
            lbfgs_history.append((total.item(), L_pde, L_bc, L_ic))
            return total

        model.train()
        lbfgs.step(closure)

        # Append L-BFGS iterations to history
        for total_v, L_pde, L_bc, L_ic in lbfgs_history:
            model.eval()
            v_total, v_pde, v_bc, v_ic = model.compute_loss(col_val, device)
            history["train_total"].append(total_v)
            history["train_pde"].append(L_pde)
            history["train_bc"].append(L_bc)
            history["train_ic"].append(L_ic)
            history["val_total"].append(v_total.item())
            history["val_pde"].append(v_pde)
            history["val_bc"].append(v_bc)
            history["val_ic"].append(v_ic)
            history["phase"].append("lbfgs")

        print(f"  L-BFGS done | train={history['train_total'][-1]:.4e} | "
              f"val={history['val_total'][-1]:.4e}")

    print(f"\nPINN training complete | total time={time.time()-t0:.1f} s")
    return history
