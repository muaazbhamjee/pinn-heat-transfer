"""
training.py
===========
Training loop functions for the ANN and PINN models.
Both track training and validation loss for monitoring overfitting.

ANN  — validation loss is MSE on the held-out validation data split
PINN — validation loss is the composite physics loss on separate
       validation collocation points (no data involved)
"""

import time
import torch
import torch.optim as optim


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
    val_loader   : DataLoader — validation split (monitored but not trained on)
    epochs       : int   — number of training epochs
    lr           : float — initial Adam learning rate
    lr_step      : int   — epoch interval for LR decay
    lr_gamma     : float — multiplicative LR decay factor
    print_every  : int   — logging interval (epochs)
    device       : str   — 'cpu' or 'cuda'

    Returns
    -------
    history : dict
        history['train_loss'] : list[float] — per-epoch mean training MSE
        history['val_loss']   : list[float] — per-epoch mean validation MSE
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

        # ── Training pass ─────────────────────────────────────────────────────
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

        # ── Validation pass ───────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for X_v, T_v in val_loader:
                X_v, T_v = X_v.to(device), T_v.to(device)
                val_loss += model.compute_loss(X_v, T_v).item()
            val_avg = val_loss / len(val_loader)

        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_avg)

        if epoch % print_every == 0:
            print(f"  epoch {epoch:6d} | "
                  f"train = {train_avg:.4e} | "
                  f"val = {val_avg:.4e} | "
                  f"lr = {scheduler.get_last_lr()[0]:.2e} | "
                  f"{time.time()-t0:.1f} s")

    print(f"\nANN done | train = {history['train_loss'][-1]:.4e} | "
          f"val = {history['val_loss'][-1]:.4e} | "
          f"time = {time.time()-t0:.1f} s")
    return history


def train_pinn(model, col, col_val,
               epochs=10000, lr=1e-3,
               lr_step=3000, lr_gamma=0.5,
               print_every=1000, device="cpu"):
    """
    Train the PINN using physics constraints with validation monitoring.

    Training loss  — composite physics loss on training collocation points
    Validation loss — composite physics loss on separate validation
                      collocation points (same formulation, different points)

    Parameters
    ----------
    model    : PINN instance
    col      : dict — training collocation points (from sample_collocation_points)
    col_val  : dict — validation collocation points
    epochs   : int   — number of training epochs
    lr       : float — initial Adam learning rate
    lr_step  : int   — epoch interval for LR decay
    lr_gamma : float — multiplicative LR decay factor
    print_every : int — logging interval
    device   : str

    Returns
    -------
    history : dict
        history['train_total']  : list[float]
        history['train_pde']    : list[float]
        history['train_bc']     : list[float]
        history['train_ic']     : list[float]
        history['val_total']    : list[float]
        history['val_pde']      : list[float]
        history['val_bc']       : list[float]
        history['val_ic']       : list[float]
    """
    model.to(device).train()
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimiser,
                                           step_size=lr_step,
                                           gamma=lr_gamma)
    history = {
        "train_total": [], "train_pde": [], "train_bc": [], "train_ic": [],
        "val_total":   [], "val_pde":   [], "val_bc":   [], "val_ic":   [],
    }
    t0 = time.time()

    print(f"Training PINN | epochs={epochs} | lr={lr} | params={model.n_params():,}")
    print(f"  λ_pde={model.lambda_pde} | λ_bc={model.lambda_bc} | "
          f"λ_ic={model.lambda_ic}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        optimiser.zero_grad()
        total, L_pde, L_bc, L_ic = model.compute_loss(col, device)
        total.backward()
        optimiser.step()
        scheduler.step()

        history["train_total"].append(total.item())
        history["train_pde"].append(L_pde)
        history["train_bc"].append(L_bc)
        history["train_ic"].append(L_ic)

        # ── Validation pass ───────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            v_total, v_pde, v_bc, v_ic = model.compute_loss(col_val, device)

        history["val_total"].append(v_total.item())
        history["val_pde"].append(v_pde)
        history["val_bc"].append(v_bc)
        history["val_ic"].append(v_ic)

        if epoch % print_every == 0:
            print(f"  epoch {epoch:6d} | "
                  f"train={total.item():.4e} val={v_total.item():.4e} | "
                  f"pde={L_pde:.4e} bc={L_bc:.4e} ic={L_ic:.4e} | "
                  f"{time.time()-t0:.1f} s")

    print(f"\nPINN done | train = {history['train_total'][-1]:.4e} | "
          f"val = {history['val_total'][-1]:.4e} | "
          f"time = {time.time()-t0:.1f} s")
    return history
