"""
training.py
===========
Training loop functions for the ANN and PINN models.

Both use the Adam optimiser with a step learning rate scheduler.
Training histories are returned as dictionaries for downstream plotting.
"""

import time
import torch
import torch.optim as optim


def train_ann(model, loader, epochs=5000, lr=1e-3,
              lr_step=2000, lr_gamma=0.5,
              print_every=500, device="cpu"):
    """
    Train the ANN on FDM reference data.

    Optimiser : Adam with step LR decay
    Loss      : MSE against FDM temperature labels

    Parameters
    ----------
    model       : ANN instance
    loader      : DataLoader — yields (X_batch, T_batch)
    epochs      : int   — number of training epochs
    lr          : float — initial Adam learning rate
    lr_step     : int   — epoch interval for LR decay
    lr_gamma    : float — multiplicative LR decay factor
    print_every : int   — logging interval (epochs)
    device      : str   — 'cpu' or 'cuda'

    Returns
    -------
    history : dict
        history['loss'] : list[float] — per-epoch average MSE loss
    """
    model.to(device).train()
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimiser,
                                           step_size=lr_step,
                                           gamma=lr_gamma)
    history = {"loss": []}
    t0 = time.time()

    print(f"Training ANN | epochs={epochs} | lr={lr} | params={model.n_params():,}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_b, T_b in loader:
            X_b, T_b = X_b.to(device), T_b.to(device)
            optimiser.zero_grad()
            loss = model.compute_loss(X_b, T_b)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        history["loss"].append(avg)

        if epoch % print_every == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:6d} | loss = {avg:.4e} | "
                  f"lr = {scheduler.get_last_lr()[0]:.2e} | "
                  f"elapsed = {elapsed:.1f} s")

    print(f"\nANN training complete | final loss = {history['loss'][-1]:.4e} | "
          f"total time = {time.time()-t0:.1f} s")
    return history


def train_pinn(model, col, epochs=10000, lr=1e-3,
               lr_step=3000, lr_gamma=0.5,
               print_every=1000, device="cpu"):
    """
    Train the PINN using physics constraints only (no FDM data).

    Optimiser : Adam with step LR decay
    Loss      : λ_pde·L_pde + λ_bc·L_bc + λ_ic·L_ic

    Parameters
    ----------
    model       : PINN instance
    col         : dict — collocation points from dataset.sample_collocation_points()
    epochs      : int   — number of training epochs
    lr          : float — initial Adam learning rate
    lr_step     : int   — epoch interval for LR decay
    lr_gamma    : float — multiplicative LR decay factor
    print_every : int   — logging interval (epochs)
    device      : str   — 'cpu' or 'cuda'

    Returns
    -------
    history : dict
        history['total'] : list[float] — weighted composite loss per epoch
        history['pde']   : list[float] — unweighted PDE loss per epoch
        history['bc']    : list[float] — unweighted BC loss per epoch
        history['ic']    : list[float] — unweighted IC loss per epoch
    """
    model.to(device).train()
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimiser,
                                           step_size=lr_step,
                                           gamma=lr_gamma)
    history = {"total": [], "pde": [], "bc": [], "ic": []}
    t0 = time.time()

    print(f"Training PINN | epochs={epochs} | lr={lr} | params={model.n_params():,}")
    print(f"  λ_pde={model.lambda_pde} | λ_bc={model.lambda_bc} | "
          f"λ_ic={model.lambda_ic}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        optimiser.zero_grad()
        total, L_pde, L_bc, L_ic = model.compute_loss(col, device)
        total.backward()
        optimiser.step()
        scheduler.step()

        history["total"].append(total.item())
        history["pde"].append(L_pde)
        history["bc"].append(L_bc)
        history["ic"].append(L_ic)

        if epoch % print_every == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:6d} | total={total.item():.4e} | "
                  f"pde={L_pde:.4e} | bc={L_bc:.4e} | ic={L_ic:.4e} | "
                  f"{elapsed:.1f} s")

    print(f"\nPINN training complete | final loss = {history['total'][-1]:.4e} | "
          f"total time = {time.time()-t0:.1f} s")
    return history
