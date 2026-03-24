"""
evaluation.py
=============
Model evaluation, benchmarking, and visualisation utilities.

Functions
---------
predict_field          — predict T(x, y) at a given time for a given material
evaluate_model         — compute predictions and error metrics vs FDM and Exact
plot_loss_curves       — training loss comparison (ANN vs PINN)
plot_field_comparison  — four-panel field plot: Exact | FDM | ANN | PINN
plot_centre_temperature — centre temperature over time for all methods
print_metrics_table    — formatted error metric summary
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from .config import (
    XMAX, YMAX, T_MAX,
    NX, NY,
    T_BOUNDARY, T_INITIAL, T_SAFE, DELTA_T,
)
from .dataset import normalise_inputs
from .fourier import T_fourier_grid, T_fourier_centre_history


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_field(model, props, t_eval, x_arr, y_arr, device="cpu"):
    """
    Predict temperature field T(x, y) at a single evaluation time.

    Parameters
    ----------
    model  : ANN or PINN instance (trained)
    props  : dict with keys 'rho', 'cp', 'k'
    t_eval : float — evaluation time [s]
    x_arr  : ndarray (nx,) — x-coordinates [m]
    y_arr  : ndarray (ny,) — y-coordinates [m]
    device : str

    Returns
    -------
    T_pred : ndarray (nx, ny) — predicted temperature field [K]
    """
    model.eval()
    rho, cp, k = props["rho"], props["cp"], props["k"]
    X_g, Y_g   = np.meshgrid(x_arr, y_arr, indexing="ij")
    x_f, y_f   = X_g.flatten(), Y_g.flatten()
    N = len(x_f)

    xn, yn, tn, rn, cn, kn = normalise_inputs(
        x_f, y_f,
        np.full(N, t_eval),
        np.full(N, rho),
        np.full(N, cp),
        np.full(N, k),
    )

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        T_pred = model.forward(to_t(xn), to_t(yn), to_t(tn),
                                to_t(rn),  to_t(cn),  to_t(kn))

    return T_pred.cpu().numpy().flatten().reshape(X_g.shape)


def predict_centre_history(model, props, t_arr, device="cpu",
                            x_c=None, y_c=None):
    """
    Predict centre temperature over the full time array.

    Parameters
    ----------
    model  : ANN or PINN instance (trained)
    props  : dict with keys 'rho', 'cp', 'k'
    t_arr  : ndarray (nt,) — time array [s]
    x_c, y_c : float — centre coordinates [m] (default: domain centre)

    Returns
    -------
    T_c : ndarray (nt,) — centre temperature [K]
    """
    if x_c is None:
        x_c = XMAX / 2.0
    if y_c is None:
        y_c = YMAX / 2.0

    model.eval()
    rho, cp, k = props["rho"], props["cp"], props["k"]
    T_c = []

    with torch.no_grad():
        for ti in t_arr:
            xn, yn, tn, rn, cn, kn = normalise_inputs(
                np.array([x_c]), np.array([y_c]), np.array([ti]),
                np.array([rho]),  np.array([cp]),  np.array([k]),
            )
            def to_t(v):
                return torch.tensor([[float(v)]], dtype=torch.float32).to(device)
            T_c.append(
                model.forward(to_t(xn[0]), to_t(yn[0]), to_t(tn[0]),
                               to_t(rn[0]),  to_t(cn[0]),  to_t(kn[0])).item()
            )
    return np.array(T_c)


# ── Benchmarking ──────────────────────────────────────────────────────────────

def evaluate_model(model, meat_name, props, fdm_data,
                   t_idx=-1, device="cpu"):
    """
    Full evaluation of a model against FDM and Exact references.

    Parameters
    ----------
    model     : ANN or PINN instance
    meat_name : str — label for printing
    props     : dict with 'rho', 'cp', 'k'
    fdm_data  : dict with 'u' (nx,ny,nt), 'x', 'y', 't'
    t_idx     : int — time index to evaluate (default: -1 = final step)
    device    : str

    Returns
    -------
    results : dict
        'T_pred'  : ndarray (nx, ny)
        'T_fdm'   : ndarray (nx, ny)
        'T_exact' : ndarray (nx, ny)
        't_eval'  : float
        'metrics' : dict — mae_fdm, rmse_fdm, mae_exact, rmse_exact [K]
    """
    rho, cp, k = props["rho"], props["cp"], props["k"]
    alpha  = k / (rho * cp)
    x_arr  = fdm_data["x"]
    y_arr  = fdm_data["y"]
    t_arr  = fdm_data["t"]
    t_eval = t_arr[t_idx]

    T_pred  = predict_field(model, props, t_eval, x_arr, y_arr, device)
    T_fdm   = fdm_data["u"][:, :, t_idx]
    T_exact = T_fourier_grid(x_arr, y_arr, t_eval, alpha)

    mae_fdm    = float(np.mean(np.abs(T_pred - T_fdm)))
    rmse_fdm   = float(np.sqrt(np.mean((T_pred - T_fdm)**2)))
    mae_exact  = float(np.mean(np.abs(T_pred - T_exact)))
    rmse_exact = float(np.sqrt(np.mean((T_pred - T_exact)**2)))

    model_name = model.__class__.__name__
    print(f"{model_name:6s} | {meat_name:8s} | t = {t_eval:.0f} s")
    print(f"  MAE  vs FDM  : {mae_fdm:.4f} K   RMSE: {rmse_fdm:.4f} K")
    print(f"  MAE  vs Exact: {mae_exact:.4f} K   RMSE: {rmse_exact:.4f} K")

    return {
        "T_pred":  T_pred,
        "T_fdm":   T_fdm,
        "T_exact": T_exact,
        "t_eval":  t_eval,
        "metrics": {
            "mae_fdm":    mae_fdm,
            "rmse_fdm":   rmse_fdm,
            "mae_exact":  mae_exact,
            "rmse_exact": rmse_exact,
        },
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_loss_curves(loss_ann, loss_pinn,
                     lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0):
    """
    Side-by-side training loss curves for ANN and PINN.

    Parameters
    ----------
    loss_ann  : dict with key 'loss'
    loss_pinn : dict with keys 'total', 'pde', 'bc', 'ic'
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].semilogy(loss_ann["loss"], color="steelblue", linewidth=1.5)
    axes[0].set_title("ANN Training Loss (MSE on FDM data)", fontsize=11)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(loss_pinn["total"], "k",  linewidth=2.0, label="Total")
    axes[1].semilogy(loss_pinn["pde"],   "--", linewidth=1.2,
                     label=f"PDE  (λ={lambda_pde})")
    axes[1].semilogy(loss_pinn["bc"],    "--", linewidth=1.2,
                     label=f"BC   (λ={lambda_bc})")
    axes[1].semilogy(loss_pinn["ic"],    "--", linewidth=1.2,
                     label=f"IC   (λ={lambda_ic})")
    axes[1].set_title("PINN Training Loss (physics only)", fontsize=11)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Loss Curves", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_field_comparison(meat_name, r_ann, r_pinn, x_arr, y_arr):
    """
    Four-panel temperature field comparison:
    Exact (Fourier) | FDM | ANN | PINN

    Parameters
    ----------
    meat_name : str
    r_ann     : dict — output of evaluate_model for ANN
    r_pinn    : dict — output of evaluate_model for PINN
    x_arr, y_arr : ndarray — coordinate arrays [m]
    """
    fields = [r_ann["T_exact"], r_ann["T_fdm"],
               r_ann["T_pred"],  r_pinn["T_pred"]]
    labels = ["Exact (Fourier)", "FDM", "ANN", "PINN"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, label, field in zip(axes, labels, fields):
        im = ax.contourf(y_arr * 100, x_arr * 100, field, 30,
                          cmap="hot", vmin=T_INITIAL, vmax=T_BOUNDARY)
        plt.colorbar(im, ax=ax, label="T [K]")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("y [cm]")
        ax.set_ylabel("x [cm]")

    plt.suptitle(
        f"{meat_name.capitalize()} — Temperature Field "
        f"at t = {r_ann['t_eval']:.0f} s",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_centre_temperature(meat_name, props, fdm_data,
                             models_dict, device="cpu"):
    """
    Centre temperature vs time for all methods.

    The centre point (x = Lx/2, y = Ly/2) is the last to reach
    cooking temperature — the critical engineering quantity.

    Parameters
    ----------
    meat_name   : str
    props       : dict with 'rho', 'cp', 'k'
    fdm_data    : dict with 'u', 'x', 'y', 't'
    models_dict : dict {label: model} — e.g. {'ANN': ann, 'PINN': pinn}
    device      : str
    """
    rho, cp, k = props["rho"], props["cp"], props["k"]
    alpha  = k / (rho * cp)
    t_arr  = fdm_data["t"]
    x_arr  = fdm_data["x"]
    y_arr  = fdm_data["y"]

    ix = round((XMAX / 2) / XMAX * (NX - 1))
    iy = round(NY / 2)
    x_c, y_c = x_arr[ix], y_arr[iy]

    T_fdm_c   = fdm_data["u"][ix, iy, :]
    T_exact_c = T_fourier_centre_history(t_arr, alpha, x_c=x_c, y_c=y_c)

    plt.figure(figsize=(10, 5))
    plt.plot(t_arr, T_fdm_c,   "k-",  linewidth=2.5, label="FDM")
    plt.plot(t_arr, T_exact_c, "g--", linewidth=2.0,
             label="Exact (Fourier)", alpha=0.85)

    colours = ["steelblue", "crimson", "darkorange", "purple"]
    for (label, model), colour in zip(models_dict.items(), colours):
        T_c = predict_centre_history(model, props, t_arr, device, x_c, y_c)
        plt.plot(t_arr, T_c, linewidth=1.8, linestyle="-.", color=colour,
                  label=label, alpha=0.9)

    plt.axhline(T_SAFE, color="red", linestyle=":", linewidth=1.5,
                label=f"Safe temp. {T_SAFE-273.15:.0f} °C", alpha=0.7)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Centre Temperature [K]", fontsize=12)
    plt.title(f"Centre Temperature — {meat_name.capitalize()}", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Metrics summary ───────────────────────────────────────────────────────────

def print_metrics_table(eval_results):
    """
    Print a formatted table of error metrics for all evaluated meats and models.

    Parameters
    ----------
    eval_results : dict
        {meat_name: {'ann': results_dict, 'pinn': results_dict}}
        where results_dict is the output of evaluate_model()
    """
    header = (f"{'Meat':<10} {'Model':<6} "
              f"{'MAE vs FDM':>12} {'RMSE vs FDM':>13} {'MAE vs Exact':>13}")
    print(header)
    print("-" * len(header))
    for meat, models in eval_results.items():
        for label, key in [("ANN", "ann"), ("PINN", "pinn")]:
            if key not in models:
                continue
            m = models[key]["metrics"]
            print(
                f"{meat:<10} {label:<6} "
                f"{m['mae_fdm']:>12.4f} K "
                f"{m['rmse_fdm']:>12.4f} K "
                f"{m['mae_exact']:>12.4f} K"
            )
        print()
