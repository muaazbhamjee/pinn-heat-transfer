"""
models.py
=========
Neural network architectures for the PINN heat transfer project.

HeatNet  — shared fully-connected backbone (6 inputs → 1 output)
ANN      — data-driven model, MSE loss against FDM data
PINN     — physics-informed model, composite loss via autograd

Both models share identical architecture. The only difference is the
loss function — which is the central pedagogical point of this project.
"""

import torch
import torch.nn as nn

from .config import (
    XMAX, YMAX, T_MAX,
    RHO_MIN, RHO_MAX,
    CP_MIN,  CP_MAX,
    K_MIN,   K_MAX,
    T_BOUNDARY, T_INITIAL, DELTA_T,
)


# ── Shared backbone ───────────────────────────────────────────────────────────

class HeatNet(nn.Module):
    """
    Fully-connected neural network for heat transfer prediction.

    Architecture
    ------------
    Input  : (x̂, ŷ, t̂, ρ̂, ĉ_p, k̂)  — 6 normalised inputs
    Hidden : n_hidden layers of n_neurons neurons each
    Output : T [K]                      — 1 neuron (unnormalised)
    Activ. : tanh — smooth, infinitely differentiable (required for PINN autograd)
    Init   : Xavier (Glorot) — calibrated for tanh to prevent vanishing gradients

    Parameters
    ----------
    n_hidden  : int — number of hidden layers       (default: 5)
    n_neurons : int — neurons per hidden layer       (default: 40)
    """

    def __init__(self, n_hidden=5, n_neurons=40):
        super().__init__()
        self.n_hidden  = n_hidden
        self.n_neurons = n_neurons

        layers = [nn.Linear(6, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)

        self._xavier_init()

    def _xavier_init(self):
        """Xavier/Glorot initialisation — optimised variance for tanh activations."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x_n, y_n, t_n, rho_n, cp_n, k_n):
        """
        Forward pass through the network.

        Parameters (all Tensor of shape (N, 1), normalised to [0, 1])
        ----------
        x_n, y_n, t_n  : normalised spatial and temporal coordinates
        rho_n, cp_n, k_n : normalised material properties

        Returns
        -------
        T : Tensor (N, 1) — predicted temperature [K]
        """
        inp = torch.cat([x_n, y_n, t_n, rho_n, cp_n, k_n], dim=1)
        return self.net(inp)

    def n_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self):
        """Print a concise architecture summary."""
        print(f"{self.__class__.__name__}")
        print(f"  Inputs   : 6  (x, y, t, rho, cp, k)")
        print(f"  Hidden   : {self.n_hidden} layers × {self.n_neurons} neurons")
        print(f"  Output   : 1  (T [K])")
        print(f"  Activation : tanh")
        print(f"  Parameters : {self.n_params():,}")


# ── ANN ───────────────────────────────────────────────────────────────────────

class ANN(HeatNet):
    """
    Data-driven Artificial Neural Network.

    Trained on FDM simulation data using mean squared error loss.
    No physics is embedded — the model learns the input-output mapping
    entirely from data.

    Loss
    ----
    L_ANN = (1/N) Σ (T_θ(x_i) - T_FDM(x_i))²

    Parameters
    ----------
    n_hidden  : int — number of hidden layers
    n_neurons : int — neurons per hidden layer
    """

    def __init__(self, n_hidden=5, n_neurons=40):
        super().__init__(n_hidden, n_neurons)

    def compute_loss(self, X_batch, T_batch):
        """
        MSE loss against FDM reference data.

        Parameters
        ----------
        X_batch : Tensor (N, 6) — normalised inputs [x,y,t,rho,cp,k]
        T_batch : Tensor (N, 1) — FDM temperature labels [K]

        Returns
        -------
        loss : scalar Tensor
        """
        T_pred = self.forward(
            X_batch[:, 0:1], X_batch[:, 1:2], X_batch[:, 2:3],
            X_batch[:, 3:4], X_batch[:, 4:5], X_batch[:, 5:6],
        )
        return nn.MSELoss()(T_pred, T_batch)


# ── PINN ──────────────────────────────────────────────────────────────────────

class PINN(HeatNet):
    """
    Physics-Informed Neural Network.

    Trained without FDM data. The 2D unsteady heat equation is embedded
    directly in the loss function via automatic differentiation (autograd).

    Composite Loss
    --------------
    L_PINN = λ_pde · L_pde  +  λ_bc · L_bc  +  λ_ic · L_ic

    L_pde : PDE residual at interior collocation points (normalised)
    L_bc  : MSE of predictions on all four walls vs T_boundary
    L_ic  : MSE of predictions at t = 0 vs T_initial

    PDE residual (heat equation)
    ----------------------------
    f(x) = ρ c_p ∂T/∂t  −  k (∂²T/∂x²  +  ∂²T/∂y²) ≈ 0

    Chain rule for normalised coordinates
    --------------------------------------
    ∂T/∂t   = ∂T/∂t̂ · (1/t_max)
    ∂²T/∂x² = ∂²T/∂x̂² · (1/L_x²)
    ∂²T/∂y² = ∂²T/∂ŷ² · (1/L_y²)

    All loss terms are normalised to O(1) at initialisation to prevent
    any single term from dominating before the weights λ can take effect.

    Parameters
    ----------
    n_hidden    : int   — number of hidden layers
    n_neurons   : int   — neurons per hidden layer
    lambda_pde  : float — weight on PDE residual loss
    lambda_bc   : float — weight on boundary condition loss
    lambda_ic   : float — weight on initial condition loss
    """

    def __init__(self, n_hidden=5, n_neurons=40,
                 lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0):
        super().__init__(n_hidden, n_neurons)
        self.lambda_pde = lambda_pde
        self.lambda_bc  = lambda_bc
        self.lambda_ic  = lambda_ic

    def _to_tensor(self, arr, device, req_grad=False):
        """Convert 1D numpy array to column tensor (N,1) on device."""
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(device)
        return t.requires_grad_(req_grad)

    def _pde_residual(self, x_n, y_n, t_n,
                      rho_n, cp_n, k_n,
                      rho, cp, k):
        """
        Compute normalised heat equation residual via automatic differentiation.

        x_n, y_n, t_n must have requires_grad=True so that autograd can
        compute spatial and temporal derivatives through the network.

        Returns
        -------
        residual_norm : Tensor (N, 1) — dimensionless residual
        """
        T = self.forward(x_n, y_n, t_n, rho_n, cp_n, k_n)
        ones = torch.ones_like(T)

        # First-order derivatives w.r.t. normalised coordinates
        dT_dt_n  = torch.autograd.grad(T, t_n, grad_outputs=ones,
                                        create_graph=True, retain_graph=True)[0]
        dT_dx_n  = torch.autograd.grad(T, x_n, grad_outputs=ones,
                                        create_graph=True, retain_graph=True)[0]
        dT_dy_n  = torch.autograd.grad(T, y_n, grad_outputs=ones,
                                        create_graph=True, retain_graph=True)[0]

        # Second-order spatial derivatives
        d2T_dx2_n = torch.autograd.grad(dT_dx_n, x_n, grad_outputs=ones,
                                         create_graph=True, retain_graph=True)[0]
        d2T_dy2_n = torch.autograd.grad(dT_dy_n, y_n, grad_outputs=ones,
                                         create_graph=True, retain_graph=True)[0]

        # Convert to physical derivatives via chain rule
        dT_dt   = dT_dt_n  / T_MAX
        d2T_dx2 = d2T_dx2_n / XMAX**2
        d2T_dy2 = d2T_dy2_n / YMAX**2

        # Heat equation residual [W/m³]
        residual = rho * cp * dT_dt - k * (d2T_dx2 + d2T_dy2)

        # Normalise: reference scale = ρ c_p ΔTΔT / t_max  [W/m³]
        ref_scale = rho * cp * DELTA_T / T_MAX
        return residual / ref_scale

    def compute_loss(self, col, device):
        """
        Compute composite PINN loss from collocation point dictionary.

        Parameters
        ----------
        col    : dict — as returned by dataset.sample_collocation_points()
        device : str  — 'cpu' or 'cuda'

        Returns
        -------
        total  : scalar Tensor — weighted composite loss
        L_pde  : float — unweighted PDE loss value
        L_bc   : float — unweighted BC loss value
        L_ic   : float — unweighted IC loss value
        """
        to = lambda a, g=False: self._to_tensor(a, device, req_grad=g)

        # ── PDE loss ─────────────────────────────────────────────────────────
        x_c, y_c, t_c, rho_c, cp_c, k_c = col["pde"]

        x_n   = to(x_c   / XMAX,  g=True)
        y_n   = to(y_c   / YMAX,  g=True)
        t_n   = to(t_c   / T_MAX, g=True)
        rho_n = to((rho_c - RHO_MIN) / (RHO_MAX - RHO_MIN + 1e-10))
        cp_n  = to((cp_c  - CP_MIN)  / (CP_MAX  - CP_MIN  + 1e-10))
        k_n   = to((k_c   - K_MIN)   / (K_MAX   - K_MIN   + 1e-10))
        rho_t = to(rho_c)
        cp_t  = to(cp_c)
        k_t   = to(k_c)

        res   = self._pde_residual(x_n, y_n, t_n, rho_n, cp_n, k_n,
                                   rho_t, cp_t, k_t)
        L_pde = torch.mean(res**2)

        # ── BC loss ──────────────────────────────────────────────────────────
        x_bc, y_bc, t_bc, rho_bc, cp_bc, k_bc = col["bc"]

        T_bc = self.forward(
            to(x_bc / XMAX),
            to(y_bc / YMAX),
            to(t_bc / T_MAX),
            to((rho_bc - RHO_MIN) / (RHO_MAX - RHO_MIN + 1e-10)),
            to((cp_bc  - CP_MIN)  / (CP_MAX  - CP_MIN  + 1e-10)),
            to((k_bc   - K_MIN)   / (K_MAX   - K_MIN   + 1e-10)),
        )
        # Normalise by DELTA_T so L_bc is dimensionless
        L_bc = torch.mean(((T_bc - T_BOUNDARY) / DELTA_T)**2)

        # ── IC loss ──────────────────────────────────────────────────────────
        x_ic, y_ic, t_ic, rho_ic, cp_ic, k_ic = col["ic"]

        T_ic = self.forward(
            to(x_ic / XMAX),
            to(y_ic / YMAX),
            to(t_ic / T_MAX),
            to((rho_ic - RHO_MIN) / (RHO_MAX - RHO_MIN + 1e-10)),
            to((cp_ic  - CP_MIN)  / (CP_MAX  - CP_MIN  + 1e-10)),
            to((k_ic   - K_MIN)   / (K_MAX   - K_MIN   + 1e-10)),
        )
        L_ic = torch.mean(((T_ic - T_INITIAL) / DELTA_T)**2)

        # ── Weighted composite ────────────────────────────────────────────────
        total = (self.lambda_pde * L_pde
                 + self.lambda_bc  * L_bc
                 + self.lambda_ic  * L_ic)

        return total, L_pde.item(), L_bc.item(), L_ic.item()
