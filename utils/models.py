"""
models.py
=========
Neural network architectures for the PINN heat transfer project.

HeatNet  — shared fully-connected backbone (6 inputs → 1 output)
ANN      — data-driven model, MSE loss against FDM data
PINN     — physics-informed model, composite loss via autograd

Key design decisions
--------------------
1. OUTPUT NORMALISATION
   The network internally predicts a normalised temperature:
       θ = (T - T_INITIAL) / DELTA_T  ∈ [0, 1]

   forward() reconstructs and returns physical temperature:
       T = θ * DELTA_T + T_INITIAL  [K]

   With Xavier initialisation, the network initially outputs θ ≈ 0,
   so T ≈ T_INITIAL everywhere. This:
     - Naturally satisfies the initial condition (T_initial at t=0)
     - Gives a physically meaningful starting point for training
     - Prevents the trivial solution T=constant (near 0 K) from occurring
     - Ensures BC loss drives training from the start

2. LOSS WEIGHTS
   The default PINN weights are λ_bc = λ_ic = 10, λ_pde = 1.
   With output normalisation, the IC loss starts near zero and the BC
   loss starts near 1.0. Higher BC/IC weights ensure the boundary and
   initial conditions are enforced before the PDE residual is reduced.

3. SHARED ARCHITECTURE
   Both ANN and PINN share identical architecture — a fully-connected
   network with tanh activations and Xavier initialisation. The only
   difference is the loss function.
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
    Hidden : n_hidden layers × n_neurons neurons  [tanh]
    Output : T [K]                                — physical temperature

    Internally the network predicts normalised temperature
        θ = (T - T_INITIAL) / DELTA_T
    and forward() returns T = θ * DELTA_T + T_INITIAL.

    With Xavier initialisation, θ ≈ 0 initially → T ≈ T_INITIAL everywhere.
    This is a physically meaningful starting point (cold meat, no heat yet).

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
        """
        Xavier/Glorot initialisation — calibrated variance for tanh.
        Output layer bias is set to zero so initial θ ≈ 0 → T ≈ T_INITIAL.
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x_n, y_n, t_n, rho_n, cp_n, k_n):
        """
        Forward pass.

        Parameters (all Tensor of shape (N, 1), normalised to [0, 1])
        ----------
        x_n, y_n, t_n    : normalised spatial and temporal coordinates
        rho_n, cp_n, k_n : normalised material properties

        Returns
        -------
        T : Tensor (N, 1) — physical temperature [K]
            Internally: θ = net(inputs),  T = θ * DELTA_T + T_INITIAL
        """
        inp   = torch.cat([x_n, y_n, t_n, rho_n, cp_n, k_n], dim=1)
        theta = self.net(inp)                          # normalised temperature θ
        return theta * DELTA_T + T_INITIAL             # physical temperature T [K]

    def n_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self):
        print(f"{self.__class__.__name__}")
        print(f"  Inputs     : 6  (x, y, t, rho, cp, k)  — normalised")
        print(f"  Hidden     : {self.n_hidden} layers × {self.n_neurons} neurons  [tanh]")
        print(f"  Output     : T [K]  (internal: θ = (T-T_i)/ΔT, then reconstructed)")
        print(f"  Parameters : {self.n_params():,}")


# ── ANN ───────────────────────────────────────────────────────────────────────

class ANN(HeatNet):
    """
    Data-driven Artificial Neural Network.

    Trained on FDM simulation data using mean squared error loss.
    No physics is embedded — the model learns the input-output mapping
    purely from data.

    Loss
    ----
    L_ANN = (1/N) Σ (T_θ(x_i) - T_FDM(x_i))²  [K²]
    """

    def __init__(self, n_hidden=5, n_neurons=40):
        super().__init__(n_hidden, n_neurons)

    def compute_loss(self, X_batch, T_batch):
        """
        MSE loss against FDM reference data.

        Parameters
        ----------
        X_batch : Tensor (N, 6) — normalised inputs
        T_batch : Tensor (N, 1) — FDM temperature labels [K]

        Returns
        -------
        loss : scalar Tensor [K²]
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

    L_pde : normalised PDE residual at interior collocation points
    L_bc  : normalised MSE on all four walls   — target: T = T_BOUNDARY
    L_ic  : normalised MSE at t = 0            — target: T = T_INITIAL

    All loss terms are normalised by DELTA_T so they are dimensionless
    and O(1) at initialisation, making the λ weights directly comparable.

    Default weights: λ_bc = λ_ic = 10, λ_pde = 1
    Higher BC/IC weights ensure boundary and initial conditions are enforced
    before the PDE residual is minimised. This prevents the trivial solution
    (T = constant → zero PDE residual) from dominating early training.

    PDE residual (heat equation)
    ----------------------------
    f = ρ c_p ∂T/∂t  −  k (∂²T/∂x²  +  ∂²T/∂y²) ≈ 0

    Chain rule for normalised coordinates (with output normalisation)
    -----------------------------------------------
    T = θ · ΔT + T_i
    ∂T/∂t   = ΔT · ∂θ/∂t̂ / t_max
    ∂²T/∂x² = ΔT · ∂²θ/∂x̂² / L_x²
    ∂²T/∂y² = ΔT · ∂²θ/∂ŷ² / L_y²

    Substituting and dividing by (ρ c_p ΔT / t_max):
    normalised residual = ∂θ/∂t̂  −  (k t_max)/(ρ c_p) · [∂²θ/∂x̂²/L_x² + ∂²θ/∂ŷ²/L_y²]

    Parameters
    ----------
    n_hidden    : int   — number of hidden layers
    n_neurons   : int   — neurons per hidden layer
    lambda_pde  : float — PDE residual weight  (default: 1.0)
    lambda_bc   : float — BC loss weight       (default: 10.0)
    lambda_ic   : float — IC loss weight       (default: 10.0)
    """

    def __init__(self, n_hidden=5, n_neurons=40,
                 lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0):
        super().__init__(n_hidden, n_neurons)
        self.lambda_pde = lambda_pde
        self.lambda_bc  = lambda_bc
        self.lambda_ic  = lambda_ic

    def _to_tensor(self, arr, device, req_grad=False):
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(device)
        return t.requires_grad_(req_grad)

    def _pde_residual(self, x_n, y_n, t_n,
                      rho_n, cp_n, k_n,
                      rho, cp, k):
        """
        Compute normalised heat equation residual via automatic differentiation.

        The network predicts θ = (T - T_INITIAL) / DELTA_T internally.
        forward() returns T = θ * DELTA_T + T_INITIAL.
        Autograd differentiates T — since T_INITIAL is a constant, all
        derivatives of T equal DELTA_T × derivatives of θ. DELTA_T
        cancels in the normalised residual.

        Parameters (all Tensor (N,1), x_n/y_n/t_n require requires_grad=True)
        -----------------------------------------------------------------------
        Returns
        -------
        residual_norm : Tensor (N,1) — dimensionless PDE residual
        """
        T    = self.forward(x_n, y_n, t_n, rho_n, cp_n, k_n)
        ones = torch.ones_like(T)

        # First-order temporal derivative
        dT_dt_n  = torch.autograd.grad(T, t_n, grad_outputs=ones,
                                        create_graph=True, retain_graph=True)[0]
        # First-order spatial derivatives (needed for second-order)
        dT_dx_n  = torch.autograd.grad(T, x_n, grad_outputs=ones,
                                        create_graph=True, retain_graph=True)[0]
        dT_dy_n  = torch.autograd.grad(T, y_n, grad_outputs=ones,
                                        create_graph=True, retain_graph=True)[0]

        # Second-order spatial derivatives
        d2T_dx2_n = torch.autograd.grad(dT_dx_n, x_n, grad_outputs=ones,
                                         create_graph=True, retain_graph=True)[0]
        d2T_dy2_n = torch.autograd.grad(dT_dy_n, y_n, grad_outputs=ones,
                                         create_graph=True, retain_graph=True)[0]

        # Apply chain rule: physical derivatives from normalised-coord derivatives
        dT_dt   = dT_dt_n   / T_MAX
        d2T_dx2 = d2T_dx2_n / XMAX**2
        d2T_dy2 = d2T_dy2_n / YMAX**2

        # Heat equation residual [W/m³]
        residual = rho * cp * dT_dt - k * (d2T_dx2 + d2T_dy2)

        # Normalise by reference scale → dimensionless O(1)
        ref_scale = rho * cp * DELTA_T / T_MAX
        return residual / ref_scale

    def compute_loss(self, col, device):
        """
        Compute composite PINN loss from collocation point dictionary.

        Parameters
        ----------
        col    : dict — from dataset.sample_collocation_points()
        device : str

        Returns
        -------
        total  : scalar Tensor
        L_pde  : float — unweighted PDE loss
        L_bc   : float — unweighted BC loss
        L_ic   : float — unweighted IC loss
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
        # Normalised: θ_bc should equal 1.0  (T = T_BOUNDARY)
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
        # Normalised: θ_ic should equal 0.0  (T = T_INITIAL)
        # With output normalisation, initial network output θ≈0 → L_ic ≈ 0 at start
        L_ic = torch.mean(((T_ic - T_INITIAL) / DELTA_T)**2)

        # ── Weighted composite ────────────────────────────────────────────────
        total = (self.lambda_pde * L_pde
                 + self.lambda_bc  * L_bc
                 + self.lambda_ic  * L_ic)

        return total, L_pde.item(), L_bc.item(), L_ic.item()
