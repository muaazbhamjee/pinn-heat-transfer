"""
utils/
======
Utility package for the PINN Heat Transfer project.

Modules
-------
config      — domain constants, material properties, normalisation bounds
fourier     — exact Fourier series solution
dataset     — data loading, normalisation, collocation point sampling
models      — HeatNet backbone, ANN, PINN
training    — train_ann, train_pinn
evaluation  — predict, benchmark, plotting helpers
"""

from .config     import *
from .fourier    import T_fourier_point, T_fourier_grid, T_fourier_centre_history
from .dataset    import normalise_inputs, load_meat_data, build_ann_dataloader, sample_collocation_points
from .models     import HeatNet, ANN, PINN
from .training   import train_ann, train_pinn
from .evaluation import (predict_field, predict_centre_history,
                          evaluate_model, plot_loss_curves,
                          plot_field_comparison, plot_centre_temperature,
                          print_metrics_table)
