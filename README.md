# PINN Heat Transfer — Weber Grill Project

**Module:** MKM411 - Computational Fluid Dynamics  
**Department:** Mechanical & Aeronautical Engineering | University of Pretoria  
**Lecturers:** Prof Ken Craig and Prof Muaaz Bhamjee  
**References:** Raissi, M., Yazdani, A., & Karniadakis, G.E. (2020). *Hidden fluid mechanics.* Science, 367(6481), 1026–1030. https://doi.org/10.1126/science.aaw4741  
Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. https://doi.org/10.1016/j.jcp

---

## Overview

In this component of the project you will train, tune, and benchmark a
Physics-Informed Neural Network (PINN) for unsteady 2D heat transfer — applied
to the problem of cooking meat on a Weber grill. You will compare your PINN
against a standard data-driven ANN and against reference solutions (FDM and
exact Fourier series).

This repository contains everything you need to run the project.

---

## Learning Outcomes

This repository supports the SciML component of MKM411. On completing the
associated lectures and this project component, you will be able to:

- Derive and implement the PINN loss function for the 2D unsteady heat equation,
  embedding the governing PDE, boundary conditions, and initial condition via
  automatic differentiation
- Train both a data-driven ANN and a PINN in PyTorch and interpret their training
  and validation loss curves
- Conduct systematic hyperparameter experiments — varying network depth, learning
  rate, and loss weights — and provide physically grounded explanations for
  observed trends
- Benchmark PINN and ANN predictions quantitatively against FDM and exact Fourier
  solutions across multiple material types
- Evaluate model generalisation to an unseen material and explain, from first
  principles, why physics-constrained learning outperforms a purely data-driven
  approach on out-of-distribution inputs
- Critically compare all solution paradigms — analytical, FDM, ANN, PINN, and
  commercial CFD — across accuracy, computational cost, data requirements, and
  interpretability

For the full set of lecture learning outcomes, see the
[lecture repository](https://github.com/muaazbhamjee/mkm411-sciml-lectures).

---

## Repository Structure

```
pinn-heat-transfer/
├── data/
│   ├── train_beef.npz       ← FDM dataset: beef
│   ├── train_chicken.npz    ← FDM dataset: chicken
│   ├── train_pork.npz       ← FDM dataset: pork
│   └── test_lamb.npz        ← FDM dataset: lamb (unseen test meat)
│
├── utils/
│   ├── __init__.py
│   ├── config.py            ← domain constants, material properties
│   ├── fourier.py           ← exact Fourier series solution
│   ├── dataset.py           ← data loading, normalisation, collocation points
│   ├── models.py            ← ANN and PINN architectures (PyTorch)
│   ├── training.py          ← train_ann, train_pinn functions
│   └── evaluation.py        ← prediction, benchmarking, plotting
│
├── generate_dataset.py      ← FDM solver used to generate the training datasets. Read it to understand how the data was produced, but do not run it — the datasets are already provided in data/.
├── PINN_Heat_Transfer.ipynb ← your working notebook
├── environment.yml          ← Anaconda environment specification
├── .gitignore
└── README.md
```

---

## Setup Instructions

Follow the steps for your operating system. Complete these **before** your
first lab session.

---

### Step 1 — Install Git

**Windows**  
Download and install Git from https://git-scm.com/download/win  
During installation, select *"Git from the command line and also from 3rd-party software"*.

**Mac**  
Git is pre-installed. Open Terminal and type `git --version` to confirm.  
If prompted, install Xcode Command Line Tools.

---

### Step 2 — Install Anaconda

Download Anaconda Individual Edition from https://www.anaconda.com/download  
Install with default settings. Do **not** add Anaconda to PATH on Windows (the
installer recommends against it — use Anaconda Prompt instead).

---

### Step 3 — Clone the Repository

Open **Anaconda Prompt** (Windows) or **Terminal** (Mac) and run:

```bash
git clone https://github.com/muaazbhamjee/pinn-heat-transfer.git
cd pinn-heat-transfer
```

---

### Step 4 — Create the Conda Environment

Inside the `pinn-heat-transfer` directory, run:

```bash
conda env create -f environment.yml
conda activate pinn-heat
pip install -r requirements.txt -vvv
```

This will download and install Python, PyTorch, NumPy, Matplotlib, and Jupyter.
It may take 5–10 minutes depending on your internet connection.

> **Troubleshooting (Windows):** If you see a permissions error, run Anaconda
> Prompt as Administrator.

> **Troubleshooting (Mac M1/M2):** If PyTorch fails to install, replace the
> pip section in `environment.yml` with:
> ```yaml
>   - pip:
>       - torch==2.1.0
> ```
> Apple Silicon uses a different PyTorch build — the CPU version installs cleanly.

---

### Step 5 — Activate the Environment

```bash
conda activate pinn-heat
```

You should see `(pinn-heat)` appear at the start of your prompt.

> You must activate this environment **every time** you open a new terminal
> before working on the project.

---

### Step 6 — Launch Jupyter Notebook

```bash
jupyter notebook
```

Your browser will open automatically. Navigate to `PINN_Heat_Transfer.ipynb`
and open it.

---

### Step 7 — Verify the Setup

In the first code cell of the notebook, run:

```python
import torch
import numpy as np
print(torch.__version__)
print(np.__version__)
```

If both print without errors, you are ready.

---


---

## Hardware Requirements and Training Time

Training the PINN is computationally intensive. Estimated runtimes for the
full Section 7 experiment set (9 PINN training jobs):

| Hardware | Estimated time |
|----------|---------------|
| NVIDIA GPU (CUDA) | ~20–30 minutes |
| Apple Silicon (M1/M2/M3) | ~60–90 minutes |
| CPU only (laptop) | ~4–8 hours |

---

### Option A — You have an NVIDIA GPU

No changes needed. Run the notebook as-is.

To confirm PyTorch can see your GPU:
```python
import torch
print(torch.cuda.is_available())   # should print True
print(torch.cuda.get_device_name(0))
```

---

### Option B — CPU only (recommended: reduce epochs for Section 7)

If you do not have a GPU, reduce the training epochs in the hyperparameter
block before running Section 7:

```python
EPOCHS_ADAM  = 2000   # reduced from 10000
EPOCHS_LBFGS = 100    # reduced from 500
```

This gives indicative results in ~1 hour. Results will be less converged
but sufficient to observe the qualitative trends required in the report.
Run the full epoch count once overnight for your final submission.

---

### Option C — Google Colab (free GPU, recommended for CPU-only users)

Google Colab provides free access to a T4 GPU and requires no local installation.

1. Go to https://colab.research.google.com
2. Click **File → Upload notebook** and upload `PINN_Heat_Transfer.ipynb`
3. Upload the `data/` and `utils/` folders using the file browser on the left
   (click the folder icon, then upload — maintain the same folder structure)
4. Click **Runtime → Change runtime type → T4 GPU → Save**
5. Run the notebook as normal — training will complete in ~30 minutes

> **Note:** Colab sessions disconnect after ~90 minutes of inactivity and
> all uploaded files are lost on disconnect. Save your notebook outputs
> frequently using **File → Save a copy in Drive**.

## Working on the Project

- All your work happens in `PINN_Heat_Transfer.ipynb`
- The `utils/` folder contains the model and training code — **read it, understand it,
  but do not modify it** unless instructed (your report must explain what is in these files)
- `generate_dataset.py` FDM solver used to generate the training datasets. Read it to understand how the data was produced, but do not run it — the datasets are already provided in data/.
- The `data/` folder is pre-populated — do not delete or modify the `.npz` files

---

## Submitting

Submit the following via [submission system TBD]:
1. Your completed `PINN_Heat_Transfer.ipynb` (with all cells executed and output visible)
2. Your group report (PDF)

Before submitting your notebook, restart the kernel and run all cells from top
to bottom to confirm it executes cleanly end-to-end:  
`Kernel → Restart & Run All`

---

## Git Workflow (Good Practice)

We encourage you to use Git to track your group's progress. After cloning:

```bash
# Check what has changed
git status

# Stage your notebook changes
git add PINN_Heat_Transfer.ipynb

# Commit with a descriptive message
git commit -m "Section 7: hyperparameter experiments for depth study"

# Push to your group's fork (if using GitHub)
git push
```

> Do not commit large model checkpoint files (`.pt`, `.pth`) —
> these are listed in `.gitignore`.

---

## Common Issues

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError: No module named 'utils'` | Make sure you launched Jupyter from inside the `pinn-heat-transfer/` directory |
| `ModuleNotFoundError: No module named 'torch'` | Conda environment not activated — run `conda activate pinn-heat` |
| Training is very slow | See Hardware Requirements section above — use Google Colab for free GPU access, or reduce `EPOCHS_ADAM` to 2000 for initial experiments |
| Notebook output not showing | Run `Kernel → Restart & Run All` |
| Git merge conflict in notebook | Each group member should work on a personal branch and merge via pull request |

---

## Citation

If you use this repository in your research or teaching, please cite it using the
reference in `CITATION.cff` or the following:

> Bhamjee, M., & Craig, K.J. (2025). *MKM411 PINN Heat Transfer — Physics-Informed
> Neural Networks for Unsteady Heat Transfer* [Software]. University of Pretoria.
> https://github.com/muaazbhamjee/pinn-heat-transfer

This work is grounded in the following seminal papers:

> Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural
> networks: A deep learning framework for solving forward and inverse problems
> involving nonlinear PDEs. *Journal of Computational Physics*, 378, 686–707.
> https://doi.org/10.1016/j.jcp.2018.10.045

> Raissi, M., Yazdani, A., & Karniadakis, G.E. (2020). Hidden fluid mechanics:
> Learning velocity and pressure fields from flow visualizations. *Science*,
> 367(6481), 1026–1030. https://doi.org/10.1126/science.aaw4741

---

## Attribution:

During development, the following AI-assisted tools were used:

* Claude AI (Anthropic):
  * debugging 
  * architecture decisions
  * physics implementation guidance
  * code generation and review
  * documentation

All research questions, pedagogical design decisions, theoretical contributions, and experimental results are the authors' own. 

---

## Contact

Raise technical issues via email to Prof Bhamjee on muaaz.bhamjee@up.ac.za  
For conceptual questions about PINNs, refer to the lecture slides and the Raissi et al. (2019) and (2020) papers.