# PINN — 2D Cylinder Flow (Navier–Stokes)

A **Physics-Informed Neural Network (PINN)** that learns the velocity and pressure fields of incompressible flow past a cylinder at **Re = 100**, trained jointly on CFD reference data and the Navier–Stokes PDE residuals.

---

## What is a PINN?

A PINN embeds physical laws directly into the loss function of a neural network.  
Instead of only minimising the error against labelled data, the network is also penalised for violating the governing differential equations enabling physics-consistent predictions even in data-sparse regions.

---

## Problem Overview

| Property | Value |
|---|---|
| Flow | 2D incompressible flow around a circular cylinder |
| Reynolds number | Re = 100 |
| Governing equations | Incompressible Navier–Stokes + continuity |
| Inputs | `(x, y, t)` |
| Outputs | `u` (x-velocity), `v` (y-velocity), `p` (pressure) |

### Navier–Stokes Equations (non-dimensional)

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \frac{1}{Re}\left(\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}\right) = 0$$

$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \frac{1}{Re}\left(\frac{\partial^2 v}{\partial x^2}+\frac{\partial^2 v}{\partial y^2}\right) = 0$$

---

## Network Architecture

```
Input  (x, y, t)
  │
  ├─ Dense(50, tanh)  × 10 hidden layers
  │
Output (u, v, p)
```

Derivatives needed for the physics residuals are computed via automatic differentiation using `tf.GradientTape` (no finite differences).

---

## Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}}\,\mathcal{L}_{\text{phys}}$$

| Term | Description |
|---|---|
| $\mathcal{L}_{\text{data}}$ | MSE between predictions and CFD reference values at sampled points |
| $\mathcal{L}_{\text{phys}}$ | Mean squared PDE residuals at **collocation points** (distinct from data points) |
| $\lambda_{\text{phys}}$ | Weighting coefficient (default: 1.0) |

---

## Repository Structure

```
.
├── PINN_cylinder2D.ipynb   # Main notebook (train + evaluate)
├── README.md
├── requirements.txt
├── data/
│   └── Cylinder2D.mat      # CFD dataset — see instructions below
├── checkpoints/            # Saved model weights (created at runtime)
└── outputs/
    ├── loss_history.csv
    ├── loss_curve.png
    └── val_images/         # Validation snapshots per epoch
```

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/Abderrafai-ETTAQI/PINN-for-2D-cylinder-flow.git
cd PINN-cylinder2D
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

The notebook expects `data/Cylinder2D.mat`.  
Download the file from the original Raissi et al. repository or any mirror:

```
https://github.com/maziarraissi/PINNs/tree/master/main/Data
```

Place it at `data/Cylinder2D.mat` (or update `DATA_PATH` in the notebook).

### 4. Run the notebook

```bash
jupyter notebook PINN_cylinder2D.ipynb
```

Set `WEIGHTS_PATH = None` to train from scratch, or point it to a checkpoint to resume.

---

## Requirements

See `requirements.txt`. Main dependencies:

| Package | Version |
|---|---|
| TensorFlow | ≥ 2.12 |
| NumPy | ≥ 1.23 |
| SciPy | ≥ 1.10 |
| Matplotlib | ≥ 3.7 |
| Pandas | ≥ 2.0 |

---

## Results

Validation at time snapshot `t = 100` after training:

| Field | Description |
|---|---|
| `u` | x-velocity — captures the Kármán vortex shedding pattern |
| `v` | y-velocity — symmetric alternating wake structure |
| `p` | pressure — high-pressure stagnation region upstream of cylinder |

Validation images are saved every 50 epochs to `outputs/val_images/`.

---

## Key Design Choices

- **Separate collocation points** for physics loss — enforces PDE across the full domain, not only at data locations (a requirement for a true PINN).
- **Input normalisation** — zero-mean, unit-variance normalisation applied to `(x, y, t)` before entering the network.
- **Custom training loop** — necessary because `tf.GradientTape` must be nested inside the physics loss; `model.compile` / `model.fit` cannot express this naturally.
- **Best-model checkpointing** — weights are saved whenever the total loss improves.

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- [Original PINNs repository](https://github.com/maziarraissi/PINNs)

---

## License

MIT
