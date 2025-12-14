# 2D Ising Model — High-Performance Monte Carlo

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Numba](https://img.shields.io/badge/accelerated-numba-brightgreen.svg)

A research-grade implementation of the 2D square-lattice Ising model. The Python backend delivers fast Monte Carlo sweeps with Numba, plots thermodynamic observables, and emits reproducible data products. The `docs/` folder ships a modern, dark-themed web lab for interactive exploration.

## Model & Method

Hamiltonian (ferromagnetic, \(J=1\)) with periodic boundary conditions:

$$
H(\sigma) = -\sum_{\langle i, j \rangle} s_i s_j , \qquad s_i \in \{-1, +1\}
$$

Spin flips follow Metropolis-Hastings:

$$
P(\Delta E) = \min \left(1, e^{-\beta \Delta E}\right), \quad \beta = 1/T
$$

We measure energy density, \(\langle |M| \rangle / N\), specific heat \(C_v\), and susceptibility \(\chi\) across a temperature sweep that captures the Onsager critical point \(T_c \approx 2.269\).

## Features
- Numba-accelerated sweeps with energy/magnetization updated incrementally (no full-lattice recomputation per step).
- Tunable lattice size, temperature grid, equilibration, and RNG seed via CLI.
- Publication-ready Matplotlib figures plus persisted `observables.npz` for downstream analysis.
- Lattice snapshots (low, critical, high \(T\)).
- Modern WebGL-free frontend (Chart.js + Canvas) with live magnetization/energy and acceptance tracking.

## Quickstart (Python backend)

```bash
python -m venv .venv
source .venv/bin/activate         # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run with defaults: L=64, 60 temps from 1.0 → 3.5
python main.py

# Example: finer grid, larger lattice, skip snapshots
python main.py --size 96 --points 90 --steps 4000 --equil 2000 --skip-snapshots
```

Outputs (written to `results/`):
- `thermodynamics.png` — energy, magnetization (with Onsager curve), \(C_v\), and \(\chi\) vs. \(T\).
- `snapshots.png` — representative equilibrated lattices at \(T<T_c\), \(T \approx T_c\), and \(T>T_c\).
- `observables.npz` — NumPy archive containing raw arrays (`temps`, `energy`, `magnetization`, `specific_heat`, `susceptibility`, `acceptance`).

Key CLI flags:
- `--size` linear lattice size \(L\) (default 64).
- `--steps` recorded Monte Carlo sweeps per temperature.
- `--equil` burn-in sweeps before sampling.
- `--tmin`, `--tmax`, `--points` temperature sweep definition.
- `--seed` for deterministic runs; `--skip-snapshots` to omit PNG snapshots.

## Interactive Web Lab (`docs/`)

Open `docs/index.html` directly or serve locally to avoid CORS issues:

```bash
python -m http.server 8000 --directory docs
# then browse to http://localhost:8000
```

Controls include temperature, sweeps-per-frame, live energy / |magnetization| per spin, and acceptance ratio. The canvas renders a 128×128 lattice using the same Metropolis update rule as the backend.

## Implementation Notes
- Metropolis sweep proposes \(N^2\) flips per sweep; energy and magnetization are updated incrementally using \(\Delta E = 2 s_i \sum_{\text{nn}} s_j\).
- Specific heat and susceptibility are computed from fluctuations of the **total** energy and magnetization, then normalized per spin:  
  \(C_v = (\langle E^2 \rangle - \langle E \rangle^2)/(N T^2)\),  
  \(\chi = (\langle M^2 \rangle - \langle M \rangle^2)/(N T)\).
- Periodic boundary conditions are enforced for both backend and frontend engines.

## Citing / Referencing
- L. Onsager, *Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition*, Phys. Rev. 65, 117 (1944).
- N. Metropolis *et al.*, *Equation of State Calculations by Fast Computing Machines*, J. Chem. Phys. 21, 1087 (1953).

---

Feel free to open issues or pull requests to extend measurements (Binder cumulants, finite-size scaling) or to add GPU backends. This project is tuned for clarity and correctness while staying performant enough for rapid parameter sweeps.
