import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm


# --- Configuration -----------------------------------------------------------
@dataclass
class SimulationConfig:
    """Container for all simulation tunables."""

    lattice_size: int = 64
    steps_per_temp: int = 3000  # Monte Carlo sweeps recorded for each T
    equilibration_sweeps: int = 1500  # Sweeps discarded as burn-in
    temp_min: float = 1.0
    temp_max: float = 3.5
    temp_points: int = 60
    output_dir: str = "results"
    seed: Optional[int] = None

    @property
    def temps(self) -> np.ndarray:
        return np.linspace(self.temp_min, self.temp_max, self.temp_points)


# --- JIT-accelerated core ----------------------------------------------------
@njit(cache=True)
def initial_state(N: int) -> np.ndarray:
    """Generate a random spin configuration (-1 or +1) as int8 to minimize memory."""
    spins = np.empty((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(N):
            spins[i, j] = 1 if np.random.rand() > 0.5 else -1
    return spins


@njit(cache=True)
def calc_energy(config: np.ndarray) -> float:
    """Compute total energy with periodic boundary conditions, counting each bond once."""
    N = config.shape[0]
    E = 0.0
    for i in range(N):
        for j in range(N):
            s = config[i, j]
            nb = config[(i + 1) % N, j] + config[i, (j + 1) % N]
            E -= s * nb
    return E


@njit(cache=True)
def calc_mag(config: np.ndarray) -> int:
    """Compute total magnetization."""
    return int(np.sum(config))


@njit(cache=True)
def metropolis_sweep(
    config: np.ndarray,
    beta: float,
    p4: float,
    p8: float,
    energy: float,
    magnetization: float,
) -> Tuple[float, float, int, int]:
    """
    Perform one Metropolis sweep (N^2 proposals).

    Returns updated (energy, magnetization, accepted_flips, total_trials).
    Acceptance uses precomputed Boltzmann factors for ΔE = 4, 8.
    """
    N = config.shape[0]
    accepted = 0
    trials = N * N

    for _ in range(trials):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s = config[i, j]
        nb = (
            config[(i + 1) % N, j]
            + config[(i - 1 + N) % N, j]
            + config[i, (j + 1) % N]
            + config[i, (j - 1 + N) % N]
        )
        dE = 2 * s * nb

        if dE <= 0:
            accept = True
        elif dE == 4:
            accept = np.random.rand() < p4
        elif dE == 8:
            accept = np.random.rand() < p8
        else:
            accept = np.random.rand() < math.exp(-dE * beta)

        if accept:
            config[i, j] = -s
            energy += dE
            magnetization -= 2 * s
            accepted += 1

    return energy, magnetization, accepted, trials


def configure_matplotlib() -> None:
    """Set a publication-ready dark theme for plots."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.figsize": (11, 9),
            "savefig.dpi": 320,
        }
    )


def run_simulation(cfg: SimulationConfig) -> Tuple[np.ndarray, ...]:
    """
    Sweep across temperatures and return thermodynamic observables.

    Observables are averaged per spin:
    - energy density ⟨E⟩/N
    - magnetization ⟨|M|⟩/N
    - specific heat C_v
    - susceptibility χ
    """
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    temps = cfg.temps
    N_sites = cfg.lattice_size**2

    E_avg = np.empty_like(temps)
    M_avg = np.empty_like(temps)
    Cv = np.empty_like(temps)
    Chi = np.empty_like(temps)
    acceptance = np.empty_like(temps)

    for idx, T in enumerate(tqdm(temps, desc="Sweeping temperatures")):
        beta = 1.0 / T
        p4 = math.exp(-4.0 * beta)
        p8 = math.exp(-8.0 * beta)
        config = initial_state(cfg.lattice_size)
        energy = calc_energy(config)
        magnetization = float(calc_mag(config))

        acc_flips = 0
        total_trials = 0

        # Equilibration
        for _ in range(cfg.equilibration_sweeps):
            energy, magnetization, a, t = metropolis_sweep(
                config, beta, p4, p8, energy, magnetization
            )
            acc_flips += a
            total_trials += t

        # Sampling
        e_sum = 0.0
        e2_sum = 0.0
        m_abs_sum = 0.0
        m_sum = 0.0
        m2_sum = 0.0

        for _ in range(cfg.steps_per_temp):
            energy, magnetization, a, t = metropolis_sweep(
                config, beta, p4, p8, energy, magnetization
            )
            acc_flips += a
            total_trials += t

            e_sum += energy
            e2_sum += energy * energy
            m_abs = abs(magnetization)
            m_abs_sum += m_abs
            m_sum += magnetization
            m2_sum += magnetization * magnetization

        norm = 1.0 / cfg.steps_per_temp
        e_mean = e_sum * norm
        e2_mean = e2_sum * norm
        m_abs_mean = m_abs_sum * norm
        m_mean = m_sum * norm
        m2_mean = m2_sum * norm

        E_avg[idx] = e_mean / N_sites
        M_avg[idx] = m_abs_mean / N_sites
        Cv[idx] = (e2_mean - e_mean**2) / (N_sites * (T**2))
        Chi[idx] = (m2_mean - m_mean**2) / (N_sites * T)
        acceptance[idx] = acc_flips / total_trials

    return temps, E_avg, M_avg, Cv, Chi, acceptance


def plot_thermodynamics(
    temps: np.ndarray,
    E: np.ndarray,
    M: np.ndarray,
    Cv: np.ndarray,
    Chi: np.ndarray,
    output_dir: str,
) -> None:
    """Plot canonical observables vs. temperature and persist PNG."""
    Tc = 2.26918531421  # Onsager critical temperature for 2D square lattice
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(temps, E, "o-", color="#f97316", markersize=4, alpha=0.9)
    axes[0, 0].set_ylabel(r"Energy per spin $\langle E \rangle / N$")
    axes[0, 0].set_title("Internal Energy")

    axes[0, 1].plot(temps, M, "o-", color="#22d3ee", markersize=4, label="Simulation")
    T_theory = np.linspace(1.0, Tc, 200)
    M_theory = (1 - np.sinh(2 / T_theory) ** -4) ** 0.125
    axes[0, 1].plot(T_theory, M_theory, "k--", linewidth=1.2, label="Onsager exact")
    axes[0, 1].set_ylabel(r"Magnetization $\langle |M| \rangle / N$")
    axes[0, 1].set_title("Spontaneous Magnetization")
    axes[0, 1].legend()

    axes[1, 0].plot(temps, Cv, "o-", color="#16a34a", markersize=4, alpha=0.9)
    axes[1, 0].axvline(Tc, color="white", linestyle="--", alpha=0.6, label=r"$T_c$")
    axes[1, 0].set_ylabel(r"Specific heat $C_v$")
    axes[1, 0].set_title("Specific Heat")
    axes[1, 0].legend()

    axes[1, 1].plot(temps, Chi, "o-", color="#a855f7", markersize=4, alpha=0.9)
    axes[1, 1].axvline(Tc, color="white", linestyle="--", alpha=0.6)
    axes[1, 1].set_ylabel(r"Susceptibility $\chi$")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Magnetic Susceptibility")

    for ax in axes.flat:
        ax.set_xlabel(r"Temperature $k_B T / J$")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/thermodynamics.png")
    plt.close(fig)
    print(f"Saved results to {output_dir}/thermodynamics.png")


def plot_snapshots(cfg: SimulationConfig) -> None:
    """Generate representative lattice snapshots at low, critical, and high T."""
    temps = (1.2, 2.3, 3.4)
    labels = (
        "Ferromagnetic (T < Tc)",
        "Critical (T ≈ Tc)",
        "Paramagnetic (T > Tc)",
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, T, label in zip(axes, temps, labels):
        beta = 1.0 / T
        p4 = math.exp(-4.0 * beta)
        p8 = math.exp(-8.0 * beta)
        config = initial_state(cfg.lattice_size)
        energy = calc_energy(config)
        magnetization = float(calc_mag(config))

        for _ in range(cfg.equilibration_sweeps * 10):
            energy, magnetization, _, _ = metropolis_sweep(
                config, beta, p4, p8, energy, magnetization
            )

        ax.imshow(config, cmap="binary", interpolation="nearest")
        ax.set_title(f"{label}\nT = {T:.2f}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/snapshots.png")
    plt.close(fig)
    print("Saved lattice snapshots to results/snapshots.png")


def parse_args() -> Tuple[SimulationConfig, bool]:
    parser = argparse.ArgumentParser(
        description="High-performance 2D Ising Model Monte Carlo simulation."
    )
    parser.add_argument("--size", type=int, default=64, help="Linear lattice size.")
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Monte Carlo sweeps recorded per temperature.",
    )
    parser.add_argument(
        "--equil", type=int, default=1500, help="Equilibration sweeps (burn-in)."
    )
    parser.add_argument(
        "--tmin", type=float, default=1.0, help="Minimum temperature (inclusive)."
    )
    parser.add_argument(
        "--tmax", type=float, default=3.5, help="Maximum temperature (inclusive)."
    )
    parser.add_argument(
        "--points", type=int, default=60, help="Number of temperature points."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip rendering lattice snapshots.",
    )
    args = parser.parse_args()
    return SimulationConfig(
        lattice_size=args.size,
        steps_per_temp=args.steps,
        equilibration_sweeps=args.equil,
        temp_min=args.tmin,
        temp_max=args.tmax,
        temp_points=args.points,
        seed=args.seed,
    ), args.skip_snapshots


def main() -> None:
    cfg, skip_snapshots = parse_args()
    configure_matplotlib()
    temps, E, M, Cv, Chi, acceptance = run_simulation(cfg)

    np.savez(
        os.path.join(cfg.output_dir, "observables.npz"),
        temps=temps,
        energy=E,
        magnetization=M,
        specific_heat=Cv,
        susceptibility=Chi,
        acceptance=acceptance,
    )
    print(f"Saved raw observables to {cfg.output_dir}/observables.npz")

    plot_thermodynamics(temps, E, M, Cv, Chi, cfg.output_dir)
    if not skip_snapshots:
        plot_snapshots(cfg)

    print(
        f"Simulation complete for L={cfg.lattice_size} over "
        f"{cfg.temp_points} temperatures."
    )


if __name__ == "__main__":
    main()
