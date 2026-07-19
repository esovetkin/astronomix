"""Combined hydro + MHD strong-scaling speedup — paper figure reconstruction.

Reproduces ``strong_speedup_hydro_mhd.svg``: the strong-scaling speedup
T(1)/T(G) vs resolution N for the finite-difference (Pallas) backend, with one
curve per (physics, GPU-count) measurement and the ideal-speedup guide lines.

The paper figure combines runs made on different machines (hydro on a 4-GPU
H100 node and an 8-GPU H200 node, MHD on the 8-GPU H200 node), so it is built
in two stages — a measurement stage that runs the strong-scaling sweep for one
(physics, GPU-count) at a time and caches it, and a plotting stage that
assembles the cached curves into the combined figure:

    # measure one curve (repeat on each machine / GPU count):
    python examples/scripts/scaling/strong_scaling_speedup.py --physics hydro --gpus 4
    python examples/scripts/scaling/strong_scaling_speedup.py --physics hydro --gpus 8
    python examples/scripts/scaling/strong_scaling_speedup.py --physics mhd   --gpus 8

    # assemble the figure from whatever curves have been cached (no GPU needed):
    python examples/scripts/scaling/strong_scaling_speedup.py --plot

The hardware label on each curve is auto-detected from the JAX device kind at
measurement time, so a reproduction on different hardware is labelled honestly
(e.g. "A100" instead of the paper's "H100"/"H200").
"""

# general
import sys
from pathlib import Path

# The GPU count must be known before autocvd runs; plotting needs no GPU at all,
# so parse the mode / GPU count from argv up front.
PLOT_ONLY = "--plot" in sys.argv


def _argv_int(flag, default):
    """Return the integer following ``flag`` on the command line, or default."""
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    return default


def _argv_str(flag, default):
    """Return the string following ``flag`` on the command line, or default."""
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


NUM_GPUS = _argv_int("--gpus", 1)
PHYSICS = _argv_str("--physics", "hydro")

# ==== GPU selection ====
# Skip GPU acquisition entirely in plot-only mode (the assembler just reads the
# cached NPZs and draws the figure on CPU).
if not PLOT_ONLY:
    from autocvd import autocvd
    autocvd(num_gpus=NUM_GPUS)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
DATA_DIR = _HERE / "data" / "strong_speedup"
FIG_DIR = _HERE / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# The strong-scaling resolution sweep of the paper figure.
N_VALUES = [128, 192, 256, 384, 512]

# One curve per (physics, GPU-count), with its plot styling. The paper draws
# hydro@4 as a dash-dot line and the two 8-GPU curves as solid lines.
CURVES = [
    ("hydro", 4, dict(color="tab:purple", linestyle="-.", marker="^")),
    ("hydro", 8, dict(color="magenta", linestyle="-", marker="o")),
    ("mhd", 8, dict(color="tab:green", linestyle="-", marker="o")),
]


def _cache_path(physics, gpus):
    """Return the cache path for one (physics, GPU-count) speedup curve."""
    return DATA_DIR / f"speedup_{physics}_{gpus}gpu.npz"


def _benchmark_and_setup(physics):
    """Return the FD (Pallas) benchmark list and setup_fn for a physics module.

    The flagship multi-GPU backend is the finite-difference Pallas solver, so
    the strong-scaling curves use it for both the hydro (sound-wave) and MHD
    (Alfvén-wave) tests.

    Args:
        physics: Either ``"hydro"`` or ``"mhd"``.

    Returns:
        A tuple ``(benchmarks, setup_fn)`` to hand to ``run_strong_scaling``.
    """
    # Imported lazily so plot-only mode never pulls in astronomix / the harness.
    if str(_REPO / "pytests") not in sys.path:
        sys.path.insert(0, str(_REPO / "pytests"))
    from _benchmark_utils import BenchmarkSpec
    from astronomix.option_classes.simulation_config import (
        FINITE_DIFFERENCE,
        PALLAS,
        SimulationConfig,
        SnapshotSettings,
        StaticFloatVector,
        BackendConfig,
    )

    common = dict(
        box_size=StaticFloatVector(3.0, 1.5, 1.5),
        dimensionality=3,
        progress_bar=False,
        memory_analysis=True,
        print_elapsed_time=True,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    fd_pallas = dict(
        backend_config=BackendConfig(
            backend=PALLAS,
            pallas_block_shape=(4, 4, 8),
            pallas_use_triton=True,
            pallas_interpret=False,
        ),
        solver_mode=FINITE_DIFFERENCE,
    )

    if physics == "hydro":
        from astronomix.test_setups.hydrodynamics.sound_wave3D import setup_sound_wave
        benchmarks = [
            BenchmarkSpec(
                label="FD (Pallas)",
                base_config=SimulationConfig(mhd=False, **fd_pallas, **common),
                cfl=1.5,
            )
        ]
        return benchmarks, setup_sound_wave

    if physics == "mhd":
        from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave
        benchmarks = [
            BenchmarkSpec(
                label="FD (Pallas)",
                base_config=SimulationConfig(mhd=True, **fd_pallas, **common),
                cfl=1.5,
            )
        ]
        return benchmarks, setup_cp_alfven_wave

    raise ValueError(f"unknown physics {physics!r} (expected 'hydro' or 'mhd')")


def measure(physics, gpus):
    """Run the strong-scaling sweep for one (physics, GPU-count) and cache it.

    Args:
        physics: Either ``"hydro"`` or ``"mhd"``.
        gpus: The multi-GPU device count to compare against a single GPU.
    """
    import jax

    from _benchmark_utils import run_strong_scaling

    benchmarks, setup_fn = _benchmark_and_setup(physics)
    result = run_strong_scaling(
        benchmarks,
        N_values=N_VALUES,
        setup_fn=setup_fn,
        num_gpus=gpus,
        name=f"{physics}_{gpus}gpu",
        title=f"{physics} strong scaling",
        data_dir=str(DATA_DIR),
        figure_dir=str(FIG_DIR),
    )
    record = result["FD (Pallas)"]
    speedup = np.array(record["runtime_1"]) / np.array(record["runtime_N"])
    hardware = jax.devices()[0].device_kind
    np.savez(
        _cache_path(physics, gpus),
        N=np.array(N_VALUES),
        speedup=speedup,
        gpus=gpus,
        hardware=hardware,
    )
    print(f"cached {physics} @ {gpus} GPU ({hardware}) -> {_cache_path(physics, gpus)}")


def plot():
    """Assemble the combined speedup figure from whatever curves are cached."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    plotted_gpu_counts = set()
    for physics, gpus, style in CURVES:
        path = _cache_path(physics, gpus)
        if not path.exists():
            print(f"skipping {physics} @ {gpus} GPU (no cache at {path})")
            continue
        cached = np.load(path, allow_pickle=True)
        hardware = str(cached["hardware"])
        ax.plot(
            cached["N"],
            cached["speedup"],
            label=f"{physics}, {gpus} GPU ({hardware})",
            linewidth=2,
            **style,
        )
        plotted_gpu_counts.add(int(gpus))

    # Ideal-speedup guide lines for each GPU count that appears in the figure.
    for gpus in sorted(plotted_gpu_counts):
        ax.axhline(gpus, color="gray", linestyle="--", alpha=0.6)
        ax.text(N_VALUES[0], gpus, f"ideal {gpus}x", va="bottom", fontsize=9,
                color="gray")

    ax.set_xscale("log")
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_xlabel(r"N  (grid $2N \times N \times N$)")
    ax.set_ylabel(r"strong-scaling speedup $T(1) / T(G)$")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "strong_speedup_hydro_mhd.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    if PLOT_ONLY:
        plot()
    else:
        measure(PHYSICS, NUM_GPUS)
