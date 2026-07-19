"""3D circularly-polarized Alfvén-wave convergence — paper figure (double precision).

Runs the L1-convergence + runtime sweep over N = 8..128 in *double precision*
for the two MHD methods-paper configurations — FV (JAX) and FD (Pallas) — against
the analytic CP-Alfvén-wave solution, with the AthenaPK references overlaid, and
writes

    figures/alfven_wave3D_dp_convergence.svg   (average L1 error vs N, paper figure)
    figures/alfven_wave3D_dp_runtime.svg        (error/runtime diagnostics)

The fast correctness version (single low resolution) lives in
``pytests/mhd/alfven_wave3D.py``; this script is the full double-precision sweep
that produces the paper figure. It reuses the shared benchmark harness in
``pytests/_benchmark_utils.py`` and the AthenaPK reference NPZs (measured by
P. Grete, single meshblock + ``minimum_number_of_teams_for_boundary_kernel=256``)
shipped under ``pytests/mhd/data/athenapk/<node>/``.

Runtimes are only comparable within one GPU generation, so results are stored
per node: the sweep derives its node tag from the live device (a100 / h100 /
h200 / ...) and writes

    pytests/mhd/data/astronomix/<node>/alfven_wave3D_dp_convergence_<benchmark>.npz
    pytests/mhd/figures/<node>/alfven_wave3D_dp_{convergence,runtime}.svg

overlaying the AthenaPK references measured on that SAME node. Nothing is
hand-copied and two nodes never write the same file, so a run cannot be filed
under the wrong hardware.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/alfven_convergence.py

To generate *all* the astronomix Alfvén data (this sweep plus the sp/dp
per-iteration runs) with one command, use ``generate_alfven_data.py``.

With ``--replot`` the sweep is skipped and the figures are regenerated on the
CPU from the NPZs saved by previous runs -- by default for every node found on
disk, or for one node with ``--node <tag>``:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/alfven_convergence.py --replot
"""

# general
# NOTE: os/sys must come first — --replot decides whether we grab a GPU via
# autocvd or stay on the CPU, and both must happen before jax is imported.
import os
import sys
from pathlib import Path

REPLOT = "--replot" in sys.argv
# --approx-rsqrt: run the FD (Pallas) benchmark with the approximate-rsqrt fast
# path (BackendConfig.use_approximate_rsqrt).  Results are stored under a
# ``<node>_rsqrt`` tag so they sit alongside — never overwrite — the IEEE
# baseline; the AthenaPK overlay still resolves from the base hardware node.
APPROX_RSQRT = "--approx-rsqrt" in sys.argv

# ==== GPU selection ====
if REPLOT:
    # Replotting touches no jax ops; keep jax off the GPUs entirely.
    os.environ["JAX_PLATFORMS"] = "cpu"
elif os.environ.get("CUDA_VISIBLE_DEVICES"):
    # A parent (generate_alfven_data.py) already picked the GPU via autocvd and
    # exported it; inherit that choice so every sweep it drives is measured on
    # the same device. Selection still goes through autocvd, just once.
    pass
else:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# jax
import jax

# Double precision: the paper's Alfvén-wave convergence figure is the x64 sweep.
jax.config.update("jax_enable_x64", True)

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    NATIVE_JAX,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SnapshotSettings,
    BackendConfig,
)
from astronomix.option_classes.simulation_config import StaticFloatVector

# astronomix functions
from astronomix.test_setups.mhd.alfven_wave3D import (
    setup_cp_alfven_wave,
    cp_alfven_wave_solution,
)


# The shared benchmark harness and the AthenaPK reference both live under
# pytests/; put that directory on the path so this example reuses the exact
# sweep driver the fast correctness test also builds on.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
_PYTESTS_DIR = str(_REPO / "pytests")
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)

from _benchmark_utils import (  # noqa: E402
    BenchmarkSpec,
    _slug,
    node_tag,
    plot_convergence_and_runtime,
    run_convergence_and_runtime,
)

# Everything is stored per GPU generation: runtimes measured on different
# hardware must never share a file (or a figure) with each other.
DATA_ROOT = _REPO / "pytests" / "mhd" / "data" / "astronomix"
FIG_ROOT = _REPO / "pytests" / "mhd" / "figures"
ATHENAPK_ROOT = _REPO / "pytests" / "mhd" / "data" / "athenapk"

NAME = "alfven_wave3D_dp"

# The convergence figure plots L1 error, which is a property of the scheme and
# is bit-identical on every GPU — naming hardware there would imply it matters.
# Only the runtime figure is hardware-dependent, so only it carries the node.
TITLE = "3D CP Alfvén wave (double precision)"


def _hw_node(tag: str) -> str:
    """Base hardware node for a storage tag (``a100_rsqrt`` -> ``a100``)."""
    return tag[: -len("_rsqrt")] if tag.endswith("_rsqrt") else tag


def _runtime_title(tag: str) -> str:
    # Title carries only the hardware node — not the rsqrt variant (the storage
    # tag / folder already distinguishes rsqrt runs from the IEEE baseline).
    return f"{TITLE} — {_hw_node(tag).upper()}"


def _data_dir(node: str) -> str:
    return str(DATA_ROOT / node)


def _fig_dir(node: str) -> str:
    return str(FIG_ROOT / node)


def _reference_npzs(node: str):
    """AthenaPK overlays measured on ``node``.

    References by P. Grete, double precision, single meshblock per rank with
    parthenon/mesh/minimum_number_of_teams_for_boundary_kernel=256 (without
    which the ghost-cell communication is inefficient): the standard
    configuration (VL2 + PLM + HLLD, CFL 0.3) and the higher-order variant
    (RK3 + PPM + HLLD, CFL 0.4) that reaches L1 ~ 1e-5 at N=128.

    Only same-node references are overlaid -- the runtime panel puts astronomix
    and AthenaPK on shared time-to-error axes, so mixing hardware there would
    invent a speed difference that is really a GPU difference.
    """
    node_dir = ATHENAPK_ROOT / _hw_node(node)
    candidates = [
        ("AthenaPK (VL2+PLM)", node_dir / "cfl03_teams256_singleblock" / "athenapk_alfven_convergence.npz"),
        ("AthenaPK (RK3+PPM)", node_dir / "cfl04_teams256_singleblock_rk3_ppm" / "athenapk_alfven_convergence.npz"),
    ]
    present = [(label, str(p)) for label, p in candidates if p.exists()]
    if not present:
        print(f"[alfven] no AthenaPK reference for node {node!r} — plotting astronomix only")
    return present


def _nodes_on_disk() -> list:
    """Node tags that already have a full set of NPZs under DATA_ROOT."""
    if not DATA_ROOT.exists():
        return []
    nodes = []
    for d in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        if all((d / f"{NAME}_convergence_{_slug(s.label)}.npz").exists() for s in BENCHMARKS):
            nodes.append(d.name)
    return nodes

# The resolution sweep for the convergence figure.
N_VALUES = [8, 16, 32, 64, 128]

# Options shared by every benchmark configuration; only the backend, solver
# mode and CFL number differ between them.
_common_kwargs = dict(
    box_size=StaticFloatVector(3.0, 1.5, 1.5),
    mhd=True,
    dimensionality=3,
    progress_bar=False,
    memory_analysis=True,
    print_elapsed_time=True,
    return_snapshots=True,
    snapshot_settings=SnapshotSettings(return_final_state=True),
)

BENCHMARKS = [
    BenchmarkSpec(
        label="FV (JAX)",
        base_config=SimulationConfig(
            backend_config=BackendConfig(backend=NATIVE_JAX),
            solver_mode=FINITE_VOLUME,
            **_common_kwargs,
        ),
        cfl=0.4,
    ),
    BenchmarkSpec(
        label="FD (JAX)",
        base_config=SimulationConfig(
            backend_config=BackendConfig(backend=NATIVE_JAX),
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
    BenchmarkSpec(
        label="FD (Pallas)",
        base_config=SimulationConfig(
            # Only the Pallas MHD kernel honours use_approximate_rsqrt; it is a
            # no-op for the two native benchmarks above, so it is set here only.
            backend_config=BackendConfig(use_approximate_rsqrt=APPROX_RSQRT),
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
]


def _error_indices(registered_variables):
    """State-array indices entered into the L1 error norm (rho, v, p, B)."""
    return (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
        registered_variables.magnetic_index.x,
        registered_variables.magnetic_index.y,
        registered_variables.magnetic_index.z,
    )


def _load_saved_results(node: str) -> dict:
    """Load the per-benchmark sweep NPZs saved by a previous run on ``node``."""
    results = {}
    for spec in BENCHMARKS:
        npz_path = Path(_data_dir(node)) / f"{NAME}_convergence_{_slug(spec.label)}.npz"
        data = np.load(npz_path)
        results[spec.label] = dict(
            N=data["N_values"],
            l1=data["l1_errors"],
            runtime=data["runtimes"],
            iterations=data["iterations"],
        )
    return results


def _replot(node: str) -> None:
    plot_convergence_and_runtime(
        _load_saved_results(node),
        N_VALUES,
        name=NAME,
        title=TITLE,
        runtime_title=_runtime_title(node),
        figure_dir=_fig_dir(node),
        reference_npzs=_reference_npzs(node),
    )
    print(f"[alfven] replotted {node} -> {_fig_dir(node)}")


if __name__ == "__main__":
    if REPLOT:
        # --node <tag> replots one node; otherwise every node found on disk, so
        # a single --replot refreshes all hardware after e.g. a plotting tweak.
        if "--node" in sys.argv:
            nodes = [sys.argv[sys.argv.index("--node") + 1].lower()]
        else:
            nodes = _nodes_on_disk()
        if not nodes:
            raise SystemExit(f"no complete sweep data under {DATA_ROOT}; run the sweep on a GPU first")
        for node in nodes:
            _replot(node)
    else:
        node = node_tag() + ("_rsqrt" if APPROX_RSQRT else "")
        print(f"[alfven] running on {jax.devices()[0].device_kind} -> storing under node {node!r}"
              + (" (approximate rsqrt)" if APPROX_RSQRT else ""))
        run_convergence_and_runtime(
            BENCHMARKS,
            N_values=N_VALUES,
            setup_fn=setup_cp_alfven_wave,
            analytic_fn=cp_alfven_wave_solution,
            error_var_indices_fn=_error_indices,
            name=NAME,
            title=TITLE,
            runtime_title=_runtime_title(node),
            data_dir=_data_dir(node),
            figure_dir=_fig_dir(node),
            reference_npzs=_reference_npzs(node),
        )
        print(f"[alfven] wrote {_data_dir(node)} and {_fig_dir(node)}")
