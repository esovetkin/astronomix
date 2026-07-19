"""3D linear sound-wave convergence — paper figure (``sound_wave3D_convergence.svg``).

Runs the L1-convergence + runtime sweep over N = 8..128 for the three pure-hydro
methods-paper configurations — FV (JAX), FD (JAX) and FD (Pallas) — against the
analytic linear sound-wave solution, and writes

    figures/sound_wave3D_convergence.svg   (average L1 error vs N, paper figure)
    figures/sound_wave3D_runtime.svg        (error/runtime diagnostics)

The fast correctness version (single low resolution) lives in
``pytests/hydrodynamics/sound_wave3D.py``; this script is the full sweep that
produces the paper figure. It reuses the shared benchmark harness in
``pytests/_benchmark_utils.py``.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/sound_wave_convergence.py
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import os
import sys
from pathlib import Path

# jax
import jax

# The tiny perturbation amplitudes of a linear sound wave need double precision
# to stay above float rounding noise.
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
from astronomix.test_setups.hydrodynamics.sound_wave3D import (
    setup_sound_wave,
    sound_wave_solution,
)


# The shared benchmark harness lives in pytests/_benchmark_utils.py; put the
# pytests directory on the path so this example can reuse the exact sweep driver
# that the fast correctness test also builds on.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
_PYTESTS_DIR = str(_REPO / "pytests")
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)

from _benchmark_utils import (  # noqa: E402
    BenchmarkSpec,
    run_convergence_and_runtime,
)

DATA_DIR = str(_HERE / "data" / "sound_wave3D")
FIG_DIR = str(_HERE / "figures")

# The resolution sweep for the convergence figure.
N_VALUES = [8, 16, 32, 64, 128]

# Options shared by every benchmark configuration; only the backend, solver
# mode and CFL number differ between them.
_common_kwargs = dict(
    box_size=StaticFloatVector(3.0, 1.5, 1.5),
    mhd=False,
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
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
]


def _error_indices(registered_variables):
    """State indices whose L1 error the convergence sweep tracks."""
    return (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
    )


if __name__ == "__main__":
    run_convergence_and_runtime(
        BENCHMARKS,
        N_values=N_VALUES,
        setup_fn=setup_sound_wave,
        analytic_fn=sound_wave_solution,
        error_var_indices_fn=_error_indices,
        name="sound_wave3D",
        title="3D linear sound wave",
        data_dir=DATA_DIR,
        figure_dir=FIG_DIR,
    )
