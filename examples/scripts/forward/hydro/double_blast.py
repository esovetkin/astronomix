"""Double blast-wave problem (Woodward & Colella 1984) — paper figure.

Generates ``figures/double_blast.pdf`` (and a ``.png`` copy): the gas density
at t = 0.038 for four solver configurations,

    - FV (HLL, minmod)   at   400 cells
    - FV (HLLC, minmod)  at   400 cells
    - FD (WENO)          at   400 cells
    - FV (HLL, minmod)   at 10000 cells   (reference)

The 10000-cell HLL run serves as a converged reference; the three 400-cell
runs show how the different schemes resolve the colliding-blast structure at
coarse resolution.

Run with the repo on PYTHONPATH (GPU picked via autocvd):

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/double_blast.py [--rerun]
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    CARTESIAN,
    REFLECTIVE_BOUNDARY,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLL,
    HLLC,
    MINMOD,
    NATIVE_JAX,
    PALLAS,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings1D,
    BackendConfig,
)

# astronomix functions
from astronomix import (
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
    construct_primitive_state,
)


# shared hydro-figure helpers
from _common import (
    DATA_DIR,
    FIG_DIR,
    FD_COLOR,
    FV_HLLC_COLOR,
    FV_HLL_COLOR,
    rerun_requested,
)

BOX_SIZE = 1.0
T_END = 0.038
GAMMA = 1.4

CACHE = DATA_DIR / "double_blast.npz"

# One entry per curve in the figure, as
# (key, label, solver_mode, riemann_solver, num_cells, color, style).
RUNS = [
    ("fv_hll_400", "FV (HLL, minmod), 400", FINITE_VOLUME, HLL, 400, FV_HLL_COLOR, "marker"),
    ("fv_hllc_400", "FV (HLLC, minmod), 400", FINITE_VOLUME, HLLC, 400, FV_HLLC_COLOR, "marker"),
    ("fd_400", "FD, 400", FINITE_DIFFERENCE, HLL, 400, FD_COLOR, "marker"),
    ("fv_hll_10000", "FV (HLL, minmod), 10000 (reference)", FINITE_VOLUME, HLL, 10000, "0.3", "line"),
]


def simulate(solver_mode, riemann_solver, num_cells):
    """Run one double-blast simulation and return its final density profile.

    Args:
        solver_mode: The solver family, FINITE_VOLUME or FINITE_DIFFERENCE.
        riemann_solver: The Riemann solver to use (e.g. HLL, HLLC).
        num_cells: The number of grid cells along the 1D domain.

    Returns:
        A tuple ``(positions, density)`` of numpy arrays: the cell-centre
        positions and the final gas density at ``T_END``.
    """
    is_finite_difference = solver_mode == FINITE_DIFFERENCE
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=solver_mode,
        boundary_settings=BoundarySettings1D(
            left_boundary=REFLECTIVE_BOUNDARY, right_boundary=REFLECTIVE_BOUNDARY
        ),
        first_order_fallback=False,
        riemann_solver=riemann_solver,
        limiter=MINMOD,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        dimensionality=1,
        mhd=False,
        # FD/WENO uses the Pallas backend (bit-compatible with native JAX).
        backend_config=BackendConfig(
            backend=PALLAS if is_finite_difference else NATIVE_JAX,
            pallas_block_shape=(4, 1, 1),
        ),
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # Woodward & Colella initial condition: uniform density at rest with two
    # high-pressure regions near the reflecting walls, quiescent in between.
    positions = helper_data.geometric_centers
    density = jnp.ones_like(positions)
    velocity_x = jnp.zeros_like(positions)
    gas_pressure = jnp.where(
        positions > 0.9, 100.0, jnp.where(positions < 0.1, 1000.0, 0.01)
    )

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        gas_pressure=gas_pressure,
    )

    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, gamma=GAMMA)

    final_state = time_integration(initial_state, config, params, registered_variables)
    final_density = final_state[registered_variables.density_index]
    return np.asarray(positions), np.asarray(final_density)


def run_and_cache():
    """Run every configuration in ``RUNS``, cache the profiles and return them.

    Returns:
        A dict of numpy arrays keyed ``<key>_r`` / ``<key>_rho`` per run, as
        stored in the ``.npz`` cache.
    """
    cached_profiles = {}
    for key, label, solver_mode, riemann_solver, num_cells, _, _ in RUNS:
        print(f"running {label} ...")
        positions, density = simulate(solver_mode, riemann_solver, num_cells)
        cached_profiles[f"{key}_r"] = positions
        cached_profiles[f"{key}_rho"] = density
    np.savez(CACHE, **cached_profiles)
    print(f"cached -> {CACHE}")
    return cached_profiles


def plot(data):
    """Draw the double-blast density figure from cached profiles.

    Args:
        data: The dict of cached profiles (see :func:`run_and_cache`).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for key, label, _, _, _, color, style in RUNS:
        positions = data[f"{key}_r"]
        density = data[f"{key}_rho"]
        # The coarse 400-cell runs are drawn as scatter markers so individual
        # cells stay visible; the 10000-cell reference is drawn as a line.
        if style == "marker":
            ax.plot(
                positions,
                density,
                label=label,
                color=color,
                marker=".",
                markersize=3,
                linestyle="None",
            )
        else:
            ax.plot(positions, density, label=label, color=color, lw=1.5)

    ax.set_xlabel("position")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 7)

    plt.tight_layout()
    out = FIG_DIR / "double_blast.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    if rerun_requested() or not CACHE.exists():
        data = run_and_cache()
    else:
        data = dict(np.load(CACHE))
    plot(data)
