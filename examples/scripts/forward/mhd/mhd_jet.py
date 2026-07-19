"""Magnetically driven jet (3D MHD) — paper figure, finite-difference solver.

A magnetic tower launched from a magnetized central region in an initially
uniform medium (vector-potential initial condition), evolved with the
finite-difference (WENO + constrained-transport) solver.  This is the
FD-only version of the jet test.

The figure is a density slice through the jet axis at t = 5.0.  The full 3D
density field is cached under ``data/`` so the figure can be re-sliced /
re-styled without re-running the (expensive) 256^3 simulation.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/mhd_jet.py [--res N] [--gpus M] [--rerun]

The run is domain-decomposed across ``--gpus M`` devices (x-axis split); the
paper's high-resolution figure is ``--res 1024 --gpus 4``.
"""

# general
import sys


def _num_gpus_from_argv():
    """Return the GPU count requested via ``--gpus N`` (default 1)."""
    if "--gpus" in sys.argv:
        return int(sys.argv[sys.argv.index("--gpus") + 1])
    return 1


def _block_shape_from_argv():
    """Return the Pallas block shape requested via ``--block bx,by,bz``.

    ``None`` (the default) lets astronomix pick its own default block shape.
    A smaller block trades a little kernel throughput for a lower peak-memory
    footprint (smaller halo padding + per-program tile temporaries) and is
    bit-identical, so it is a safe knob for fitting a large run on-device.
    """
    if "--block" in sys.argv:
        raw = sys.argv[sys.argv.index("--block") + 1]
        return tuple(int(p) for p in raw.split(","))
    return None


# ==== GPU selection ====
from autocvd import autocvd
NUM_GPUS = _num_gpus_from_argv()
BLOCK_SHAPE = _block_shape_from_argv()
autocvd(num_gpus=NUM_GPUS)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# jax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

# astronomix constants
from astronomix import PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import (
    RK4_LSRK,
    RK4_SSP,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
    BackendConfig,
)

# astronomix functions
from astronomix import (
    get_registered_variables,
    construct_primitive_state,
    time_integration,
    finalize_config,
    setup_magnetic_fields_from_vector_potential,
)


# shared figure helpers
from _common import DATA_DIR
from _common import FIG_DIR
from _common import rerun_requested
from _common import mhd_registered_variables


GAMMA = 5.0 / 3.0
BOX_SIZE = 24.0
T_END = 5.0
C_CFL = 0.8
RHO_0 = 1.0
P_0 = 1.0
A0 = 20.0


def _resolution_from_argv():
    """Return the per-dimension resolution requested via ``--res N`` (default 256)."""
    if "--res" in sys.argv:
        return int(sys.argv[sys.argv.index("--res") + 1])
    return 256


def cache_path(num_cells):
    """Return the cache path for the FD jet run at ``num_cells``^3."""
    return DATA_DIR / f"mhd_jet_fd_{num_cells}.npz"


def simulate(num_cells):
    """Run the finite-difference MHD jet simulation at ``num_cells``^3.

    Sets up the magnetic-tower initial condition from a vector potential in an
    initially uniform medium, evolves it to ``T_END`` with the finite-difference
    (WENO + constrained-transport) solver on the Pallas backend, and returns the
    final density field.

    Args:
        num_cells: The per-dimension resolution of the cubic grid.

    Returns:
        The final density field as a numpy array of shape
        ``(num_cells, num_cells, num_cells)``.
    """

    # -------------------------------------------------------------
    # ============ ↓ Grid and solver configuration ↓ =============
    # -------------------------------------------------------------

    grid_spacing = BOX_SIZE / num_cells
    center = BOX_SIZE / 2.0

    config = SimulationConfig(
        positivity_config=PositivityConfig(
            default_positivity_protection=False,
        ),
        # FD/WENO runs ~10x faster through the Pallas (Triton) backend;
        # bit-compatible with native JAX.
        grid_spacing=grid_spacing,
        mhd=True,
        progress_bar=True,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        # Memory savers are enabled only for the multi-GPU (high-resolution)
        # path so the single-GPU run reproduces the paper's 256^3 figure
        # byte-for-byte (default SSPRK4, no buffer donation).  For NUM_GPUS > 1
        # we donate the state buffer (numerically neutral) and switch to the
        # 2N-storage low-memory RK4 (one fewer full-state buffer than SSPRK4)
        # to fit the 1024^3 run across the devices; see multi_gpu.py.
        donate_state=(NUM_GPUS > 1),
        time_integrator=(RK4_LSRK if NUM_GPUS > 1 else RK4_SSP),
        # Optional smaller Pallas block shape (``--block bx,by,bz``): a
        # numerics-preserving lever to shave peak device memory (smaller halo
        # padding + tile temporaries). ``None`` keeps the astronomix default.
        backend_config=BackendConfig(
            pallas_block_shape=BLOCK_SHAPE,
        ),
    )
    rv = get_registered_variables(config)

    # Domain-decompose along the x axis across the GPUs. With NUM_GPUS == 1 this
    # is a trivial single-device mesh, so the single- and multi-GPU paths share
    # exactly the same code (and the same numerics).
    mesh = jax.make_mesh((1, NUM_GPUS, 1, 1), (VARAXIS, XAXIS, YAXIS, ZAXIS))
    sharding = NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))

    # -------------------------------------------------------------
    # ============ ↑ Grid and solver configuration ↑ =============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ Initial magnetic tower ↓ =================
    # -------------------------------------------------------------

    # The magnetic field is seeded from a vector potential so that the
    # constrained-transport solver starts from a divergence-free field.
    def jet_vector_potential(X, Y, Z):
        r = jnp.sqrt((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)
        A_x = -jnp.exp(-r**2) * (Y - center)
        A_y = jnp.exp(-r**2) * (X - center)
        A_z = 0.5 * A0 * jnp.exp(-r**2)
        return A_x, A_y, A_z

    # -------------------------------------------------------------
    # =============== ↑ Initial magnetic tower ↑ =================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Uniform background and evolve ↓ =============
    # -------------------------------------------------------------

    def build_initial_state():
        """Assemble the sharded initial primitive state on the devices.

        The vector-potential magnetic field and the uniform background fields
        are built inside this closure so that, under the ``out_shardings`` jit
        below, XLA partitions the whole construction and every device only ever
        computes and stores *its own shard* of each field (strategy 3 in
        ``examples/scripts/multi_gpu/multi_gpu.py``). No single device holds the
        full state, which is what makes the 1024^3 run fit.
        """
        B_x, B_y, B_z, bxb, byb, bzb = setup_magnetic_fields_from_vector_potential(
            config=config,
            vector_potential_func=jet_vector_potential,
        )

        shape = (num_cells, num_cells, num_cells)
        rho = jnp.ones(shape) * RHO_0
        zeros = jnp.zeros(shape)
        p = jnp.ones(shape) * P_0

        return construct_primitive_state(
            config=config,
            registered_variables=rv,
            density=rho,
            velocity_x=zeros,
            velocity_y=zeros,
            velocity_z=zeros,
            gas_pressure=p,
            magnetic_field_x=B_x,
            magnetic_field_y=B_y,
            magnetic_field_z=B_z,
            interface_magnetic_field_x=bxb,
            interface_magnetic_field_y=byb,
            interface_magnetic_field_z=bzb,
            sharding=sharding,
        )

    # Build the state directly on the individual GPUs — XLA partitions the whole
    # construction, so no single device ever holds the full state.
    initial_state = jax.jit(build_initial_state, out_shardings=sharding)()

    if NUM_GPUS > 1:
        # Confirm the domain decomposition (x-axis split across the devices).
        jax.debug.visualize_array_sharding(
            initial_state[rv.density_index, :, :, num_cells // 2]
        )

    params = SimulationParams(
        C_cfl=C_CFL,
        dt_max=0.1,
        t_end=T_END,
        gamma=GAMMA,
        minimum_density=1e-4 * RHO_0,
        minimum_pressure=1e-4 * P_0,
    )

    config = finalize_config(config, initial_state.shape)
    final_state = time_integration(
        initial_state, config, params, rv, sharding=sharding
    )

    density = np.asarray(final_state[rv.density_index])
    return density

    # -------------------------------------------------------------
    # ============ ↑ Uniform background and evolve ↑ =============
    # -------------------------------------------------------------


def get_run(num_cells, rerun):
    """Return the FD jet density field, from cache or by (re-)running.

    Args:
        num_cells: The per-dimension resolution of the run.
        rerun: When True, ignore any cache and re-run the simulation.

    Returns:
        The final density field as a numpy array.
    """
    path = cache_path(num_cells)
    if path.exists() and not rerun:
        return np.load(path)["density"]
    print(f"running FD MHD jet at {num_cells}^3 ...")
    density = simulate(num_cells)
    np.savez(path, density=density)
    print(f"  cached -> {path}")
    return density


def plot(num_cells, rerun):
    """Render the jet-axis density slice figure.

    Loads (or regenerates) the cached density field, slices it through the jet
    axis (the x-z plane at mid-y) and saves the figure as both ``.png`` and
    ``.pdf`` (``mhd_jet_fd_{num_cells}``).

    Args:
        num_cells: The per-dimension resolution of the run to plot.
        rerun: When True, force the underlying simulation to be re-run.
    """
    density = get_run(num_cells, rerun)
    y_index = num_cells // 2

    # Slice through the jet axis: the x-z plane at mid-y.
    density_slice = density[:, y_index, :].T

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    image = ax.imshow(
        density_slice,
        origin="lower",
        extent=(0, BOX_SIZE, 0, BOX_SIZE),
        cmap="YlOrRd",
    )
    ax.set_aspect("equal", adjustable="box")
    # Bare density panel: no axis labels or tick numbers.
    ax.set_xticks([])
    ax.set_yticks([])
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(image, cax=cax, label="density")

    plt.tight_layout()
    # High raster resolution so the density panel stays crisp — including the
    # imshow embedded (rasterized) inside the PDF/SVG vector outputs, which
    # otherwise defaults to the figure's ~100 dpi.
    dpi = 600
    out = FIG_DIR / f"mhd_jet_fd_{num_cells}.png"
    fig.savefig(out, dpi=dpi)
    # Vector outputs cropped tight to the drawn content (panel + colorbar).
    for suffix in (".pdf", ".svg"):
        fig.savefig(out.with_suffix(suffix), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    num_cells = _resolution_from_argv()
    plot(num_cells, rerun_requested())
