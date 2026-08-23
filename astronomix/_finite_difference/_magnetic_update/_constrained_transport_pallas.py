"""Pallas backend for the Constrained-Transport (CT) helpers.

This file is the **Pallas backend** for ``_constrained_transport.py``.
Two CT helpers are ported here:

* ``update_cell_center_fields`` — one Pallas kernel per call.  Just three
  ``interp_face_to_center`` stencils (halo 3 per axis, no chained
  closures) plus the cell-centered B / energy update.  Compiles in well
  under a second.

* ``constrained_transport_rhs_from_slices`` — split into **three**
  bounded-halo Pallas kernels so Triton never has to lower the full
  chained EMF closure tree at once.  Each kernel is a thin per-cell
  stencil with halo ≤ 4 per axis.  Their JAX-level glue materialises one
  intermediate per stage instead of the 12+ named intermediates the
  native code carries:

    Stage 1 — ``_ct_modified_flux_pallas``
        rho, mom_*, B_*  +  6 raw B-flux slices
          → 6 ``B_flux_axis_mod`` slices  (halo 2 along one axis each).

    Stage 2 — ``_ct_edge_emf_pallas``
        6 modified fluxes  →  Omega_z, Omega_x, Omega_y at cell edges
        (halo 2 per axis, two axes per output).

    Stage 3 — ``_ct_curl_pallas``
        Omega_z, Omega_x, Omega_y (edge values)  →  rhs_b{x,y,z}.
        Fuses ``point_values_to_averages`` smoothing with the
        ``finite_difference_int6`` curl in one tile; combined halo ≤ 4
        along the curl axis, ≤ 1 along the smoothing axis.

Multi-GPU: every public entry point routes its ``pl.pallas_call`` through
``_pallas_call_sharded`` (a bare ``pl.pallas_call`` is opaque to GSPMD,
which would ``all-gather`` the full grid on every device).  The
single-channel field slices are therefore *stacked* along a leading axis
— ``(6, nx, ny, nz)`` modified fluxes, ``(3, nx, ny, nz)`` EMFs — so
they ride the same vars-first halo-exchange machinery as the conserved
state, and each stage costs one ppermute halo exchange instead of the
many per-shift collective-permutes of the native roll-based stencils.
The ``*_local`` bodies derive every shape from their *own* arguments, so
the same kernel build runs on the global grid (single device) or a
halo-padded local shard (multi device).

The developer never touches this file by hand; the pallasify skill
regenerates it from the native ``_constrained_transport.py`` when those
change.  See ``pallas_backend_implementation_guide.md`` §4.4 for the
recipe and the diagnostic that motivated the split.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _backend_is_pallas,
    _pallas_call_sharded,
    _pallas_compiler_params,
    pl,
)


XAXIS = 0
YAXIS = 1
ZAXIS = 2


def _ct_pallas_block_ok(state_shape, config: SimulationConfig) -> bool:
    """Spatial-block divisibility check shared by all CT kernels."""
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, ndim, spatial_shape=state_shape[1:])
    for n, b in zip(state_shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _ct_block_and_grid(spatial_shape, config: SimulationConfig):
    """Block shape and Pallas grid for a (possibly halo-padded) local
    spatial shape.  Called inside every ``*_local`` body so the grid
    resizes automatically when ``_pallas_call_sharded`` hands the build a
    padded shard."""
    nx, ny, nz = (int(x) for x in spatial_shape)
    bx_blk, by_blk, bz_blk = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape, 3, spatial_shape=(nx, ny, nz)
    )
    grid = (nx // bx_blk, ny // by_blk, nz // bz_blk)
    return (nx, ny, nz), (bx_blk, by_blk, bz_blk), grid


# -----------------------------------------------------------------------------
# Cell-center reconstruction from interface B-fields (one Pallas kernel).
# -----------------------------------------------------------------------------


def _ct_update_cell_center_fields_pallas_supported(state, config: SimulationConfig) -> bool:
    """Whether the Pallas cell-center reconstruction can run.

    3D ideal-gas MHD only — the smaller iso/lower-dim paths fall through
    to the native version, which is short and unproblematic.  Enabled
    by default; the single-kernel implementation is just three
    independent face-to-center stencils with halo 3 each, so it compiles
    quickly.
    """
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.backend_config.pallas_ct:
        return False
    if not config.mhd:
        return False
    if config.equation_of_state != IDEAL_GAS:
        return False
    if int(config.dimensionality) != 3:
        return False
    if state.ndim != 4:
        return False
    return _ct_pallas_block_ok(state.shape, config)


def _ct_update_cell_center_fields_pallas(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Pallas ``update_cell_center_fields`` — public, shard-aware wrapper.

    Stacks the three interface fields along a leading axis so they ride
    the vars-first halo exchange, then routes the kernel build through
    ``_pallas_call_sharded`` (halo 3: the face-to-center stencil reads
    offsets −3..+2 along each axis).
    """
    assert _ct_update_cell_center_fields_pallas_supported(conserved_state, config)
    ndim = int(config.dimensionality)
    _, block, _ = _ct_block_and_grid(conserved_state.shape[1:], config)
    b_stacked = jnp.stack([bx_interface, by_interface, bz_interface])
    b_halo = (3, 3, 3)[:ndim]
    zero_halo = (0,) * ndim

    def _local(state_local, b_local):
        return _ct_update_cell_center_fields_pallas_local(
            state_local, b_local, config, registered_variables
        )

    return _pallas_call_sharded(
        _local,
        state_inputs=(conserved_state, b_stacked),
        halo=b_halo,
        # exchange only magnetic interface
        #
        # _ct_update_cell_center_fields_pallas_local uses
        # conserved_state -> q_ref
        # ...
        #     Bx_old = q_ref[BX, ii, jj, kk]
        #     By_old = q_ref[BY, ii, jj, kk]
        #     Bz_old = q_ref[BZ, ii, jj, kk]
        # ...
        #     E_old = q_ref[E, ii, jj, kk]
        # ...
        #         out_ref[var, ...] = q_ref[var, ii, jj, kk]
        # which are all local reads
        input_halos=(zero_halo, b_halo),
        block_shape=block[:ndim],
    )


def _ct_update_cell_center_fields_pallas_local(
    conserved_state,
    b_interfaces,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Single-shard kernel build: 3 face-to-center stencils (halo 3 per
    axis, independent stencils — no chained closures), one state
    pass-through + cell-centered B + energy fix-up.  All shapes come from
    *this* call's arguments so the build works on padded local shards.
    """
    nvars = int(conserved_state.shape[0])
    (nx, ny, nz), (bx_blk, by_blk, bz_blk), grid = _ct_block_and_grid(
        conserved_state.shape[1:], config
    )

    BX = int(registered_variables.magnetic_index.x)
    BY = int(registered_variables.magnetic_index.y)
    BZ = int(registered_variables.magnetic_index.z)
    E = int(registered_variables.pressure_index)

    state_out_spec = pl.BlockSpec((nvars, bx_blk, by_blk, bz_blk),
                                  lambda bi, bj, bk: (0, bi, bj, bk))
    state_in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    b_in_spec = pl.BlockSpec(b_interfaces.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    def kernel(q_ref, b_ref, out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
        jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
        kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        # interp_face_to_center coefficients (3, -25, 150, 150, -25, 3) / 256
        # over offsets (-3, -2, -1, 0, 1, 2) along the axis: see the native
        # ``interp_face_to_center`` derivation in ``_interpolate.py``.
        def f2c_x():
            return (
                3.0  * b_ref[0, (ii - 3) % nx, jj, kk]
              - 25.0  * b_ref[0, (ii - 2) % nx, jj, kk]
              + 150.0 * b_ref[0, (ii - 1) % nx, jj, kk]
              + 150.0 * b_ref[0, ii,            jj, kk]
              - 25.0  * b_ref[0, (ii + 1) % nx, jj, kk]
              + 3.0   * b_ref[0, (ii + 2) % nx, jj, kk]
            ) / 256.0

        def f2c_y():
            return (
                3.0  * b_ref[1, ii, (jj - 3) % ny, kk]
              - 25.0  * b_ref[1, ii, (jj - 2) % ny, kk]
              + 150.0 * b_ref[1, ii, (jj - 1) % ny, kk]
              + 150.0 * b_ref[1, ii, jj,            kk]
              - 25.0  * b_ref[1, ii, (jj + 1) % ny, kk]
              + 3.0   * b_ref[1, ii, (jj + 2) % ny, kk]
            ) / 256.0

        def f2c_z():
            return (
                3.0  * b_ref[2, ii, jj, (kk - 3) % nz]
              - 25.0  * b_ref[2, ii, jj, (kk - 2) % nz]
              + 150.0 * b_ref[2, ii, jj, (kk - 1) % nz]
              + 150.0 * b_ref[2, ii, jj, kk]
              - 25.0  * b_ref[2, ii, jj, (kk + 1) % nz]
              + 3.0   * b_ref[2, ii, jj, (kk + 2) % nz]
            ) / 256.0

        Bx_center = f2c_x()
        By_center = f2c_y()
        Bz_center = f2c_z()

        Bx_old = q_ref[BX, ii, jj, kk]
        By_old = q_ref[BY, ii, jj, kk]
        Bz_old = q_ref[BZ, ii, jj, kk]
        b2_old = Bx_old * Bx_old + By_old * By_old + Bz_old * Bz_old
        b2_new = Bx_center * Bx_center + By_center * By_center + Bz_center * Bz_center
        E_old = q_ref[E, ii, jj, kk]
        E_new = E_old + 0.5 * (b2_new - b2_old)

        for var in range(nvars):
            if var == BX:
                out_ref[var, ...] = Bx_center
            elif var == BY:
                out_ref[var, ...] = By_center
            elif var == BZ:
                out_ref[var, ...] = Bz_center
            elif var == E:
                out_ref[var, ...] = E_new
            else:
                out_ref[var, ...] = q_ref[var, ii, jj, kk]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[state_in_spec, b_in_spec],
        out_specs=state_out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_update_cell_center_fields",
        **kwargs,
    )(conserved_state, b_interfaces)


# -----------------------------------------------------------------------------
# CT EMF — split into three bounded-halo Pallas kernels.
# -----------------------------------------------------------------------------


def _ct_rhs_pallas_supported(state, config: SimulationConfig) -> bool:
    """Whether the staged Pallas CT-RHS path can run (3D only).  Gated
    on ``config.backend_config.pallas_ct`` (default off): the staged kernel chain is
    correct and stable but adds ~25 s of one-time compile cost while
    giving only marginal memory savings at production grid sizes."""
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.backend_config.pallas_ct:
        return False
    if not config.mhd:
        return False
    if int(config.dimensionality) != 3:
        return False
    if state.ndim != 4:
        return False
    return _ct_pallas_block_ok(state.shape, config)


def _ct_modified_flux_pallas(
    conserved_state,
    flux_slices,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Stage 1 public wrapper: per-axis modified magnetic flux.

    ``flux_slices`` is the stacked ``(6, nx, ny, nz)`` array of raw WENO
    magnetic-flux slices in the order (By_fx, Bz_fx, Bx_fy, Bz_fy,
    Bx_fz, By_fz); the output keeps the same layout.  Halo 2: the
    center-to-face stencil reads offsets −1..+2 along one axis per slice.
    """
    ndim = int(config.dimensionality)
    _, block, _ = _ct_block_and_grid(conserved_state.shape[1:], config)
    q_halo = (2, 2, 2)[:ndim]
    zero_halo = (0,) * ndim

    def _local(state_local, flux_local):
        return _ct_modified_flux_pallas_local(
            state_local, flux_local, config, registered_variables
        )

    return _pallas_call_sharded(
        _local,
        state_inputs=(conserved_state, flux_slices),
        halo=q_halo,
        # exchange only conserved state
        #
        # _ct_modified_flux_pallas_local uses flux_slices -> f_ref
        #
        # ...
        #     out_ref[0, ...] = f_ref[0, ii, jj, kk] + c2f(Bvy_at_x)
        #     out_ref[1, ...] = f_ref[1, ii, jj, kk] + c2f(Bvz_at_x)
        #     out_ref[2, ...] = f_ref[2, ii, jj, kk] + c2f(Bvx_at_y)
        #     out_ref[3, ...] = f_ref[3, ii, jj, kk] + c2f(Bvz_at_y)
        #     out_ref[4, ...] = f_ref[4, ii, jj, kk] + c2f(Bvx_at_z)
        #     out_ref[5, ...] = f_ref[5, ii, jj, kk] + c2f(Bvy_at_z)
        # ...
        # which are all local reads
        input_halos=(q_halo, zero_halo),
        block_shape=block[:ndim],
    )


def _ct_modified_flux_pallas_local(
    conserved_state,
    flux_slices,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Stage 1 single-shard build: per-axis modified magnetic-flux
    (Eq. 12–17).

    Six output channels, each obtained by adding to the raw WENO magnetic
    flux one ``interp_center_to_face`` of a cell-centered product:

      out[0] = By_flux_x + interp_c2f_x(Bx * vy)
      out[1] = Bz_flux_x + interp_c2f_x(Bx * vz)
      out[2] = Bx_flux_y + interp_c2f_y(By * vx)
      out[3] = Bz_flux_y + interp_c2f_y(By * vz)
      out[4] = Bx_flux_z + interp_c2f_z(Bz * vx)
      out[5] = By_flux_z + interp_c2f_z(Bz * vy)

    Worst-case stencil halo: 2 along each axis (independent per output).
    """
    (nx, ny, nz), (bx_blk, by_blk, bz_blk), grid = _ct_block_and_grid(
        conserved_state.shape[1:], config
    )

    DENSITY = int(registered_variables.density_index)
    MX = int(registered_variables.momentum_index.x)
    MY = int(registered_variables.momentum_index.y)
    MZ = int(registered_variables.momentum_index.z)
    BX = int(registered_variables.magnetic_index.x)
    BY = int(registered_variables.magnetic_index.y)
    BZ = int(registered_variables.magnetic_index.z)

    state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    flux_in_spec = pl.BlockSpec(flux_slices.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((6, bx_blk, by_blk, bz_blk),
                            lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(q_ref, f_ref, out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
        jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
        kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        # Product Bn*vm at a cell offset along ONE axis from (ii,jj,kk).
        def Bvy_at_x(off):
            rho = q_ref[DENSITY, (ii + off) % nx, jj, kk]
            return q_ref[BX, (ii + off) % nx, jj, kk] * q_ref[MY, (ii + off) % nx, jj, kk] / rho

        def Bvz_at_x(off):
            rho = q_ref[DENSITY, (ii + off) % nx, jj, kk]
            return q_ref[BX, (ii + off) % nx, jj, kk] * q_ref[MZ, (ii + off) % nx, jj, kk] / rho

        def Bvx_at_y(off):
            rho = q_ref[DENSITY, ii, (jj + off) % ny, kk]
            return q_ref[BY, ii, (jj + off) % ny, kk] * q_ref[MX, ii, (jj + off) % ny, kk] / rho

        def Bvz_at_y(off):
            rho = q_ref[DENSITY, ii, (jj + off) % ny, kk]
            return q_ref[BY, ii, (jj + off) % ny, kk] * q_ref[MZ, ii, (jj + off) % ny, kk] / rho

        def Bvx_at_z(off):
            rho = q_ref[DENSITY, ii, jj, (kk + off) % nz]
            return q_ref[BZ, ii, jj, (kk + off) % nz] * q_ref[MX, ii, jj, (kk + off) % nz] / rho

        def Bvy_at_z(off):
            rho = q_ref[DENSITY, ii, jj, (kk + off) % nz]
            return q_ref[BZ, ii, jj, (kk + off) % nz] * q_ref[MY, ii, jj, (kk + off) % nz] / rho

        # interp_center_to_face: (-f[i-1] + 9 f[i] + 9 f[i+1] - f[i+2]) / 16
        def c2f(prod):
            return (-prod(-1) + 9.0 * prod(0) + 9.0 * prod(1) - prod(2)) / 16.0

        out_ref[0, ...] = f_ref[0, ii, jj, kk] + c2f(Bvy_at_x)
        out_ref[1, ...] = f_ref[1, ii, jj, kk] + c2f(Bvz_at_x)
        out_ref[2, ...] = f_ref[2, ii, jj, kk] + c2f(Bvx_at_y)
        out_ref[3, ...] = f_ref[3, ii, jj, kk] + c2f(Bvz_at_y)
        out_ref[4, ...] = f_ref[4, ii, jj, kk] + c2f(Bvx_at_z)
        out_ref[5, ...] = f_ref[5, ii, jj, kk] + c2f(Bvy_at_z)

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(flux_slices.shape, flux_slices.dtype),
        grid=grid,
        in_specs=[state_spec, flux_in_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_modified_flux",
        **kwargs,
    )(conserved_state, flux_slices)


def _ct_edge_emf_pallas(flux_mod_slices, config: SimulationConfig):
    """Stage 2 public wrapper: edge EMFs from the stacked modified
    fluxes.  Halo 2 (center-to-face stencil along two axes per output)."""
    ndim = int(config.dimensionality)
    _, block, _ = _ct_block_and_grid(flux_mod_slices.shape[1:], config)

    def _local(flux_local):
        return _ct_edge_emf_pallas_local(flux_local, config)

    return _pallas_call_sharded(
        _local,
        state_inputs=(flux_mod_slices,),
        halo=(2, 2, 2)[:ndim],
        block_shape=block[:ndim],
    )


def _ct_edge_emf_pallas_local(flux_mod_slices, config: SimulationConfig):
    """Stage 2 single-shard build: edge EMFs Omega_z, Omega_x, Omega_y
    (Eq. 19–21), stacked as output channels (0, 1, 2):

      Omega_z = interp_c2f_x(Bx_flux_y_mod) − interp_c2f_y(By_flux_x_mod)
      Omega_x = interp_c2f_y(By_flux_z_mod) − interp_c2f_z(Bz_flux_y_mod)
      Omega_y = interp_c2f_z(Bz_flux_x_mod) − interp_c2f_x(Bx_flux_z_mod)

    Input channel order matches stage 1's output: (By_fx, Bz_fx, Bx_fy,
    Bz_fy, Bx_fz, By_fz).  Halo: 2 per axis.
    """
    (nx, ny, nz), (bx_blk, by_blk, bz_blk), grid = _ct_block_and_grid(
        flux_mod_slices.shape[1:], config
    )

    field_spec = pl.BlockSpec(flux_mod_slices.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((3, bx_blk, by_blk, bz_blk),
                            lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(f_ref, out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
        jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
        kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        # interp_center_to_face along each axis on one stacked channel.
        def c2f_x(ch):
            return (
                -f_ref[ch, (ii - 1) % nx, jj, kk]
              + 9.0 * f_ref[ch, ii, jj, kk]
              + 9.0 * f_ref[ch, (ii + 1) % nx, jj, kk]
              - f_ref[ch, (ii + 2) % nx, jj, kk]
            ) / 16.0

        def c2f_y(ch):
            return (
                -f_ref[ch, ii, (jj - 1) % ny, kk]
              + 9.0 * f_ref[ch, ii, jj, kk]
              + 9.0 * f_ref[ch, ii, (jj + 1) % ny, kk]
              - f_ref[ch, ii, (jj + 2) % ny, kk]
            ) / 16.0

        def c2f_z(ch):
            return (
                -f_ref[ch, ii, jj, (kk - 1) % nz]
              + 9.0 * f_ref[ch, ii, jj, kk]
              + 9.0 * f_ref[ch, ii, jj, (kk + 1) % nz]
              - f_ref[ch, ii, jj, (kk + 2) % nz]
            ) / 16.0

        # Channels: 0=By_fx, 1=Bz_fx, 2=Bx_fy, 3=Bz_fy, 4=Bx_fz, 5=By_fz.
        out_ref[0, ...] = c2f_x(2) - c2f_y(0)   # Omega_z
        out_ref[1, ...] = c2f_y(5) - c2f_z(3)   # Omega_x
        out_ref[2, ...] = c2f_z(1) - c2f_x(4)   # Omega_y

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shape = jax.ShapeDtypeStruct(
        (3,) + tuple(flux_mod_slices.shape[1:]), flux_mod_slices.dtype
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[field_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_edge_emf",
        **kwargs,
    )(flux_mod_slices)


def _ct_curl_pallas(omega_slices, dtdx, dtdy, dtdz, config: SimulationConfig):
    """Stage 3 public wrapper: smoothed 6th-order curl of the stacked
    edge EMFs.  Halo 4: FD6 reach 3 plus 1 for the fused PVA smoothing."""
    ndim = int(config.dimensionality)
    _, block, _ = _ct_block_and_grid(omega_slices.shape[1:], config)

    def _local(omega_local, dtdx_a, dtdy_a, dtdz_a):
        return _ct_curl_pallas_local(
            omega_local, dtdx_a, dtdy_a, dtdz_a, config
        )

    return _pallas_call_sharded(
        _local,
        state_inputs=(omega_slices,),
        other_args=(
            jnp.asarray(dtdx, dtype=omega_slices.dtype),
            jnp.asarray(dtdy, dtype=omega_slices.dtype),
            jnp.asarray(dtdz, dtype=omega_slices.dtype),
        ),
        halo=(4, 4, 4)[:ndim],
        block_shape=block[:ndim],
    )


def _ct_curl_pallas_local(omega_slices, dtdx, dtdy, dtdz, config: SimulationConfig):
    """Stage 3 single-shard build: edge-average smoothing
    (``point_values_to_averages``) + 6th-order curl
    (``finite_difference_int6``).  Outputs the stacked interface-B RHS
    channels (rhs_bx, rhs_by, rhs_bz).

    Input channels: 0=Omega_z, 1=Omega_x, 2=Omega_y.  Fuses two short
    stencils per output: PVA (halo 1 on its two axes) and FD6 (halo 3 on
    its axis).  Worst-case combined halo along any axis is therefore ≤ 4,
    well inside Triton's comfort zone.
    """
    (nx, ny, nz), (bx_blk, by_blk, bz_blk), grid = _ct_block_and_grid(
        omega_slices.shape[1:], config
    )

    field_spec = pl.BlockSpec(omega_slices.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((3, bx_blk, by_blk, bz_blk),
                            lambda bi, bj, bk: (0, bi, bj, bk))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    c1 = 75.0 / 64.0
    c2 = -25.0 / 384.0
    c3 = 3.0 / 640.0

    def kernel(om_ref, dtdx_ref, dtdy_ref, dtdz_ref, out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
        jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
        kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        dtdx_v = dtdx_ref[()]
        dtdy_v = dtdy_ref[()]
        dtdz_v = dtdz_ref[()]

        # ---- PVA helpers (Omega_bar at one specific offset along EACH
        # of its two smoothing axes; the curl below loops over offsets
        # of these PVA results along its differentiation axis).
        # 3D PVA is on two axes; the third axis stays at the cell.
        # Channels: 0 = Omega_z (smoothed in X, Y), 1 = Omega_x (Y, Z),
        # 2 = Omega_y (X, Z).
        def pva_xy_omz(ox, oy, oz):
            q_c = om_ref[0, (ii + ox) % nx, (jj + oy) % ny, (kk + oz) % nz]
            sx = (
                om_ref[0, (ii + ox + 1) % nx, (jj + oy) % ny, (kk + oz) % nz]
              - 2.0 * q_c
              + om_ref[0, (ii + ox - 1) % nx, (jj + oy) % ny, (kk + oz) % nz]
            ) / 24.0
            sy = (
                om_ref[0, (ii + ox) % nx, (jj + oy + 1) % ny, (kk + oz) % nz]
              - 2.0 * q_c
              + om_ref[0, (ii + ox) % nx, (jj + oy - 1) % ny, (kk + oz) % nz]
            ) / 24.0
            return q_c + sx + sy

        def pva_yz_omx(ox, oy, oz):
            q_c = om_ref[1, (ii + ox) % nx, (jj + oy) % ny, (kk + oz) % nz]
            sy = (
                om_ref[1, (ii + ox) % nx, (jj + oy + 1) % ny, (kk + oz) % nz]
              - 2.0 * q_c
              + om_ref[1, (ii + ox) % nx, (jj + oy - 1) % ny, (kk + oz) % nz]
            ) / 24.0
            sz = (
                om_ref[1, (ii + ox) % nx, (jj + oy) % ny, (kk + oz + 1) % nz]
              - 2.0 * q_c
              + om_ref[1, (ii + ox) % nx, (jj + oy) % ny, (kk + oz - 1) % nz]
            ) / 24.0
            return q_c + sy + sz

        def pva_xz_omy(ox, oy, oz):
            q_c = om_ref[2, (ii + ox) % nx, (jj + oy) % ny, (kk + oz) % nz]
            sx = (
                om_ref[2, (ii + ox + 1) % nx, (jj + oy) % ny, (kk + oz) % nz]
              - 2.0 * q_c
              + om_ref[2, (ii + ox - 1) % nx, (jj + oy) % ny, (kk + oz) % nz]
            ) / 24.0
            sz = (
                om_ref[2, (ii + ox) % nx, (jj + oy) % ny, (kk + oz + 1) % nz]
              - 2.0 * q_c
              + om_ref[2, (ii + ox) % nx, (jj + oy) % ny, (kk + oz - 1) % nz]
            ) / 24.0
            return q_c + sx + sz

        # ---- 6th-order interface FD: c1·(f[i]−f[i−1]) + c2·(f[i+1]−f[i−2])
        # + c3·(f[i+2]−f[i−3]).
        def fd6_x(pva):
            return (
                c1 * (pva(0, 0, 0) - pva(-1, 0, 0))
              + c2 * (pva(1, 0, 0) - pva(-2, 0, 0))
              + c3 * (pva(2, 0, 0) - pva(-3, 0, 0))
            )

        def fd6_y(pva):
            return (
                c1 * (pva(0, 0, 0) - pva(0, -1, 0))
              + c2 * (pva(0, 1, 0) - pva(0, -2, 0))
              + c3 * (pva(0, 2, 0) - pva(0, -3, 0))
            )

        def fd6_z(pva):
            return (
                c1 * (pva(0, 0, 0) - pva(0, 0, -1))
              + c2 * (pva(0, 0, 1) - pva(0, 0, -2))
              + c3 * (pva(0, 0, 2) - pva(0, 0, -3))
            )

        out_ref[0, ...] = -dtdy_v * fd6_y(pva_xy_omz) + dtdz_v * fd6_z(pva_xz_omy)
        out_ref[1, ...] = -dtdz_v * fd6_z(pva_yz_omx) + dtdx_v * fd6_x(pva_xy_omz)
        out_ref[2, ...] = -dtdx_v * fd6_x(pva_xz_omy) + dtdy_v * fd6_y(pva_yz_omx)

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shape = jax.ShapeDtypeStruct(
        (3,) + tuple(omega_slices.shape[1:]), omega_slices.dtype
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[field_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_curl",
        **kwargs,
    )(omega_slices, dtdx, dtdy, dtdz)


def _ct_rhs_pallas(
    conserved_state,
    By_flux_x_interface,
    Bz_flux_x_interface,
    Bx_flux_y_interface,
    Bz_flux_y_interface,
    Bx_flux_z_interface,
    By_flux_z_interface,
    dtdx,
    dtdy,
    dtdz,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Three-stage Pallas CT-RHS: chain ``_ct_modified_flux_pallas`` →
    ``_ct_edge_emf_pallas`` → ``_ct_curl_pallas``.

    Each stage is a single bounded-halo Pallas kernel — compile time
    stays sub-second per kernel — and each stage costs exactly one halo
    exchange under a multi-device mesh (the wrappers route through
    ``_pallas_call_sharded``).  The flux slices travel stacked between
    stages, so the peak temporary footprint stays well below the native
    code's 12+ intermediates.
    """
    assert _ct_rhs_pallas_supported(conserved_state, config)
    flux_slices = jnp.stack([
        By_flux_x_interface, Bz_flux_x_interface,
        Bx_flux_y_interface, Bz_flux_y_interface,
        Bx_flux_z_interface, By_flux_z_interface,
    ])
    flux_mod = _ct_modified_flux_pallas(
        conserved_state, flux_slices, config, registered_variables
    )
    del flux_slices
    omega = _ct_edge_emf_pallas(flux_mod, config)
    del flux_mod
    rhs_b = _ct_curl_pallas(omega, dtdx, dtdy, dtdz, config)
    return rhs_b[0], rhs_b[1], rhs_b[2]
