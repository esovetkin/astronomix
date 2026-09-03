"""Shared Pallas-backend utilities used across the FD and FV paths.

This module is the single place that:
- imports Pallas / Triton (and exposes ``pl is None`` if unavailable),
- normalises ``config.backend_config.pallas_block_shape`` to a 3-tuple,
- builds Triton ``CompilerParams`` from config knobs,
- exposes the ``backend == PALLAS`` predicate,
- provides the ``_pallas_call_sharded`` multi-GPU wrapper that turns an
  opaque ``pl.pallas_call`` into a ``shard_map`` + ppermute halo-exchange
  body when the user runs on a multi-device mesh.

Every Pallas kernel module under ``astronomix`` should import from here so
new knobs / fallbacks only need to be added once.
"""

# general
import contextvars
from contextlib import contextmanager

# jax
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

# astronomix containers
from astronomix.option_classes.simulation_config import PALLAS, SimulationConfig

# Pallas / Triton are optional: a CPU-only or older JAX install may lack one or
# both. We import them defensively so the rest of the module loads (callers gate
# on ``pl is None`` / ``pltriton is None``) rather than failing at import time.
try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - Pallas optional
    pl = None

try:
    from jax.experimental.pallas import triton as pltriton
except Exception:  # pragma: no cover - Triton GPU backend optional
    pltriton = None


def _backend_is_pallas(config: SimulationConfig) -> bool:
    """Return whether the configured backend is the Pallas/Triton GPU backend."""
    return config.backend_config.backend == PALLAS


def _default_pallas_block_shape(ndim: int) -> tuple[int, int, int]:
    """Return the default Pallas block shape ``(bx, by, bz)`` for ``ndim`` spatial
    dimensions (inactive dimensions forced to 1).

    The 3D default (2, 2, 32) keeps one element per thread at the default
    ``num_warps=4`` (128-cell blocks) with a 256-byte-contiguous fast axis:
    on an A100 this ran the dp MHD WENO kernel ~10% faster end-to-end than
    the previous (4, 4, 8) at identical results (2026-07 Alfvén-wave sweep;
    128-cell blocks with longer z-rows all tie within noise, larger or
    smaller blocks register-spill or idle threads)."""
    if ndim == 1:
        return (128, 1, 1)
    if ndim == 2:
        return (16, 16, 1)
    return (2, 2, 32)


def _as_3tuple_block_shape(block_shape, ndim: int, spatial_shape=None) -> tuple[int, int, int]:
    """Normalise whatever the user supplied (None / str / tuple) to
    ``(bx, by, bz)`` with the inactive dims forced to 1.  Pallas grid
    construction depends on this tuple being canonical.

    When ``spatial_shape`` is given, each active block dimension is clamped
    to its grid extent, so a default tuned for production grids (e.g. bz=32)
    stays a valid tiling on small grids (bz -> nz) instead of tripping the
    grid-divisibility support predicates and silently dropping the whole run
    to the native backend."""
    if block_shape is None:
        parts = _default_pallas_block_shape(ndim)
    elif isinstance(block_shape, str):
        parts = tuple(int(p.strip()) for p in block_shape.split(",") if p.strip())
    else:
        parts = tuple(int(x) for x in block_shape)
    if len(parts) == 1:
        parts = (parts[0], 1, 1)
    elif len(parts) == 2:
        parts = (parts[0], parts[1], 1)
    elif len(parts) >= 3:
        parts = parts[:3]
    else:
        parts = _default_pallas_block_shape(ndim)
    if ndim == 1:
        parts = (parts[0], 1, 1)
    elif ndim == 2:
        parts = (parts[0], parts[1], 1)
    if spatial_shape is not None:
        parts = tuple(
            min(int(b), int(n)) for b, n in zip(parts, tuple(spatial_shape) + (1, 1))
        )[:3]
        parts = tuple(parts) + (1,) * (3 - len(parts))
    return parts


def _pallas_compiler_params(config: SimulationConfig):
    """Return Triton ``CompilerParams`` (or None if the Triton backend is
    not available / the user opted out via ``pallas_use_triton=False``)."""
    use_triton = config.backend_config.pallas_use_triton
    if use_triton and pltriton is not None:
        return pltriton.CompilerParams(
            num_warps=config.backend_config.pallas_num_warps,
        )
    return None


# -----------------------------------------------------------------------------
# Multi-GPU shard_map + halo wrapper.
#
# Every Pallas kernel in this codebase passes its state-shape input(s) to
# ``pl.pallas_call`` via ``BlockSpec(state.shape, lambda ...: (0, 0, 0, 0))``.
# That tells Pallas/Triton "each block program can read anywhere in the
# array", which is the correct (and fast) shape on a single device — but
# it makes the call entirely opaque to GSPMD.  When the input is sharded
# across a device mesh, XLA's only legal lowering is to ``all-gather`` the
# whole state on every device before each ``pallas_call``, which dominates
# every kernel hot-loop and kills strong scaling (~0.95× on the FD Pallas
# sound-wave benchmark before this fix).
#
# The fix is mechanical: wrap each ``pl.pallas_call`` in a ``shard_map``
# body that
#   1. ppermutes a halo of ``stencil_reach`` cells from each neighbour
#      shard along every sharded spatial axis (periodic ring),
#   2. concatenates [left_halo, local, right_halo] on each sharded axis,
#   3. calls the existing kernel on the local-padded shard (its modular
#      indexing wraps within the padded shape; halo cells provide the
#      correct neighbour values for interior reads),
#   4. strips the halo from the output.
#
# So the user-facing knob is just: ``pallas_mesh_context(mesh)`` around
# the JIT trace, plus each kernel calling ``_pallas_call_sharded`` instead
# of ``pl.pallas_call(...)(args)`` directly.  No kernel arithmetic changes.
# -----------------------------------------------------------------------------


_pallas_mesh_ctx: contextvars.ContextVar = contextvars.ContextVar(
    "astronomix_pallas_mesh", default=None
)


@contextmanager
def pallas_mesh_context(mesh):
    """Set the active mesh for Pallas kernel sharding.

    ``time_integration`` enters this context around the JIT trace whenever
    the user supplies a ``sharding`` argument.  Inside the context every
    Pallas kernel that calls ``_pallas_call_sharded`` will route through
    a ``shard_map`` + ppermute halo exchange instead of the bare
    ``pl.pallas_call``.

    When ``mesh`` is ``None`` (single-device run) or has size 1, the
    helper is a no-op — the kernel runs exactly as before.
    """
    token = _pallas_mesh_ctx.set(mesh)
    try:
        yield
    finally:
        _pallas_mesh_ctx.reset(token)


def _current_pallas_mesh():
    return _pallas_mesh_ctx.get()


def _round_halo_up_to_block(halo, block_shape) -> tuple[int, ...]:
    """Round each natural halo width up to a multiple of the corresponding
    Pallas block size.  The kernel's internal ``grid = (nx // bx, ...)``
    must remain block-divisible after halo padding, so we always grow the
    halo to the nearest block multiple."""
    out = []
    for h, b in zip(halo, block_shape, strict=False):
        h_i = int(h)
        b_i = max(int(b), 1)
        if h_i <= 0:
            out.append(0)
        else:
            q, r = divmod(h_i, b_i)
            out.append(b_i * (q + (1 if r else 0)))
    return tuple(out)


def _spatial_sharded_axes(mesh, pspec, ndim):
    """Return a list of ``(array_axis_idx, mesh_axis_name, num_dev)`` for
    every spatial array axis that is split across more than one device.
    The variable axis (index 0) is always skipped."""
    out = []
    for ax in range(1, ndim + 1):
        if ax >= len(pspec):
            break
        name = pspec[ax]
        if name is None:
            continue
        if isinstance(name, tuple):
            for nm in name:
                n = mesh.shape[nm]
                if n > 1:
                    out.append((ax, nm, n))
        else:
            n = mesh.shape[name]
            if n > 1:
                out.append((ax, name, n))
    return out


def _default_state_pspec(mesh, ndim) -> PartitionSpec:
    """Best-effort PartitionSpec when an input array's ``.sharding`` is an
    ``UnspecifiedValue`` (which can happen for intermediates inside a
    JIT trace).  Assumes the standard ``(VARAXIS, XAXIS, YAXIS, ZAXIS)``
    mesh emitted by ``pytests/_benchmark_utils.py::_build_sharding`` and
    by callers that mirror it."""
    axis_names = tuple(mesh.axis_names)
    return PartitionSpec(*axis_names[: 1 + ndim])


def _normalize_input_halos(input_halos, n_inputs: int, ndim: int):
    """validate and normalize one halo tuple per input."""
    if input_halos is None:
        return None
    if len(input_halos) != n_inputs:
        raise ValueError("input_halos must match state_inputs")

    out = []
    # pad short tuples with zeros, truncate long tuples
    # ndim == 3
    #     (2,)      -> (2, 0, 0)
    #     (2, 1)    -> (2, 1, 0)
    #     (2, 1, 4) -> (2, 1, 4)
    # ndim == 2
    #     (2, 1, 4) -> (2, 1)
    for halo in input_halos:
        halo_n = tuple(int(x) for x in halo) + (0,) * max(0, ndim - len(halo))
        halo_n = halo_n[:ndim]
        if any(h < 0 for h in halo_n):
            raise ValueError("input_halos must be non-negative")
        out.append(halo_n)
    return tuple(out)


def _normalize_output_halos(output_halo, n_outputs: int, ndim: int):
    zero = (0,) * ndim
    if output_halo is None:
        return tuple(zero for _ in range(n_outputs))

    if (
        len(output_halo) == ndim
        and all(isinstance(x, int) for x in output_halo)
    ):
        halos = tuple(output_halo for _ in range(n_outputs))
    else:
        if len(output_halo) != n_outputs:
            raise ValueError("output_halo must match state outputs")
        halos = tuple(output_halo)

    out = []
    # ndim == 3
    #     (1,)      -> (1, 0, 0)
    #     (1, 0)    -> (1, 0, 0)
    #     (1, 0, 0) -> (1, 0, 0)
    #
    # ndim == 2
    #     (1,)      -> (1, 0)
    #     (1, 0)    -> (1, 0)
    #     (1, 0, 4) -> (1, 0)
    for halo in halos:
        halo_n = tuple(int(x) for x in halo) + (0,) * max(0, ndim - len(halo))
        halo_n = halo_n[:ndim]
        if any(h < 0 for h in halo_n):
            raise ValueError("output_halo must be non-negative")
        out.append(halo_n)
    return tuple(out)


def _local_edge_pad(arr, width: int, axis: int, *, left: bool):
    """creates local fake padding from edge values

    this is used when a local pallas shape needs more halo padding than the input actually communicates
    """
    if width <= 0:
        return None
    size = arr.shape[axis]
    if left:
        edge = jax.lax.slice_in_dim(arr, 0, 1, axis=axis)
    else:
        edge = jax.lax.slice_in_dim(arr, size - 1, size, axis=axis)
    reps = [1] * arr.ndim
    reps[axis] = int(width)
    return jnp.tile(edge, reps)


def _pair_groups(num_dev: int, parity: int):
    if parity == 0:
        return [[i, i + 1] for i in range(0, num_dev, 2)]
    return [[i, (i + 1) % num_dev] for i in range(1, num_dev, 2)]


def _pair_all_to_all_halos(arr, axis: int, width: int, mesh_axis_name, num_dev: int):
    if num_dev % 2:
        raise ValueError("pairwise all_to_all halo exchange requires an even mesh axis")

    rank = jax.lax.axis_index(mesh_axis_name)
    size = arr.shape[axis]
    left_edge = jax.lax.slice_in_dim(arr, 0, width, axis=axis)
    right_edge = jax.lax.slice_in_dim(arr, size - width, size, axis=axis)
    zero = jnp.zeros_like(left_edge)

    def phase(parity: int):
        first = (rank % 2) == parity
        payload = jnp.stack(
            (
                jnp.where(first, zero, left_edge),
                jnp.where(first, right_edge, zero),
            ),
            axis=0,
        )
        exchanged = jax.lax.all_to_all(
            payload,
            mesh_axis_name,
            split_axis=0,
            concat_axis=0,
            axis_index_groups=_pair_groups(num_dev, parity),
            tiled=False,
        )
        left = jnp.where(first, zero, exchanged[0])
        right = jnp.where(first, exchanged[1], zero)
        return left, right

    left0, right0 = phase(0)
    left1, right1 = phase(1)
    return left0 + left1, right0 + right1


def _pad_axis_with_input_halo(arr, axis, shape_h, comm_h, mesh_axis_name, num_dev):
    """combines real communicated halo cells with fake local padding

    shape_h = 4; comm_h = 4;
        old behaviour, for real halo cells are exchanged on each side

    shape_h = 4; comm_h = 0;
        it exchanges nothing and fills the four left/right padded
        cells by repeating the local edge values

    shape_h = 4; comm_h = 1;
        it exchanges one real neighbor cell and adds three local fake
        padding cells on each side.

    """
    if comm_h > shape_h:
        raise ValueError("input halo cannot exceed padded shape halo")

    pieces = []
    fake_left = _local_edge_pad(arr, shape_h - comm_h, axis, left=True)
    if fake_left is not None:
        pieces.append(fake_left)

    if comm_h > 0:
        left_halo, right_halo = _pair_all_to_all_halos(
            arr, axis, comm_h, mesh_axis_name, num_dev
        )
        # _pair_all_to_all_halos scales better than ppermute
        # size = arr.shape[axis]
        # left_edge = jax.lax.slice_in_dim(arr, 0, comm_h, axis=axis)
        # right_edge = jax.lax.slice_in_dim(arr, size - comm_h, size, axis=axis)
        # left_perm = [(j, (j - 1) % num_dev) for j in range(num_dev)]
        # right_perm = [(j, (j + 1) % num_dev) for j in range(num_dev)]
        # left_halo = jax.lax.ppermute(right_edge, mesh_axis_name, perm=right_perm)
        # right_halo = jax.lax.ppermute(left_edge, mesh_axis_name, perm=left_perm)
        pieces.extend((left_halo, arr, right_halo))
    else:
        pieces.append(arr)

    fake_right = _local_edge_pad(arr, shape_h - comm_h, axis, left=False)
    if fake_right is not None:
        pieces.append(fake_right)

    return jnp.concatenate(pieces, axis=axis)


def _pallas_call_sharded(
        kernel_build_fn,
        state_inputs,
        other_args=(),
        *,
        halo,
        block_shape,
        num_state_outputs: int = 1,
        input_halos=None,
        output_halo=None,
):
    """Optionally wrap a Pallas-kernel build-and-call in ``shard_map``.

    Args:
        kernel_build_fn:
            Callable ``(state_inputs_local_padded..., other_args...) -> out``
            whose body builds and calls ``pl.pallas_call``.  When the call
            runs inside a ``shard_map`` body, each invocation sees the
            *local* (halo-padded) shape and the kernel's internal
            ``grid``/``BlockSpec`` are built for that shape automatically.
        state_inputs:
            Tuple of state-shape arrays that all share the same sharding
            (same ``PartitionSpec``).  Each one is padded with halo cells
            from neighbour shards along every sharded spatial axis.
        other_args:
            Tuple of replicated arrays (scalar dt, scalar gamma, ...)
            passed through as ``PartitionSpec()``.
        halo:
            Per-spatial-axis natural stencil reach ``(hx, hy, hz)``.
            Pointwise kernels pass ``(0, 0, 0)`` — they still get the
            ``shard_map`` (so the kernel runs locally on each shard),
            just with no ppermute.
        block_shape:
            Per-spatial-axis Pallas block size ``(bx, by, bz)``.  The
            halo is rounded up to the nearest block multiple so the
            padded shard remains block-divisible.
        num_state_outputs:
            Number of state-shape outputs of ``kernel_build_fn`` (1 for
            most kernels; >1 for the CT staged kernels which return
            tuples of single-channel arrays).
        input_halos:
            Optional per-state-input natural halo widths. ``None`` preserves
            the original behavior. When provided, every input is padded to the
            same block-rounded shape halo, but only its requested natural halo
            is exchanged with neighbours.
        output_halo:
            Optional natural halo width to keep on state-shaped outputs. The
            default strips all padded cells. Keeping a small output halo lets a
            following local stencil consume halo data that this kernel already
            computed, avoiding another exchange.

    Returns:
        Either ``kernel_build_fn(*state_inputs, *other_args)`` directly
        (single-device path) or the equivalent ``shard_map``-wrapped
        result with the halo stripped from each state-shape output.
    """
    mesh = _current_pallas_mesh()
    if mesh is None or mesh.size <= 1:
        return kernel_build_fn(*state_inputs, *other_args)

    state0 = state_inputs[0]
    ndim = state0.ndim - 1

    sharding = getattr(state0, "sharding", None)
    if isinstance(sharding, NamedSharding):
        pspec = sharding.spec
    else:
        pspec = _default_state_pspec(mesh, ndim)

    sharded_axes = _spatial_sharded_axes(mesh, pspec, ndim)
    if not sharded_axes:
        return kernel_build_fn(*state_inputs, *other_args)

    block_3 = tuple(block_shape) + (1,) * max(0, 3 - len(block_shape))
    halo_3 = tuple(halo) + (0,) * max(0, 3 - len(halo))
    requested_input_halos = _normalize_input_halos(
        input_halos, len(state_inputs), ndim
    )

    # shape_halo: block-rounded halo used to shape the local arrays
    # passed into the Pallas kernel
    #
    # comm_halos: are the real neighbor halo widths to exchange for
    # each input
    kept_output_halos = _normalize_output_halos(
        output_halo, int(num_state_outputs), ndim
    )
    if requested_input_halos is None:
        shape_halo = _round_halo_up_to_block(halo_3[:ndim], block_3[:ndim])
        comm_halos = tuple(shape_halo for _ in state_inputs)
    else:
        max_halo = tuple(
            max([int(halo_3[ax])] + [h[ax] for h in requested_input_halos])
            for ax in range(ndim)
        )
        shape_halo = _round_halo_up_to_block(max_halo, block_3[:ndim])
        comm_halos = requested_input_halos

    for kept_halo in kept_output_halos:
        for kept, shaped in zip(kept_halo, shape_halo, strict=False):
            if kept > shaped:
                raise ValueError("output_halo cannot exceed padded shape halo")

    try:
        from jax.shard_map import shard_map  # jax >= 0.8 (promoted out of experimental)
    except ImportError:  # jax < 0.8
        from jax.experimental.shard_map import shard_map

    def body(*all_args):
        state_arrays = list(all_args[: len(state_inputs)])
        others = all_args[len(state_inputs):]

        for array_axis_idx, mesh_axis_name, num_dev in sharded_axes:
            spatial_idx = array_axis_idx - 1
            if spatial_idx >= len(shape_halo):
                continue
            h = shape_halo[spatial_idx]
            if h <= 0:
                continue
            # without fake padding equivalent:
            #     for i, arr in enumerate(state_arrays):
            #         left_edge = slice local left edge
            #         right_edge = slice local right edge
            #         left_halo = ppermute(right_edge)
            #         right_halo = ppermute(left_edge)
            #         state_arrays[i] = concatenate([left_halo, arr, right_halo])
            for i, arr in enumerate(state_arrays):
                comm_h = comm_halos[i][spatial_idx]
                # Fake cells fill block-rounded shape padding only. Callers may
                # request comm_h < h when kept outputs never read those cells.
                state_arrays[i] = _pad_axis_with_input_halo(
                    arr, array_axis_idx, h, comm_h, mesh_axis_name, num_dev
                )

        # Re-enter the wrapper with mesh=None so the recursive
        # ``kernel_build_fn`` call goes through the no-wrap path.  Without
        # this, a kernel that calls ``_pallas_call_sharded`` from its body
        # would wrap itself forever.
        with pallas_mesh_context(None):
            out = kernel_build_fn(*state_arrays, *others)

        def _strip(o, kept_halo):
            for array_axis_idx, _, _ in sharded_axes:
                spatial_idx = array_axis_idx - 1
                if spatial_idx >= len(shape_halo):
                    continue
                h = shape_halo[spatial_idx]
                keep = kept_halo[spatial_idx]
                strip = h - keep
                if strip <= 0:
                    continue
                size = o.shape[array_axis_idx]
                o = jax.lax.slice_in_dim(
                    o, strip, size - strip, axis=array_axis_idx
                )
            return o

        if isinstance(out, tuple):
            return tuple(_strip(o, h) for o, h in zip(out, kept_output_halos, strict=True))
        return _strip(out, kept_output_halos[0])

    state_specs = tuple(pspec for _ in state_inputs)
    other_specs = tuple(PartitionSpec() for _ in other_args)
    if num_state_outputs > 1:
        out_specs = tuple(pspec for _ in range(num_state_outputs))
    else:
        out_specs = pspec

    wrapped = shard_map(
        body,
        mesh=mesh,
        in_specs=state_specs + other_specs,
        out_specs=out_specs,
        check_rep=False,
    )
    return wrapped(*state_inputs, *other_args)


# -----------------------------------------------------------------------------
# Differentiability: pair every Pallas entry with a native-JAX backward.
# -----------------------------------------------------------------------------
#
# Pallas kernels in this codebase use ``input_output_aliases`` for memory
# efficiency. JAX cannot transpose an aliased ``pl.pallas_call`` (``JVP with
# aliasing not supported``), so any path that hits a Pallas kernel is
# non-differentiable by default. We bridge that gap with a ``jax.custom_jvp``
# whose primal still calls the (aliased, fast) Pallas branch and whose
# tangent rule delegates to the equivalent native-JAX branch — which is
# already JVP-differentiable. Reverse-mode (``jax.grad``) is then derived by
# JAX via transposition.
#
# Forward simulation perf is unaffected: outside of AD the custom_jvp
# rule isn't invoked and the call collapses to the bare Pallas branch.
#
# Both branches must produce the same pytree-structured output. The Pallas
# guide promises bit-identical primal outputs for the existing kernels, so
# the gradient computed by transposing the native JVP at the Pallas-evaluated
# inputs is the correct gradient of the Pallas operation.
#
# Hand-rolled Pallas adjoint kernels can later replace the native tangent
# branch on a per-kernel basis without changing call sites.

def diffable_pallas_call(state, params, *, pallas_branch, native_branch):
    """Run ``pallas_branch(state, params)`` with a custom_jvp boundary that
    routes tangent computation through ``native_branch``.

    Both branches must accept the same positional ``(state, params)`` pair
    and produce the same pytree structure. Anything static (config,
    registered_variables, axis index, ...) should be closed over.

    Outside of AD the call collapses to ``pallas_branch(state, params)``
    directly — no overhead. Under ``jax.jvp`` / ``jax.jacfwd`` /
    ``jax.grad`` / ``jax.vjp`` / ``jax.jacrev`` the custom rule fires and
    the tangent goes through ``native_branch``.
    """
    @jax.custom_jvp
    def _f(s, p):
        return pallas_branch(s, p)

    @_f.defjvp
    def _f_jvp(primals, tangents):
        primal_out = pallas_branch(*primals)
        _, tangent_out = jax.jvp(native_branch, primals, tangents)
        return primal_out, tangent_out

    return _f(state, params)


def diffable_pallas_call_n(primals, *, pallas_branch, native_branch):
    """Same as :func:`diffable_pallas_call` but takes a tuple of arbitrary
    differentiable primals (so callers with more than two diff args, e.g.
    extra rhs/accumulator buffers, can still get a custom_jvp boundary)."""
    @jax.custom_jvp
    def _f(*args):
        return pallas_branch(*args)

    @_f.defjvp
    def _f_jvp(args, tangents):
        primal_out = pallas_branch(*args)
        _, tangent_out = jax.jvp(native_branch, args, tangents)
        return primal_out, tangent_out

    return _f(*primals)


def pallas_vjp_call(state, aux, *, pallas_forward, pallas_backward):
    """Run ``pallas_forward(state, aux)`` with a ``jax.custom_vjp`` boundary
    whose reverse rule is a *native Pallas adjoint kernel* ``pallas_backward``.

    Unlike :func:`diffable_pallas_call` (which routes the tangent — and hence
    the transposed gradient — through native JAX), this keeps the entire
    backward pass on the Pallas/GPU backend: ``pallas_backward(state, aux, cot)``
    returns the input cotangent ``d(loss)/d(state)`` directly from a
    hand-built adjoint kernel.

    Differentiates w.r.t. ``state`` only.  ``aux`` (e.g. the traced
    ``SimulationParams``) is threaded *through* the boundary and given a zero
    cotangent — it must be passed explicitly rather than closed over because
    ``jax.custom_vjp`` cannot capture traced values in its forward/backward
    closures (only static/concrete data — config, axis — may be closed over by
    the two branches).  Treating the physical constants as non-differentiable
    matches the inverse-problem regime (gradients w.r.t. the state, not params).

    NOTE: ``jax.custom_vjp`` supports reverse-mode only — ``jax.jvp`` /
    forward-mode AD on this boundary raises.  Use it for reverse-mode
    (``jax.grad`` / ``differentiation_mode = BACKWARDS``); for forward-mode keep
    :func:`diffable_pallas_call`.
    """
    @jax.custom_vjp
    def _f(s, a):
        return pallas_forward(s, a)

    def _f_fwd(s, a):
        return pallas_forward(s, a), (s, a)

    def _f_bwd(residual, cotangent):
        s, a = residual
        state_bar = pallas_backward(s, a, cotangent)

        def _zero(x):  # correctly-typed zero cotangent (float0 for non-inexact)
            x = jnp.asarray(x)
            if jnp.issubdtype(x.dtype, jnp.inexact):
                return jnp.zeros_like(x)
            return jnp.zeros(x.shape, dtype=jax.dtypes.float0)

        return (state_bar, jax.tree_util.tree_map(_zero, a))

    _f.defvjp(_f_fwd, _f_bwd)
    return _f(state, aux)
