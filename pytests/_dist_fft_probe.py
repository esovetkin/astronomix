"""Probe: can the GSPMD partitioner run the periodic Poisson solve without
replicating the grid, if the 3D FFT is expressed per-axis with resharding
hints?

Compares, on an X-sharded field:

  A. ``jnp.fft.fftn`` (the current Poisson-solver formulation) -- the SPMD
     partitioner replicates the operand of an FFT along its transform
     dimensions, so per-device memory is the *global* grid.
  B. per-axis FFTs with ``with_sharding_constraint`` hints: FFT over (y, z)
     batched over the sharded X axis, reshard X <-> Y (an all-to-all), FFT
     over x batched over the sharded Y axis, Green-function multiply, and
     the mirrored inverse.  Per-device memory should stay ~ the local shard.

Reports max|A - B|, wall time, and per-device compiled memory for both.

Launch::

    srun --ntasks=4 --ntasks-per-node=4 --gpu-bind=none \
        python pytests/_dist_fft_probe.py --n 512
"""

# general
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=512)
parser.add_argument("--reps", type=int, default=5)
args = parser.parse_args()

# third-party (raw jax before distributed init)
import jax

_cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
_localid = int(os.environ.get("SLURM_LOCALID", "0"))
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize(
        local_device_ids=[_localid] if len(_cvd) > 1 else [0]
    )

jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P  # noqa: E402

G = jax.device_count()
rank = jax.process_index()
n = args.n

mesh = jax.make_mesh((G,), ("x",), axis_types=(AxisType.Auto,))
shard_x = NamedSharding(mesh, P("x", None, None))
shard_y = NamedSharding(mesh, P(None, "x", None))


def _greens(rho_k):
    """Multiply by the periodic Green's function -1/k^2 (Jeans-swindled)."""
    freqs = jnp.fft.fftfreq(n) * (2.0 * jnp.pi * n)
    kx = freqs.reshape(n, 1, 1)
    ky = freqs.reshape(1, n, 1)
    kz = freqs.reshape(1, 1, n)
    k_squared = kx ** 2 + ky ** 2 + kz ** 2
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    potential_k = -rho_k / k_squared_safe
    return jnp.where(k_squared == 0.0, 0.0, potential_k)


@jax.jit
def poisson_fftn(rho):
    rho_k = jnp.fft.fftn(rho)
    return jnp.real(jnp.fft.ifftn(_greens(rho_k)))


@jax.jit
def poisson_per_axis(rho):
    # Forward: (y,z) transform batched over sharded x, reshard, x transform
    # batched over sharded y.
    rho_k = jnp.fft.fftn(rho, axes=(1, 2))
    rho_k = jax.lax.with_sharding_constraint(rho_k, shard_y)
    rho_k = jnp.fft.fft(rho_k, axis=0)
    rho_k = jax.lax.with_sharding_constraint(rho_k, shard_y)

    potential_k = _greens(rho_k)

    # Inverse, mirrored.
    potential_k = jnp.fft.ifft(potential_k, axis=0)
    potential_k = jax.lax.with_sharding_constraint(potential_k, shard_x)
    potential = jnp.fft.ifftn(potential_k, axes=(1, 2))
    potential = jax.lax.with_sharding_constraint(jnp.real(potential), shard_x)
    return potential


def _bench(fn, rho, label):
    lowered = fn.lower(rho)
    compiled = lowered.compile()
    mem = compiled.memory_analysis()
    out = compiled(rho)
    out.block_until_ready()
    start = time.time()
    for _ in range(args.reps):
        out = compiled(rho)
    out.block_until_ready()
    elapsed = (time.time() - start) / args.reps
    if rank == 0:
        print(
            f"[{label}] {elapsed * 1e3:8.1f} ms/solve  "
            f"temp/dev = {mem.temp_size_in_bytes / 2**30:6.2f} GiB  "
            f"args/dev = {mem.argument_size_in_bytes / 2**30:6.2f} GiB",
            flush=True,
        )
    return out


def main():
    key = jax.random.PRNGKey(0)

    @jax.jit
    def _make_rho():
        rho = jax.random.normal(key, (n, n, n), dtype=jnp.float32)
        return jax.lax.with_sharding_constraint(rho, shard_x)

    rho = jax.jit(_make_rho, out_shardings=shard_x)()

    out_b = _bench(poisson_per_axis, rho, "per-axis   ")
    out_a = _bench(poisson_fftn, rho, "fftn (ref) ")

    from jax.experimental import multihost_utils
    a = multihost_utils.process_allgather(out_a, tiled=True)
    b = multihost_utils.process_allgather(out_b, tiled=True)
    if rank == 0:
        diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        scale = float(np.max(np.abs(np.asarray(a))))
        print(f"max|A-B| = {diff:.3e} (max|A| = {scale:.3e})", flush=True)
        print("DIST FFT PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
