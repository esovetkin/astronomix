"""Minimal multi-process JAX sanity check: rendezvous + one collective.

Run under Slurm with one process per GPU and *no* GPU binding (all node GPUs
visible to every task, each task picks its own by node-local rank)::

    srun --ntasks=<G> --ntasks-per-node=4 --gpu-bind=none \
        python pytests/_dist_sanity.py

Expected output on rank 0: ``allgather = [0. 1. ... G-1.]`` followed by
``PASS``.  This exercises exactly the paths that break first on a new
machine: the coordinator rendezvous, intra-node NCCL P2P, and (from two
nodes on) the inter-node network.
"""

# general
import os

# third-party (raw jax only -- astronomix must NOT be imported before
# jax.distributed.initialize(), because importing it creates the backend)
import jax


def _local_device_ids():
    """This rank's local GPU, robust to either Slurm GPU binding mode.

    With ``--gpu-bind=none`` every task sees all node GPUs and must pick its
    own by node-local rank; with cgroup binding one GPU appears as ordinal 0.
    """
    visible = [
        x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x
    ]
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    return [local_rank] if len(visible) > 1 else [0]


def main():
    jax.distributed.initialize(local_device_ids=_local_device_ids())

    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    rank = jax.process_index()
    world = jax.process_count()
    print(
        f"[rank {rank}/{world}] local={jax.local_devices()} "
        f"global={jax.device_count()} host={os.uname().nodename}",
        flush=True,
    )

    gathered = multihost_utils.process_allgather(jnp.asarray(float(rank)))
    if rank == 0:
        print(f"allgather = {gathered}", flush=True)

    expected = jnp.arange(world, dtype=gathered.dtype)
    assert (gathered == expected).all(), f"allgather mismatch: {gathered}"
    multihost_utils.sync_global_devices("dist_sanity_done")
    if rank == 0:
        print("PASS", flush=True)


if __name__ == "__main__":
    main()
