"""
Multi-node WEAK-scaling driver: ideal MHD, FD solver, Pallas backend.

Weak scaling keeps the per-GPU sub-domain fixed while growing the global grid
with the device count, so ideal scaling is a flat wall-clock curve.  The state
is built already globally-sharded (each process materialises only its local
shard via ``build_cp_alfven_state_sharded``), so the full grid never lives on
one host -- including the staggered vector-potential construction that keeps
the discrete face-centered B divergence-free.

Launch (one process per GPU) under Slurm::

    srun --ntasks=<G> --ntasks-per-node=4 --gpu-bind=none \
        python examples/scripts/scaling/weak_scaling_mhd.py --bx 64 --by 2048 --bz 2048 \
            --steps 10 --block-shape 4,4,8

The mesh shards the X axis only: global grid = (bx * G, by, bz).  The box is
set numerically equal to the grid so the spacing stays uniform (h = 1) at
every rung; this is a timing/throughput benchmark, so wave periodicity is
irrelevant, and the wave amplitude is shrunk to the sound-wave test's 1e-6 so
the (deliberately) unresolved wavelength stays a harmless linear perturbation
on the uniform background.  The MHD state carries 11 fields versus hydro's 5,
so the default per-GPU sub-domain is half the hydro one.
"""

# --- IMPORTANT: bootstrap multi-process mode BEFORE importing astronomix. ---
# Importing astronomix creates the JAX backend (NamedTuple jnp.array defaults),
# and jax.distributed.initialize() must run before the backend exists.
import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--bx", type=int, default=64, help="per-GPU cells along sharded X")
parser.add_argument("--by", type=int, default=2048, help="cells along Y (global)")
parser.add_argument("--bz", type=int, default=2048, help="cells along Z (global)")
parser.add_argument("--steps", type=int, default=10, help="fixed number of timesteps")
parser.add_argument("--dt", type=float, default=0.4,
                    help="fixed timestep (stable for h=1, fast speed ~1.1)")
parser.add_argument("--block-shape", type=str, default="4,4,8")
parser.add_argument("--cfl", type=float, default=1.5)
parser.add_argument("--tag", type=str, default="gh200")
parser.add_argument("--approx-rsqrt", action="store_true",
                    help="use the approximate-rsqrt path in the MHD WENO "
                         "kernel (BackendConfig.use_approximate_rsqrt)")
parser.add_argument("--pallas-ct", action="store_true",
                    help="route the constrained-transport helpers through "
                         "their Pallas kernels (BackendConfig.pallas_ct); "
                         "under sharding this replaces the many per-shift "
                         "collective-permutes of the native roll-based "
                         "stencils with one halo exchange per kernel")
parser.add_argument("--gpus", type=int, default=0,
                    help="SINGLE-process mode: grab this many local GPUs via "
                         "autocvd (no jax.distributed). Ignored under srun "
                         "multi-process (SLURM_NTASKS>1).")
args = parser.parse_args()

_BLOCK = tuple(int(x) for x in args.block_shape.split(","))

_multi = "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1

# Single-process multi-GPU: select GPUs via autocvd BEFORE importing jax.
if not _multi and args.gpus > 0:
    from autocvd import autocvd
    autocvd(num_gpus=args.gpus)

# Bootstrap distributed mode first (raw jax, no astronomix import yet).
import jax  # noqa: E402

# CRITICAL: jax.distributed.initialize() must run before ANYTHING touches the
# backend (see weak_scaling_hydro.py for the full rationale).
if _multi:
    _cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    _localid = int(os.environ.get("SLURM_LOCALID", "0"))
    _local_ids = [_localid] if len(_cvd) > 1 else [0]
    jax.distributed.initialize(local_device_ids=_local_ids)

jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)  # fp32

# Now safe to import astronomix.
from astronomix.parallel.distributed import init_distributed  # noqa: E402
from astronomix.option_classes.simulation_config import (  # noqa: E402
    FINITE_DIFFERENCE,
    PALLAS,
    RK4_LSRK,
    SimulationConfig,
    BackendConfig,
)
from astronomix.test_setups.mhd.alfven_wave3D import (  # noqa: E402
    CPAlfvenWave3DSettings,
    build_cp_alfven_state_sharded,
)

_PYTESTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "pytests"
)
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)

from _benchmark_utils import run_weak_scaling_point  # noqa: E402

info = init_distributed()
G = info.global_device_count

DATA_DIR = os.path.join(_HERE, "data", "weak_scaling")

# Global grid: shard X across all G devices, keep Y/Z fixed per process.
GLOBAL = (args.bx * G, args.by, args.bz)
MESH_SHAPE = (1, G, 1, 1)
# Box numerically equal to the grid -> uniform spacing h = 1 at every rung.
BOX = (float(GLOBAL[0]), float(GLOBAL[1]), float(GLOBAL[2]))

# A memory-lean FD/Pallas config; LSRK4 + donate keep the per-GPU footprint low.
base_config = SimulationConfig(
    backend_config=BackendConfig(
        backend=PALLAS,
        pallas_block_shape=_BLOCK,
        pallas_use_triton=True,
        pallas_interpret=False,
        use_approximate_rsqrt=args.approx_rsqrt,
        pallas_ct=args.pallas_ct,
    ),
    solver_mode=FINITE_DIFFERENCE,
    time_integrator=RK4_LSRK,
    mhd=True,
    dimensionality=3,
    donate_state=True,
    progress_bar=False,
    print_elapsed_time=True,
)

# Tiny amplitude: the benchmark box is not wave-periodic, so keep the field a
# linear perturbation on the uniform (rho, p, B) background.
settings = CPAlfvenWave3DSettings(box_size=BOX, amplitude=1.0e-6)

if info.process_index == 0:
    total = GLOBAL[0] * GLOBAL[1] * GLOBAL[2]
    print(
        f"[weak-mhd] G={G} mesh={MESH_SHAPE} global_grid={GLOBAL} "
        f"({total:.3e} cells, ~{total / G:.3e} cells/GPU) block={_BLOCK} "
        f"steps={args.steps} dt={args.dt}",
        flush=True,
    )

run_weak_scaling_point(
    build_cp_alfven_state_sharded,
    base_config,
    settings,
    mesh_shape=MESH_SHAPE,
    global_cells=GLOBAL,
    box_size=BOX,
    cfl=args.cfl,
    dt=args.dt,
    num_timesteps=args.steps,
    name=f"weak_mhd_pallas_{args.tag}",
    data_dir=DATA_DIR,
    setup_key="mhd_weak",
)
