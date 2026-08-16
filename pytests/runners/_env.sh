# Shared environment for astronomix Slurm runners on JUPITER (JSC).
#
# Source this by ABSOLUTE path from every runner (under Slurm, $0 is the
# spool copy, so relative includes break)::
#
#     source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh
#
# Provides: micromamba env activation, repo cwd, the JAX flags the sharded
# code path needs, and $RUN_LOG_DIR for job output.

# no `set -e`: module/micromamba shell functions misbehave under it
set -u

export ASTRONOMIX_REPO="/e/home/jusers/storcks1/jupiter/astronomix"
export RUN_LOG_DIR="/e/project1/astronomix/storcks1/runs"
mkdir -p "$RUN_LOG_DIR"

# --- micromamba env (never `conda activate`: silently falls back to
# --- system python) ---
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate astronomix

# --- JAX flags for the sharded code path ---
# The repo uses integer mesh-axis names + with_sharding_constraint (GSPMD
# style); the Shardy partitioner rejects them.
export JAX_USE_SHARDY_PARTITIONER=false
# Preallocate one large contiguous pool per process.  JAX only initialises
# the GPU picked via local_device_ids, so even with --gpu-bind=none (all four
# node GPUs visible to every task) each process preallocates only its own
# device.  On-demand allocation (PREALLOCATE=false) fragments the BFC pool
# and made a 43 GB compiled temp buffer fail on a 96 GB GH200 with plenty of
# total memory free.  0.92 leaves ~8 GB for the CUDA context and NCCL.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

# --- optional communication optimizations (gated so existing runners are
# --- unaffected; Slurm --export eats commas, hence one variable per knob) ---
# Latency-hiding scheduler: overlap collectives (halo exchanges, all-to-all)
# with compute instead of serialising them.
if [ "${OPT_LHS:-0}" = "1" ]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true"
fi
# Decompose collective-permutes into send/recv pairs and pipeline them so
# the halo exchange of one field overlaps the stencil work of another.
if [ "${OPT_P2P:-0}" = "1" ]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_pipelined_p2p=true --xla_gpu_collective_permute_decomposer_threshold=0"
fi
# NCCL transport diagnostics (stderr): verify IB/GDRDMA and per-GPU NIC use.
if [ "${OPT_NCCL_DEBUG:-0}" = "1" ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT
fi

cd "$ASTRONOMIX_REPO"
