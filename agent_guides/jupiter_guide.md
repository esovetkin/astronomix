# JUPITER (JSC) guide — storage, environment, and queueing

Working reference for running astronomix on the JUPITER Booster at Jülich
Supercomputing Centre. Facts below were verified on the system on 2026-08-05;
re-check limits with the listed commands if something looks off.

**The one thing to know first: JUPITER is aarch64 (ARM).** Every node — login
and compute — is an NVIDIA GH200 Grace-Hopper superchip (72-core Grace ARM CPU
+ H100-class GPU, 96 GB HBM). x86 wheels, containers, and binaries do not run
here. `pip` on the machine resolves aarch64 wheels automatically; just never
copy environments over from an x86 cluster.

## Storage map

| Location | Env var | What it's for |
|---|---|---|
| `/e/home/jusers/storcks1/jupiter` | `$HOME` | Config and code only. **Small quota: ~19 GB soft / 21 GB hard, 80k inodes.** |
| `/e/project1/astronomix/storcks1/` | under `$PROJECT_astronomix` | Persistent bulk storage: results, checkpoints, datasets, environments. No practical quota. |
| `/e/scratch/astronomix` | `$SCRATCH_astronomix` | Large transient run output. **Purged periodically — anything worth keeping must be moved to project.** |
| `/e/fscratch/astronomix` | `$FSCRATCH_astronomix` | Flash (fast) scratch — same purge caveat; use for I/O-heavy runs. |

Rules of thumb:

- **Never write simulation output into `$HOME`** — point run output at
  `$SCRATCH_astronomix` (transient) or `/e/project1/astronomix/storcks1/`
  (keep), and don't commit large `.npz`/animations to the repo (history bloat
  is why the clone is now a `--filter=blob:none` partial clone).
- `~/.vscode-server` and `~/micromamba` are already **symlinks** into
  `/e/project1/astronomix/storcks1/` to keep home under quota. If you create
  new caches (pip, HF, etc.), put them on project too.
- Check quota and usage: `jutil user dataquota` (limits) and
  `du -sh ~/.[!.]* ~/*` / `find ~ | wc -l` (usage; the inode limit bites
  before the byte limit).

## Software environment

Two options, both verified present:

1. **System module** (quick): `module load jax/0.8.1` (there are also
   `CUDA/13`, `NCCL`, `PyTorch/2.9.1` modules; `(g)` marks GPU-enabled).
2. **micromamba env** (flexible — what we control): env `astronomix`
   (Python 3.12) exists but is currently **empty** — no JAX installed yet.
   To make it usable:

   ```bash
   micromamba activate astronomix
   pip install -U "jax[cuda12]"        # check current CUDA-extra name in JAX docs; system CUDA is 13
   pip install -e .                    # editable install of astronomix, from the repo root
   ```

   Editable install matters: queued jobs then pick up code fixes at runtime
   (see the multi-node skill). Activate with `micromamba activate`, **not**
   `conda activate` (silently falls back to system python).

For CPU-only style/import checks, stay off the GPU entirely
(`JAX_PLATFORMS=cpu`, as in `agent_guides/CLAUDE.md`).

## GPUs on the login node — shared, be polite

Login nodes have a GH200 GPU that **everyone shares** (it was 84/96 GB
occupied when this was written). For any interactive JAX-on-GPU run, use
`autocvd` to pick a free GPU rather than grabbing device 0 blindly:

```bash
autocvd -n 1 -- python my_script.py    # sets CUDA_VISIBLE_DEVICES to a free GPU
```

If no GPU is free (or the run is more than a smoke test), get a compute node
instead — see interactive jobs below. Real runs never belong on the login node.

## Queueing (Slurm)

- **Account (required):** `--account=astronomix`
- **Partition:** `booster` (default) — 5566 nodes × 4 GH200 each,
  `--gres=gpu:4`. `largebooster` exists for very large runs (was down at time
  of writing).
- **Walltime:** default 1 h if unspecified; **max 12 h** (QoS `part_booster`).
- **Compute budget:** 570,000 core-h for 2026-07-01 → 2026-11-30. Billing is
  per Grace core: one node-hour = 288 core-h (≈ 1,980 node-hours total), one
  GPU (¼ node) ≈ 72 core-h/h. Check with `jutil user cpuquota`.

### Batch job template

Adapted from `examples/scripts/multi_gpu/multi_node.sh` — the launch model is
**one process per GPU**, and the `srun --gpu-bind=none` line is load-bearing
(see gotcha list below):

```bash
#!/bin/bash
#SBATCH --job-name=astronomix
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gres=gpu:4                 # 4 GH200 per node on JUPITER
#SBATCH --time=02:00:00              # max 12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -u   # no `set -e`: module/micromamba shell functions misbehave under it

export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate astronomix

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_USE_SHARDY_PARTITIONER=false   # repo uses GSPMD-style sharding

srun --gpu-bind=none python examples/scripts/multi_gpu/multi_node.py
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc            # never mask a failed srun with an unconditional "DONE"
```

Single-node / single-GPU jobs: same skeleton with `--nodes=1` and
`--ntasks-per-node=1 --gres=gpu:1`.

### Interactive jobs

```bash
salloc --account=astronomix --partition=booster --nodes=1 --gres=gpu:4 --time=01:00:00
srun --gpu-bind=none --pty bash    # shell on the compute node
```

### Monitoring

```bash
squeue --me                                   # queue state
scontrol show job <id>                        # why is it pending
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS
jutil user cpuquota                           # budget burn-down
```

## Multi-GPU / multi-node JAX

All the distributed-JAX gotchas (GPU binding, `jax.distributed.initialize()`
ordering, Shardy flags, sharding-aware ICs, the validation ladder) are
documented in `.claude/skills/multi_node/MULTI_NODE.md`. That skill was
written on HoreKa, but the launch model and every JAX-side fix transfer
directly; only the site specifics change:

- partition/account: `booster` / `astronomix` (not `accelerated-h100`)
- GPUs per node: 4 (GH200), so `--ntasks-per-node=4`
- there is no separate dev queue — validate with a short-walltime (≤1 h)
  1-node job on `booster` before scaling out
- architecture: aarch64 — rebuild any env rather than copying one over

Follow its validation ladder: `_dist_sanity.py` on 1 node → 2 nodes → real
2-node run → scale.
