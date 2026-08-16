#!/bin/bash
#SBATCH --job-name=turb_diag
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Offline diagnostics over a turbulence_dist.py checkpoint sequence:
# scalar time series on the GPUs (sharded restore) + host-side spectra for
# every SPECTRA_EVERY-th checkpoint.  2 nodes are enough for a 2048^3 run
# (sharded state restore across 8 GPUs; one field gathers to 34 GB of the
# 480 GB host memory for each spectrum).

source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh

N="${N:?set N}"
CKPT="${CKPT:?set CKPT}"
OUT="${OUT:?set OUT (output npz)}"
MTURB="${MTURB:-0.5}"
SPECTRA_EVERY="${SPECTRA_EVERY:-5}"

srun --gpu-bind=none python examples/scripts/forward/mhd/turbulence/turb_diagnostics.py \
    --ckpt "$CKPT" --N "$N" --mturb "$MTURB" \
    --spectra-every "$SPECTRA_EVERY" --out "$OUT"
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
