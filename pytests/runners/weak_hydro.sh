#!/bin/bash
#SBATCH --job-name=weak_hydro
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Weak-scaling rung for the FD (Pallas) hydro benchmark on JUPITER.  The
# geometry is controlled by environment variables so the same runner serves
# every rung of the ladder (override the Slurm shape on the command line)::
#
#     sbatch --nodes=4 --ntasks-per-node=4 \
#         --export=ALL,BX=128,BY=2048,BZ=2048,STEPS=10,BLOCK=4x4x8,TAG=gh200 \
#         pytests/runners/weak_hydro.sh
#
# BLOCK uses 'x' separators because commas inside --export values are eaten
# by Slurm's own list parsing; the runner converts them back.
#
# One process per GPU; G = nodes * ntasks-per-node; global grid
# (BX * G, BY, BZ).

source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh

BX="${BX:-128}"
BY="${BY:-2048}"
BZ="${BZ:-2048}"
STEPS="${STEPS:-10}"
BLOCK="${BLOCK:-4x4x8}"
BLOCK="${BLOCK//x/,}"
TAG="${TAG:-gh200}"

srun --gpu-bind=none python examples/scripts/scaling/weak_scaling_hydro.py \
    --bx "$BX" --by "$BY" --bz "$BZ" \
    --steps "$STEPS" --block-shape "$BLOCK" --tag "$TAG"
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
