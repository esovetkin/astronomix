#!/bin/bash
#SBATCH --job-name=weak_selfgrav
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Weak-scaling rung for the FD (Pallas) hydro + self-gravity benchmark on
# JUPITER (see weak_hydro.sh for the launch pattern; BLOCK uses 'x'
# separators).

source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh

BX="${BX:-64}"
BY="${BY:-1024}"
BZ="${BZ:-1024}"
STEPS="${STEPS:-10}"
BLOCK="${BLOCK:-4x4x8}"
BLOCK="${BLOCK//x/,}"
TAG="${TAG:-gh200}"

srun --gpu-bind=none python examples/scripts/scaling/weak_scaling_selfgrav.py \
    --bx "$BX" --by "$BY" --bz "$BZ" \
    --steps "$STEPS" --block-shape "$BLOCK" --tag "$TAG"
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
