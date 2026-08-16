#!/bin/bash
#SBATCH --job-name=weak_mhd
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Weak-scaling rung for the FD (Pallas) MHD benchmark on JUPITER (see
# weak_hydro.sh for the launch pattern; BLOCK uses 'x' separators).

source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh

BX="${BX:-64}"
BY="${BY:-2048}"
BZ="${BZ:-2048}"
STEPS="${STEPS:-10}"
BLOCK="${BLOCK:-4x4x8}"
BLOCK="${BLOCK//x/,}"
TAG="${TAG:-gh200}"

EXTRA_ARGS=""
if [ "${RSQRT:-0}" = "1" ]; then
    EXTRA_ARGS="--approx-rsqrt"
fi
if [ "${PALLAS_CT:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --pallas-ct"
fi

srun --gpu-bind=none python examples/scripts/scaling/weak_scaling_mhd.py \
    --bx "$BX" --by "$BY" --bz "$BZ" \
    --steps "$STEPS" --block-shape "$BLOCK" --tag "$TAG" $EXTRA_ARGS
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
