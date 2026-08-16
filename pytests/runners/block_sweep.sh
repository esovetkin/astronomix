#!/bin/bash
#SBATCH --job-name=block_sweep
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Pallas block-shape sweep for the FD hydro kernel on a GH200 (the tuned
# block in the paper, (4,4,8), was found on H100 -- re-verify on this
# architecture before committing the whole weak-scaling campaign to it).

source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh

python examples/scripts/scaling/scaling_campaign.py \
    --phase block --gpus 1 --block-n "${BLOCK_N:-256}" --tag gh200
rc=$?
echo "=== exit code: $rc ==="
exit $rc
