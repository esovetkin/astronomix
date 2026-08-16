#!/bin/bash
#SBATCH --job-name=dist_sanity_2n
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh

srun --gpu-bind=none python pytests/_dist_sanity.py
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
