#!/bin/bash
#SBATCH --job-name=turb_hero
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Driven MHD turbulence (low-Mach ICM case by default), sharded across all
# allocated GPUs.  The driver auto-resumes from the latest checkpoint in
# CKPT, so hitting the walltime is harmless: resubmit the same command and
# the run continues.  Override the shape on the command line::
#
#     sbatch --nodes=16 --time=12:00:00 \
#         --export=ALL,N=2048,TCROSS=5,NSEG=100,CKPT=/e/scratch/astronomix/turb2048,TAG=icm2048 \
#         pytests/runners/turb_hero.sh

source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh

N="${N:-256}"
MTURB="${MTURB:-0.5}"
BETA="${BETA:-1e6}"
# F0 = 3.5 saturates at v_rms ~ 1.44 (measured at 256^3); 2.4 targets the
# v_rms ~ 1 the normalisation assumes, i.e. M_turb ~ the --mturb aim.
F0="${F0:-2.4}"
TCROSS="${TCROSS:-3}"
NSEG="${NSEG:-30}"
SYNTH="${SYNTH:-64}"
BLOCK="${BLOCK:-4x4x8}"
BLOCK="${BLOCK//x/,}"
CKPT="${CKPT:?set CKPT (checkpoint dir on scratch)}"
TAG="${TAG:?set TAG}"

srun --gpu-bind=none python examples/scripts/forward/mhd/turbulence/turbulence_dist.py \
    --mturb "$MTURB" --beta "$BETA" --N "$N" --tcross "$TCROSS" --F0 "$F0" \
    --nseg "$NSEG" --synth "$SYNTH" --block-shape "$BLOCK" \
    --ckpt "$CKPT" --tag "$TAG"
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
