#!/bin/bash
#SBATCH --job-name=gan_oasis
#SBATCH --output=gan_%j.out            # %j is the job id
#SBATCH --error=gan_%j.err
#SBATCH --time=02:00:00                # GANs need far more epochs than a VAE - see note below
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # must be >= GAN_WORKERS below
#SBATCH --partition=comp3710
#SBATCH --account=comp3710             # mandatory: without it the job hangs in PartitionConfig
#SBATCH --gres=gpu:1
# NOTE: deliberately no --mem. This partition rejects it.

# ---------------------------------------------------------------------------
# COMP3710 Part 4 Task 3 - WGAN-GP brain MR generation on OASIS
#
# Submit with:   sbatch run_gan.sh
# Watch with:    squeue -u $USER      /  tail -f gan_<jobid>.out
# Cancel with:   scancel <jobid>
# Override config, e.g.:
#                sbatch --export=ALL,GAN_EPOCHS=200 run_gan.sh
#
# RUNTIME: at 64x64 with batch 64, one epoch is roughly 20-40 s on a cluster GPU
# (151 iterations x 5 critic steps, each including the second-derivative gradient
# penalty). So 100 epochs ~ 40-70 min, 200 epochs ~ 1.5-2.5 h. At 128x128 multiply
# by ~3-4. If you raise GAN_EPOCHS past ~150, raise --time above too.
# ---------------------------------------------------------------------------

set -euo pipefail

echo "job     : $SLURM_JOB_ID"
echo "node    : $(hostname)"
echo "started : $(date)"
nvidia-smi || echo "WARNING: nvidia-smi failed - no GPU visible?"

# --- environment -----------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./pytorch-env

# --- configuration ---------------------------------------------------------
# ${VAR:-default} means "use VAR if already set, else this default", so
# sbatch --export=ALL,GAN_EPOCHS=200 actually overrides these.
export GAN_DATA="${GAN_DATA:-/home/groups/comp3710/OASIS}"

export GAN_IMG="${GAN_IMG:-64}"            # 64 is the stable regime; 128 supported
export GAN_LATENT="${GAN_LATENT:-128}"     # length of z
export GAN_NGF="${GAN_NGF:-64}"            # generator base width
export GAN_NDF="${GAN_NDF:-64}"            # critic base width
export GAN_BATCH="${GAN_BATCH:-64}"
export GAN_EPOCHS="${GAN_EPOCHS:-100}"

# WGAN-GP hyperparameters. These are load-bearing - change them only deliberately.
export GAN_LR="${GAN_LR:-1e-4}"
export GAN_BETA1="${GAN_BETA1:-0.0}"       # 0.0, not the 0.9 you would use for a classifier
export GAN_BETA2="${GAN_BETA2:-0.9}"
export GAN_N_CRITIC="${GAN_N_CRITIC:-5}"   # critic steps per generator step
export GAN_LAMBDA_GP="${GAN_LAMBDA_GP:-10.0}"

export GAN_SAMPLE_EVERY="${GAN_SAMPLE_EVERY:-10}"   # epochs between fixed-z snapshots
export GAN_SAVE_EVERY="${GAN_SAVE_EVERY:-20}"       # epochs between checkpoint overwrites
export GAN_WORKERS="${GAN_WORKERS:-4}"              # keep <= --cpus-per-task
export GAN_SEED="${GAN_SEED:-42}"

export GAN_STAGE2="${GAN_STAGE2:-1}"   # smoke test
export GAN_STAGE3="${GAN_STAGE3:-1}"   # full training
export GAN_STAGE4="${GAN_STAGE4:-1}"   # mode-collapse evidence  <- the graded deliverable
export GAN_STAGE5="${GAN_STAGE5:-1}"   # artifacts + results JSON

# Disk: two checkpoints (~14 MB each at the defaults) overwritten in place, plus the
# progress/ snapshots. Flat regardless of run length - safe for the 16GB home quota.
export GAN_OUT="${GAN_OUT:-./gan_outputs}"

# --- run -------------------------------------------------------------------
# --inplace writes outputs back into the notebook so plots and printed results
# are preserved for the report.
# ExecutePreprocessor.timeout=-1 disables the per-cell timeout.
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    part4_gan.ipynb

echo "finished: $(date)"
echo "--- results ---"
cat "$GAN_OUT/gan_results.json" || true
echo
echo "artifacts:"
ls -la "$GAN_OUT" || true
echo "fixed-z progression:"
ls "$GAN_OUT/progress" || true

# ---------------------------------------------------------------------------
# QUICK SMOKE TEST (stages 1-2 only, ~2 min) - run BEFORE queueing a long job.
# Confirms the env, the data path, and that the gradient penalty's second
# derivative does not NaN:
#
#   GAN_STAGE2=1 GAN_STAGE3=0 GAN_STAGE4=0 GAN_STAGE5=0 \
#     jupyter nbconvert --to notebook --execute --inplace part4_gan.ipynb
#
# LIVE DEMO (inference + all four mode-collapse checks, no training - loads the
# saved generator). This is what to run in front of a demonstrator:
#
#   GAN_STAGE2=0 GAN_STAGE3=0 GAN_STAGE4=1 GAN_STAGE5=1 \
#     jupyter nbconvert --to notebook --execute --inplace part4_gan.ipynb
#
# IF THE BRAINS ARE NOT REALISTIC ENOUGH, in rough order of payoff:
#   sbatch --export=ALL,GAN_EPOCHS=200 run_gan.sh          # undertrained is the usual cause
#   sbatch --export=ALL,GAN_NGF=96,GAN_NDF=96 run_gan.sh   # more capacity
#   sbatch --export=ALL,GAN_IMG=128,GAN_BATCH=32,GAN_EPOCHS=200 run_gan.sh   # more detail
# Get a good 64x64 result BEFORE trying 128 - it is markedly less stable.
#
# IF THE DIVERSITY RATIO IS LOW (< ~0.5), first check it is not just undertrained
# by looking at progress/ - if the fixed-z samples are still changing shape between
# snapshots, it needs more epochs, not different hyperparameters.
# ---------------------------------------------------------------------------
