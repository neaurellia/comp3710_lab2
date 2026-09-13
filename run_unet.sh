#!/bin/bash
#SBATCH --job-name=unet_oasis
#SBATCH --output=unet_%j.out           # %j is the job id
#SBATCH --error=unet_%j.err
#SBATCH --time=02:00:00                # 25 epochs at 128x128 is ~30-50 min; this is slack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # must be >= UNET_WORKERS below
#SBATCH --partition=comp3710
#SBATCH --account=comp3710             # mandatory: without it the job hangs in PartitionConfig
#SBATCH --gres=gpu:1
# NOTE: deliberately no --mem. This partition rejects it.

# ---------------------------------------------------------------------------
# COMP3710 Part 4 Task 2 - UNet segmentation of OASIS brain MRI
#
# Submit with:   sbatch run_unet.sh
# Watch with:    squeue -u $USER      /  tail -f unet_<jobid>.out
# Cancel with:   scancel <jobid>
# Override any config value, e.g.:
#                sbatch --export=ALL,UNET_IMG=256 run_unet.sh
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
# sbatch --export=ALL,UNET_IMG=256 actually overrides these.
export UNET_DATA="${UNET_DATA:-/home/groups/comp3710/OASIS}"

export UNET_IMG="${UNET_IMG:-128}"       # 256 available; must be divisible by 16
export UNET_BASE="${UNET_BASE:-32}"      # first encoder width; 64 is the paper's
export UNET_BATCH="${UNET_BATCH:-16}"
export UNET_EPOCHS="${UNET_EPOCHS:-25}"
export UNET_LR="${UNET_LR:-1e-3}"
export UNET_CE_W="${UNET_CE_W:-0.5}"     # cross-entropy weight
export UNET_DICE_W="${UNET_DICE_W:-0.5}" # Dice weight; raise if the small class lags
export UNET_WORKERS="${UNET_WORKERS:-4}" # keep <= --cpus-per-task
export UNET_SEED="${UNET_SEED:-42}"

export UNET_STAGE2="${UNET_STAGE2:-1}"   # smoke test
export UNET_STAGE3="${UNET_STAGE3:-1}"   # full training
export UNET_STAGE4="${UNET_STAGE4:-1}"   # test-set per-class DSC  <- the graded deliverable
export UNET_STAGE5="${UNET_STAGE5:-1}"   # save artifacts + results JSON

# Outputs: loss/DSC curves, per-class DSC bar chart, triptych visualisations,
# results JSON, and one checkpoint (~30 MB at UNET_BASE=32). Well inside the 16GB quota.
export UNET_OUT="${UNET_OUT:-./unet_outputs}"

# --- run -------------------------------------------------------------------
# --inplace writes outputs back into the notebook so plots and printed results
# are preserved for the report.
# ExecutePreprocessor.timeout=-1 disables the per-cell timeout.
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    part4_unet.ipynb

echo "finished: $(date)"
echo "--- per-class test DSC ---"
cat "$UNET_OUT/unet_results.json" || true
echo
echo "artifacts:"
ls -la "$UNET_OUT" || true

# ---------------------------------------------------------------------------
# QUICK SMOKE TEST (stages 1-2 only, ~1 min) - run BEFORE queueing a long job,
# to confirm the env, the data path and the image/mask pairing all work:
#
#   UNET_STAGE2=1 UNET_STAGE3=0 UNET_STAGE4=0 UNET_STAGE5=0 \
#     jupyter nbconvert --to notebook --execute --inplace part4_unet.ipynb
#
# LIVE DEMO (inference only - loads the saved checkpoint, runs the test set,
# prints the per-class DSC table and saves the visualisations). This is the
# cell to run in front of a demonstrator; it needs no training:
#
#   UNET_STAGE2=0 UNET_STAGE3=0 UNET_STAGE4=1 UNET_STAGE5=1 \
#     jupyter nbconvert --to notebook --execute --inplace part4_unet.ipynb
#
# IF CLASS 1 FALLS SHORT OF 0.9, in rough order of expected payoff:
#   sbatch --export=ALL,UNET_IMG=256 run_unet.sh                  # boundary accuracy
#   sbatch --export=ALL,UNET_DICE_W=0.7,UNET_CE_W=0.3 run_unet.sh # push class balance
#   sbatch --export=ALL,UNET_BASE=64 run_unet.sh                  # more capacity
#   sbatch --export=ALL,UNET_EPOCHS=50 run_unet.sh                # longer schedule
# Note UNET_IMG=256 needs roughly 4x the memory, so pair it with UNET_BATCH=8.
# ---------------------------------------------------------------------------
