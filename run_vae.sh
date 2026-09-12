#!/bin/bash
#SBATCH --job-name=vae_oasis
#SBATCH --output=vae_%j.out            # %j is the job id
#SBATCH --error=vae_%j.err
#SBATCH --time=01:00:00                # 30 epochs at 64x64 runs in ~10-15 min; this is slack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # must be >= VAE_WORKERS below
#SBATCH --partition=comp3710
#SBATCH --account=comp3710             # mandatory: without it the job hangs in PartitionConfig
#SBATCH --gres=gpu:1

# ---------------------------------------------------------------------------
# COMP3710 Part 4 Task 1 - VAE on the OASIS brain MRI dataset
#
# Submit with:   sbatch run_vae.sh
# Watch with:    squeue -u $USER      /  tail -f vae_<jobid>.out
# Cancel with:   scancel <jobid>
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
# Read-only group directory. Nothing is copied out of it and nothing is written to it.
export VAE_DATA="/home/groups/comp3710/OASIS"

export VAE_IMG=64                      # 256 -> 64; must be divisible by 16. Set 128 for sharper
export VAE_LATENT=16                   # set 2 for the classic direct-sweep manifold grid
export VAE_BATCH=128
export VAE_EPOCHS=30
export VAE_LR=1e-3
export VAE_BETA=1.0                    # 1.0 is the true ELBO; >1 is a beta-VAE
export VAE_KL_WARMUP=10                # linear beta ramp, guards against posterior collapse
export VAE_WORKERS=4                   # keep <= --cpus-per-task

export VAE_STAGE2=1                    # smoke test
export VAE_STAGE3=1                    # full training
export VAE_STAGE4=1                    # manifold visualisation  <- the marked deliverable
export VAE_STAGE5=1                    # save PNG artifacts

# Outputs land here: loss curves, manifold grid, latent scatter, reconstructions,
# samples, interpolation, plus one ~15 MB checkpoint. Well inside the 16GB home quota.
export VAE_OUT="./vae_outputs"

# --- run -------------------------------------------------------------------
# --inplace writes outputs back into the notebook so plots and printed results
# are preserved for the report.
# ExecutePreprocessor.timeout=-1 disables the per-cell timeout.
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    part4_vae.ipynb

echo "finished: $(date)"
echo "artifacts:"
ls -la "$VAE_OUT" || true

# ---------------------------------------------------------------------------
# QUICK SMOKE TEST (stages 1-2 only, ~1 min) - run this BEFORE queueing a long
# job, to confirm the env and the data path work. Safe to run on a login node:
#
#   VAE_STAGE2=1 VAE_STAGE3=0 VAE_STAGE4=0 VAE_STAGE5=0 \
#     jupyter nbconvert --to notebook --execute --inplace part4_vae.ipynb
#
# LIVE DEMO (one training epoch end to end, then the manifold):
#
#   VAE_EPOCHS=1 VAE_STAGE3=1 VAE_STAGE4=1 VAE_STAGE5=1 \
#     jupyter nbconvert --to notebook --execute --inplace part4_vae.ipynb
#
# CLASSIC 2D MANIFOLD (direct latent sweep instead of the PCA plane):
#
#   sbatch --export=ALL,VAE_LATENT=2 run_vae.sh
#
# NOTE ON UMAP: the notebook prefers UMAP but falls back to sklearn's t-SNE and
# reports which it used. Compute nodes have no internet, so if you want UMAP,
# install it on the login node first:
#   conda activate ./pytorch-env && pip install umap-learn
# ---------------------------------------------------------------------------
