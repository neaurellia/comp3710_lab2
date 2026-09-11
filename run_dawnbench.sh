#!/bin/bash
#SBATCH --job-name=dawnbench
#SBATCH --output=dawnbench_%j.out      # %j is the job id
#SBATCH --error=dawnbench_%j.err
#SBATCH --time=00:45:00                # walltime; 45 min is generous for 30 epochs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # must be >= DAWN_WORKERS below
#SBATCH --mem=16G
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1                  # CHECK THIS: some clusters use --gpus=1 instead

# ---------------------------------------------------------------------------
# COMP3710 Part 3.2 - DAWNBench ResNet-18 on CIFAR-10
#
# Submit with:   sbatch run_dawnbench.sh
# Watch with:    squeue -u $USER        /  tail -f dawnbench_<jobid>.out
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
# CIFAR-10 must already be downloaded (see PRE-DOWNLOAD below) because compute
# nodes typically have no outbound internet access.
export DAWN_DATA="$HOME/data"
export DAWN_DOWNLOAD=0

export DAWN_EPOCHS=30
export DAWN_BATCH=128
export DAWN_MAX_LR=0.1
export DAWN_WORKERS=4                  # keep <= --cpus-per-task

export DAWN_STAGE3=1                   # full FP32 run
export DAWN_STAGE4=1                   # full mixed-precision run

# --- run -------------------------------------------------------------------
# --inplace writes outputs back into the notebook so the plots and printed
# results are preserved for the report.
# ExecutePreprocessor.timeout=-1 disables the per-cell timeout.
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    part3_dawnbench.ipynb

echo "finished: $(date)"

# ---------------------------------------------------------------------------
# PRE-DOWNLOAD (run ONCE on the login node, not in this job):
#
#   python -c "import torchvision; \
#     torchvision.datasets.CIFAR10(root='$HOME/data', train=True,  download=True); \
#     torchvision.datasets.CIFAR10(root='$HOME/data', train=False, download=True)"
#
# QUICK SMOKE TEST (stages 1-2 only, ~1 min, verifies the environment works
# before you queue for a long GPU allocation):
#
#   DAWN_STAGE3=0 DAWN_STAGE4=0 DAWN_DATA=$HOME/data DAWN_DOWNLOAD=0 \
#     jupyter nbconvert --to notebook --execute --inplace part3_dawnbench.ipynb
#
# LIVE DEMO (one training epoch, ~10-20 s on an A100):
#
#   DAWN_EPOCHS=1 DAWN_STAGE4=1 DAWN_STAGE3=0 DAWN_DATA=$HOME/data DAWN_DOWNLOAD=0 \
#     jupyter nbconvert --to notebook --execute --inplace part3_dawnbench.ipynb
# ---------------------------------------------------------------------------
