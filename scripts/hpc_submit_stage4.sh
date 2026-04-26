#!/bin/bash -l
# =============================================================================
# hpc_submit_stage4.sh  —  SGE job script for Stage 4 QLoRA fine-tuning on BU SCC
#
# Submit with:
#   qsub scripts/hpc_submit_stage4.sh
#
# Monitor:
#   qstat -u $USER
#   tail -f logs/stage4_$JOB_ID.log
# =============================================================================

# ── SGE directives ────────────────────────────────────────────────────────────
#$ -N stage4_finetune
#$ -P YOUR_PROJECT_NAME          # <-- REPLACE with your BU SCC project name
#$ -l h_rt=24:00:00              # 24-hour wall time
#$ -l gpus=1
#$ -l gpu_c=8.0                  # A100 / H100 (compute cap >= 8.0 for bf16 + 4-bit)
#$ -l gpu_mem=40G                # 40G minimum; use 80G if available
#$ -pe omp 8                     # 8 CPU cores for data loading
#$ -l mem_per_core=8G            # 64 GB total RAM
#$ -j y                          # merge stdout + stderr
#$ -o logs/stage4_$JOB_ID.log
#$ -cwd

# ── Environment ───────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="stage4_finetune"

echo "======================================================"
echo "  Job ID   : $JOB_ID"
echo "  Host     : $(hostname)"
echo "  GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Start    : $(date)"
echo "  Dir      : $(pwd)"
echo "======================================================"

# ── Modules + conda ───────────────────────────────────────────────────────────
module purge
module load miniconda/23.11.0
module load cuda/12.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── GPU assignment ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=$SGE_GPU

# ── Dirs ──────────────────────────────────────────────────────────────────────
mkdir -p logs outputs/checkpoints data/processed

# ── Run Stage 4 ───────────────────────────────────────────────────────────────
python scripts/stage4_finetune.py \
    --input  noised_chunks_hpc_relabeled.jsonl \
    --model  microsoft/Phi-3-mini-4k-instruct

echo "======================================================"
echo "  Finished : $(date)"
echo "  Best model : outputs/checkpoints/run_*/best"
echo "======================================================"
