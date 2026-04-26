#!/bin/bash -l
# =============================================================================
# hpc_submit.sh  —  SLURM/SGE job script for Stage 3 vLLM on BU SCC
#
# BU SCC uses SGE (not SLURM). Submit with:
#   qsub scripts/hpc_submit.sh
#
# Monitor:
#   qstat -u $USER
#   tail -f data/stage3/stage3.log
# =============================================================================

# ── SGE directives ────────────────────────────────────────────────────────────
#$ -N stage3_vllm
#$ -P YOUR_PROJECT_NAME          # <-- REPLACE with your BU SCC project name
#$ -l h_rt=24:00:00              # 24-hour wall time (adjust as needed)
#$ -l gpus=1                     # 1 GPU (140GB)
#$ -l gpu_c=8.0                  # CUDA compute capability >= 8.0 (A100/H100)
#$ -l mem_per_core=8G            # 8 GB RAM per core × 32 cores = 256 GB RAM
#$ -pe omp 32                    # 32 CPU cores — feeds ProcessPoolExecutor workers
#$ -j y                          # merge stdout + stderr
#$ -o logs/stage3_vllm_$JOB_ID.log
#$ -cwd                          # run from submission directory

# ── Environment ───────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(pwd)"
ENV_NAME="stage3_vllm"

echo "======================================================"
echo "  Job ID        : $JOB_ID"
echo "  Host          : $(hostname)"
echo "  GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Start         : $(date)"
echo "  Project dir   : $PROJECT_DIR"
echo "======================================================"

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load miniconda/23.11.0
module load cuda/12.2

# ── Activate conda env ────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── HuggingFace token (set in your ~/.bashrc or pass via qsub -v) ─────────────
# export HF_TOKEN="hf_xxxxxxxxxxxx"   # uncomment and fill in, OR set in ~/.bashrc

# ── CUDA visibility (SGE assigns GPU automatically) ───────────────────────────
export CUDA_VISIBLE_DEVICES=$SGE_GPU

# ── Create log dir ────────────────────────────────────────────────────────────
mkdir -p logs data/stage3

# ── Run Stage 3 vLLM ──────────────────────────────────────────────────────────
# Default model: meta-llama/Meta-Llama-3.1-8B-Instruct
# If you downloaded Qwen instead, change --vllm-model accordingly.
#
# --vllm-batch 256   : safe for A100 40GB / V100 32GB
# --vllm-batch 512   : use on A100 80GB
# --tensor-parallel 1: 1 GPU; increase if you requested gpus=2 etc.

# CPU workers = SGE slots - 2 (leave 2 cores for OS + vLLM scheduler)
N_CPU_WORKERS=$(( NSLOTS - 2 ))

python scripts/stage3_noise_injection.py \
    --backend vllm \
    --vllm-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --vllm-batch 2048 \
    --cpu-workers "$N_CPU_WORKERS" \
    --prefetch 6 \
    --tensor-parallel 1 \
    --temperature 0.2

echo "======================================================"
echo "  Finished : $(date)"
echo "  Output   : $PROJECT_DIR/data/stage3/noised_chunks.jsonl"
echo "======================================================"
