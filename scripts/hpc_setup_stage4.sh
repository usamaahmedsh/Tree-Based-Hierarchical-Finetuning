#!/usr/bin/env bash
# =============================================================================
# hpc_setup_stage4.sh  —  One-time environment setup for Stage 4 (QLoRA) on BU SCC
#
# Run ONCE from a login node or interactive session:
#   ssh scc1.bu.edu
#   cd /path/to/Tree-Based-Hierarchical-Finetuning
#   bash scripts/hpc_setup_stage4.sh
#
# Interactive session to test after setup:
#   qrsh -P <your_project> \
#        -l h_rt=2:00:00 \
#        -l gpus=1 \
#        -l gpu_c=8.0 \
#        -l gpu_mem=40G \
#        -pe omp 8 \
#        -l mem_per_core=8G
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="stage4_finetune"
PYTHON_VERSION="3.11"

echo "============================================================"
echo "  BU SCC  —  Stage 4 QLoRA Fine-Tuning Environment Setup"
echo "  Project : $PROJECT_DIR"
echo "  Env     : $ENV_NAME"
echo "============================================================"

# ── 1. Load modules ──────────────────────────────────────────────────────────
echo ""
echo "[1/4] Loading modules..."
module purge
module load miniconda/23.11.0
module load cuda/12.2

# ── 2. Create conda environment ──────────────────────────────────────────────
echo ""
echo "[2/4] Creating conda env '$ENV_NAME' (Python $PYTHON_VERSION)..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Env '$ENV_NAME' already exists — skipping creation."
else
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 3. Install packages ───────────────────────────────────────────────────────
echo ""
echo "[3/4] Installing packages..."
pip install --upgrade pip --quiet

# PyTorch with CUDA 12.1 wheel (compatible with cuda/12.2 module)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 --quiet

# Core fine-tuning stack
pip install \
    transformers>=4.44.0 \
    peft>=0.12.0 \
    trl>=1.2.0 \
    accelerate>=0.34.0 \
    bitsandbytes>=0.43.0 \
    --quiet

# Data + evaluation
pip install \
    datasets>=2.18.0 \
    evaluate>=0.4.0 \
    numpy \
    psutil \
    --quiet

echo "  Packages installed."

# ── 4. Verify install ─────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying install..."
python - <<'PYEOF'
import torch, transformers, peft, trl, accelerate, bitsandbytes, datasets
print(f"  torch         : {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print(f"  transformers  : {transformers.__version__}")
print(f"  peft          : {peft.__version__}")
print(f"  trl           : {trl.__version__}")
print(f"  accelerate    : {accelerate.__version__}")
print(f"  bitsandbytes  : {bitsandbytes.__version__}")
print(f"  datasets      : {datasets.__version__}")
if torch.cuda.is_available():
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PYEOF

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Submit the training job:"
echo "    qsub scripts/hpc_submit_stage4.sh"
echo ""
echo "  Monitor:"
echo "    qstat -u \$USER"
echo "    tail -f logs/stage4_\$JOB_ID.log"
echo "============================================================"
