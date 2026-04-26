#!/usr/bin/env bash
# =============================================================================
# hpc_setup.sh  —  One-time environment setup for Stage 3 (vLLM) on BU SCC
#
# Run this ONCE before submitting your job. It:
#   1. Loads BU SCC modules
#   2. Creates conda env with vLLM + deps
#   3. Downloads NLTK corpora
#   4. Downloads data from HuggingFace  (skips files that already exist)
#   5. Downloads the vLLM model
#
# Usage (from a login node or interactive session):
#   ssh scc1.bu.edu
#   cd /path/to/Tree-Based-Hierarchical-Finetuning
#   bash scripts/hpc_setup.sh
#
# ── Requesting resources on BU SCC ──────────────────────────────────────────
#
#   INTERACTIVE (qrsh) — use this to test / debug:
#
#     qrsh -P <your_project> \
#          -l h_rt=4:00:00 \
#          -l gpus=1 \
#          -l gpu_c=8.0 \
#          -l gpu_mem=80G \
#          -pe omp 32 \
#          -l mem_per_core=8G
#
#   Key flags:
#     -P <project>      your BU SCC project group (e.g. "cs" or "ds")
#     -l h_rt=HH:MM:SS  wall-clock time limit  (setup needs ~30 min)
#     -l gpus=1         number of GPUs  (use 2 if you want tensor_parallel=2)
#     -l gpu_c=8.0      min CUDA compute capability (8.0 = A100/H100)
#     -l gpu_mem=80G    min GPU VRAM  (request 80G per GPU; omit if unavailable)
#     -pe omp 32        CPU cores for ProcessPoolExecutor workers
#     -l mem_per_core=8G  8 GB RAM × 32 cores = 256 GB total RAM
#
#   BATCH (qsub) — for the actual production run:
#     qsub scripts/hpc_submit.sh
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="stage3_vllm"
PYTHON_VERSION="3.11"
HF_REPO="usamaahmedsh/tree-hierarchical-finetuning"
HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"   # set via: export HF_TOKEN=hf_xxx

echo "============================================================"
echo "  BU SCC  —  Stage 3 vLLM Environment Setup"
echo "  Project : $PROJECT_DIR"
echo "  Env     : $ENV_NAME"
echo "  HF repo : $HF_REPO"
echo "============================================================"

# ── 1. Load modules ──────────────────────────────────────────────────────────
echo ""
echo "[1/6] Loading modules..."
module purge
module load miniconda/23.11.0
module load cuda/12.2

# ── 2. Create conda environment ──────────────────────────────────────────────
echo ""
echo "[2/6] Creating conda env '$ENV_NAME' (Python $PYTHON_VERSION)..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Env '$ENV_NAME' already exists — skipping creation."
else
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 3. Install Python packages ────────────────────────────────────────────────
echo ""
echo "[3/6] Installing packages..."
pip install --upgrade pip --quiet

# PyTorch (CUDA 12.1 wheel — works with cuda/12.2 module)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 --quiet

# vLLM
pip install vllm==0.6.3 --quiet

# Pipeline deps
pip install nltk loguru tqdm huggingface_hub boto3 --quiet

echo "  Packages installed."

# ── 4. Download NLTK corpora ──────────────────────────────────────────────────
echo ""
echo "[4/6] Downloading NLTK corpora..."
python -c "
import nltk
for pkg in ('wordnet', 'punkt', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'):
    nltk.download(pkg, quiet=False)
print('NLTK data ready.')
"

# ── 5. Download data from HuggingFace  (skip files that already exist) ────────
echo ""
echo "[5/6] Downloading project data from HuggingFace..."

python - <<PYEOF
import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

TOKEN   = "$HF_TOKEN"
REPO    = "$HF_REPO"
ROOT    = Path("$PROJECT_DIR")

files = [
    # (path_in_repo,                        local_path)
    ("data/pruned/pruned_chunks.jsonl",  ROOT / "data/pruned/pruned_chunks.jsonl"),
    ("data/stage3/noised_chunks.jsonl",  ROOT / "data/stage3/noised_chunks.jsonl"),
    ("data/stage3/checkpoint.json",      ROOT / "data/stage3/checkpoint.json"),
    ("scripts/stage3_noise_injection.py",ROOT / "scripts/stage3_noise_injection.py"),
    ("scripts/hpc_setup.sh",             ROOT / "scripts/hpc_setup.sh"),
    ("scripts/hpc_submit.sh",            ROOT / "scripts/hpc_submit.sh"),
]

for repo_path, local_path in files:
    local_path = Path(local_path)
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"  SKIP (exists)  {local_path.relative_to(ROOT)}")
        continue
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading    {repo_path} ...", flush=True)
    hf_hub_download(
        repo_id=REPO,
        filename=repo_path,
        repo_type="dataset",
        token=TOKEN,
        local_dir=str(ROOT),
        local_dir_use_symlinks=False,
    )
    size_mb = local_path.stat().st_size / 1e6
    print(f"  Done           {local_path.relative_to(ROOT)}  ({size_mb:.1f} MB)")

print("All data files ready.")
PYEOF

# ── 6. Download vLLM model ───────────────────────────────────────────────────
echo ""
echo "[6/6] Downloading vLLM model (Llama 3.1 8B Instruct)..."
echo ""
echo "  NOTE: Llama 3.1 8B requires accepting the Meta licence on HuggingFace:"
echo "  https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
echo ""
echo "  If you have accepted it, the model will download now using your HF token."
echo "  If not, Ctrl-C and use the Qwen alternative (no licence needed):"
echo "    python -c \"from huggingface_hub import snapshot_download;"
echo "      snapshot_download('Qwen/Qwen2.5-7B-Instruct')\""
echo "  Then pass --vllm-model Qwen/Qwen2.5-7B-Instruct in hpc_submit.sh."
echo ""

python -c "
from huggingface_hub import snapshot_download
import os
print('Downloading meta-llama/Meta-Llama-3.1-8B-Instruct ...')
snapshot_download(
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    token=os.environ.get('HF_TOKEN', ''),
    ignore_patterns=['*.pth'],  # skip old-format weights
)
print('Model ready.')
" || echo "  Model download skipped (licence not accepted or token missing). Set HF_TOKEN and re-run step 6 manually."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  INTERACTIVE TEST (get a GPU node first):"
echo "    qrsh -P <your_project> -l h_rt=2:00:00 -l gpus=1 \\"
echo "         -l gpu_c=8.0 -l gpu_mem=80G -pe omp 32 -l mem_per_core=8G"
echo "    cd $PROJECT_DIR"
echo "    conda activate $ENV_NAME"
echo "    python scripts/stage3_noise_injection.py --backend vllm --vllm-batch 512"
echo ""
echo "  BATCH JOB (edit hpc_submit.sh first — set your project name):"
echo "    qsub scripts/hpc_submit.sh"
echo "============================================================"
