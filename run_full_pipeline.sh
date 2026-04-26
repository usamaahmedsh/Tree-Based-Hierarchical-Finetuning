#!/bin/zsh

echo "========================================================="
echo "        FULL DE-BIASING PIPELINE EXECUTION (SCALE)       "
echo "========================================================="

# 1. Clean previous data
echo "[1/6] Cleaning old data directories..."
rm -rf data/raw data/chunks data/tree outputs/tree_viz

# 2. Source environment
echo "[2/6] Sourcing Python Virtual Env and .env..."
source .venv/bin/activate
export $(cat .env | grep -v '^#' | xargs)

# 3. Create necessary folders
mkdir -p data/chunks

# 4. Run Stage 0 (Download and Chunk)
echo "[3/6] Running Stage 0 (Corpus Builder) - 2000 Pages max..."
while read -r line; do
  # Ignore empty lines and headers/separators
  if [[ -n "$line" && "$line" != *"="* && "$line" != *"—"* ]]; then
    echo "-> Harvesting $line"
    python corpus_builder.py --topic "$line" --max-pages 2000
  fi
done < topics.txt

# 5. Combine chunks
echo "[4/6] Combining individual chunk files into 'chunks.jsonl'..."
python -c 'import json; from pathlib import Path; p=Path("data/chunks"); chunks=[]; [chunks.extend([json.loads(line) for line in open(f)]) for f in p.glob("*_chunks.jsonl")]; Path("data/chunks/chunks.jsonl").write_text("\n".join((json.dumps(c) for c in chunks)))'

# 5.5 Global deduplication
echo "[4.5/6] Performing Global Deduplication across all topics..."
python global_dedup.py

# 6. Run Stage 1 (Tree construction using 100% data)
echo "[5/6] Running Stage 1 (Tree Strategy Construction)..."
python stage1_hierarchical_tree.py

# 7. Generate Validation Report
echo "[6/6] Generating updated interactive visualization report..."
python build_rigorous_html_report.py

echo "========================================================="
echo "                    PIPELINE COMPLETE                    "
echo "========================================================="
