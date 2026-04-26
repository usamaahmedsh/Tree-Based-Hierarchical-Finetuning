#!/usr/bin/env python3
"""
Prepare full_chunks_with_tree.jsonl
====================================
Joins chunks.jsonl (335k raw chunks) with data/tree/leaf_assignments.jsonl
to produce a single enriched file ready for stage3_noise_injection.py.

Adds per-chunk:  _gidx, leaf_id, label_path, majority_score, noise_tier

Usage:
    python scripts/prepare_full_data.py
"""

import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
CHUNKS_FILE     = ROOT / "data/chunks/chunks.jsonl"
TREE_FILE       = ROOT / "data/tree/leaf_assignments.jsonl"
OUT_FILE        = ROOT / "data/full_chunks_with_tree.jsonl"

# ── Noise tier thresholds (must match stage3_noise_injection.py) ───────────────
HIGH_THRESHOLD  = 1.5    # majority_score > this  → HIGH
MED_THRESHOLD   = 0.5    # majority_score > this  → MED, else LOW


def assign_tier(majority_score: float) -> str:
    if majority_score > HIGH_THRESHOLD:
        return "HIGH"
    if majority_score > MED_THRESHOLD:
        return "MED"
    return "LOW"


def load_tree_index(tree_path: Path) -> dict[tuple, dict]:
    """
    Load leaf_assignments.jsonl into a dict keyed by (chunk_id, topic_name).
    chunk_id is per-topic sequential (not globally unique), so the composite
    key is required for a correct join.
    """
    print(f"Loading tree index from {tree_path} ...")
    index = {}
    with open(tree_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["chunk_id"], rec["topic_name"])
            index[key] = rec
    print(f"  Tree index loaded: {len(index):,} entries")
    return index


def main() -> None:
    tree_index = load_tree_index(TREE_FILE)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    missing   = 0
    written   = 0
    tier_counts = {"HIGH": 0, "MED": 0, "LOW": 0}

    print(f"Joining chunks from {CHUNKS_FILE} ...")
    with open(CHUNKS_FILE, encoding="utf-8") as fin, \
         open(OUT_FILE,    "w", encoding="utf-8") as fout:

        gidx = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            cid   = chunk["chunk_id"]

            leaf  = tree_index.get((cid, chunk["topic_name"]))
            if leaf is None:
                missing += 1
                continue

            tier = assign_tier(leaf["majority_score"])
            tier_counts[tier] += 1

            record = {
                # original chunk fields
                "_gidx":               gidx,   # globally unique line index (0 … N-1)
                "chunk_id":            cid,
                "article_id":          chunk["article_id"],
                "chunk_idx":           chunk["chunk_idx"],
                "topic_name":          chunk["topic_name"],
                "topic_slug":          chunk.get("topic_slug", ""),
                "title":               chunk.get("title", ""),
                "text":                chunk["text"],
                "token_count":         chunk.get("token_count", 0),
                # tree fields
                "leaf_id":             leaf["leaf_id"],
                "label_path":          leaf["label_path"],
                "majority_score":      leaf["majority_score"],
                "noise_tier":          tier,
            }
            fout.write(json.dumps(record) + "\n")
            written += 1
            gidx    += 1

            if written % 50_000 == 0:
                print(f"  {written:,} written ...")

    print()
    print("Done.")
    print(f"  Written : {written:,}")
    print(f"  Missing : {missing:,}  (no tree entry — skipped)")
    print(f"  Tier breakdown:")
    for t, n in sorted(tier_counts.items()):
        print(f"    {t}: {n:,}  ({100*n/written:.1f}%)")
    print(f"  Output  : {OUT_FILE}")


if __name__ == "__main__":
    main()
