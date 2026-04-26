"""
relabel_noise_tiers.py
======================
Recomputes noise_tier in noised_chunks_hpc.jsonl using two-group z-scoring:
  - T0 leaves scored against T0 population (mu/sigma of T0 leaf counts)
  - All other leaves scored against non-T0 population

Adds field `noise_tier_corrected` and `majority_score_corrected`.
Does NOT touch original_text / noised_text / instruction.

Usage:
    python relabel_noise_tiers.py
    python relabel_noise_tiers.py --input path/to/file.jsonl --output path/to/out.jsonl
"""

import argparse
import json
import statistics
from pathlib import Path

LEAF_NODES = Path("data/tree/leaf_nodes.json")
DEFAULT_IN  = Path("noised_chunks_hpc.jsonl")
DEFAULT_OUT = Path("noised_chunks_hpc_relabeled.jsonl")

HIGH_T = 1.5
MED_T  = 0.5


def assign_tier(score: float) -> str:
    if score > HIGH_T:
        return "HIGH"
    if score > MED_T:
        return "MED"
    return "LOW"


def build_leaf_score_map(leaf_nodes_path: Path) -> dict[str, float]:
    """
    Compute two-group corrected majority scores for every leaf_id.
    T0 leaves are normalised within the T0 population.
    All other leaves are normalised within the non-T0 population.
    """
    with open(leaf_nodes_path) as f:
        leaves = json.load(f)

    t0     = {k: v for k, v in leaves.items() if k.startswith("ROOT_T0_")}
    non_t0 = {k: v for k, v in leaves.items() if not k.startswith("ROOT_T0_")}

    t0_counts  = [v["document_count"] for v in t0.values()]
    nt0_counts = [v["document_count"] for v in non_t0.values()]

    t0_mu,  t0_sig  = statistics.mean(t0_counts),  statistics.stdev(t0_counts)
    nt0_mu, nt0_sig = statistics.mean(nt0_counts), statistics.stdev(nt0_counts)

    print(f"T0  group : n={len(t0_counts):,}  mu={t0_mu:.2f}  sigma={t0_sig:.2f}")
    print(f"non-T0    : n={len(nt0_counts):,}  mu={nt0_mu:.4f}  sigma={nt0_sig:.4f}")
    print(f"non-T0 HIGH threshold : ≥{nt0_mu + HIGH_T * nt0_sig:.1f} docs/leaf")
    print(f"non-T0 MED  threshold : ≥{nt0_mu + MED_T  * nt0_sig:.1f} docs/leaf")
    print()

    score_map: dict[str, float] = {}

    for lid, info in t0.items():
        score_map[lid] = (info["document_count"] - t0_mu) / t0_sig

    for lid, info in non_t0.items():
        score_map[lid] = (info["document_count"] - nt0_mu) / nt0_sig

    return score_map


def relabel(input_path: Path, output_path: Path, score_map: dict[str, float]) -> dict:
    stats = {"HIGH": 0, "MED": 0, "LOW": 0, "missing_leaf": 0, "total": 0}

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            d = json.loads(line)
            lid = d.get("leaf_id", "")
            corrected_score = score_map.get(lid)

            if corrected_score is None:
                # leaf not in tree — keep original tier, flag it
                d["majority_score_corrected"] = d.get("majority_score", 0.0)
                d["noise_tier_corrected"]     = d.get("noise_tier", "LOW")
                stats["missing_leaf"] += 1
            else:
                tier = assign_tier(corrected_score)
                d["majority_score_corrected"] = round(corrected_score, 6)
                d["noise_tier_corrected"]     = tier
                stats[tier] += 1

            stats["total"] += 1
            fout.write(json.dumps(d) + "\n")

            if stats["total"] % 50_000 == 0:
                print(f"  {stats['total']:,} processed …")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(DEFAULT_IN))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading leaf nodes from {LEAF_NODES} …")
    score_map = build_leaf_score_map(LEAF_NODES)

    print(f"Relabeling {input_path} → {output_path} …")
    stats = relabel(input_path, output_path, score_map)

    total = stats["total"]
    print()
    print("Done.")
    print(f"  Total processed : {total:,}")
    print(f"  Missing leaf    : {stats['missing_leaf']:,}")
    print(f"  HIGH            : {stats['HIGH']:,}  ({100*stats['HIGH']/total:.1f}%)")
    print(f"  MED             : {stats['MED']:,}  ({100*stats['MED']/total:.1f}%)")
    print(f"  LOW             : {stats['LOW']:,}  ({100*stats['LOW']/total:.1f}%)")
    print(f"  Output          : {output_path}")


if __name__ == "__main__":
    main()
