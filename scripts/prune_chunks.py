"""
scripts/prune_chunks.py
Chunk Deduplication & Pruning Pipeline (Stage 1.5)
Optimised for Apple M4 Pro — 12 cores, 24 GB unified memory, MPS, Accelerate BLAS.

Reduces 335,882 Wikipedia chunks to ~60-80k high-quality, diversity-maximising
chunks via 5 sequential layers before Stage 2 undersampling.

NOTE: chunk_id resets per-topic. True unique key = (_gidx) = global line index.

Usage:
    python scripts/prune_chunks.py [--resume] [--config config.yaml]
"""

import argparse
import collections
import gc
import hashlib
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

# Force offline mode so HF Hub doesn't attempt network calls
# (model is already cached locally from previous runs)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
from datasketch import MinHash, MinHashLSH
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm
# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE        = ROOT / "data/chunks/chunks.jsonl"
LEAF_ASSIGNMENTS   = ROOT / "data/tree/leaf_assignments.jsonl"
LEAF_NODES_FILE    = ROOT / "data/tree/leaf_nodes.json"
PRUNED_DIR         = ROOT / "data/pruned"
CKPT_DIR           = PRUNED_DIR / "checkpoints"

L1_CKPT    = CKPT_DIR / "l1_survivors.jsonl"
L2_CKPT    = CKPT_DIR / "l2_survivors.jsonl"
L3_CKPT    = CKPT_DIR / "l3_survivors.jsonl"
L4_CKPT    = CKPT_DIR / "l4_survivors.jsonl"
EMBED_NPY  = CKPT_DIR / "embeddings.npy"
EMBED_IDX  = CKPT_DIR / "gidx_index.json"
FINAL_OUT  = PRUNED_DIR / "pruned_chunks.jsonl"
REPORT_FILE= PRUNED_DIR / "pruning_report.json"

RANDOM_SEED = 42
N_WORKERS   = 8      # M4 Pro performance cores
np.random.seed(RANDOM_SEED)

# ── Micro-minority topic protection ───────────────────────────────────────────
# Topics with fewer than PROTECT_THRESHOLD L4-survivor chunks are fully
# preserved in L5 (all their chunks bypass the greedy-diversity cap).
# This prevents the per-leaf target from erasing already-thin minority signals.
PROTECT_THRESHOLD = 1_000_000   # effectively protects ALL minority topics unconditionally
MINORITY_TOPICS = frozenset({
    "Feminism", "Hinduism", "Buddhism", "Islam", "Colonialism", "Slavery",
    "Apartheid", "Suffrage", "Matriarchy", "Yoruba", "Swahili", "Aztec",
    "Quechua", "Malayalam", "Confucianism", "Shintoism", "Zoroastrianism",
    "Aboriginals", "Dalit", "LGBT",
})


# ── I/O helpers ───────────────────────────────────────────────────────────────

def stream_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def stream_chunks_with_gidx(path):
    """Inject _gidx = global 0-based line index into each chunk."""
    with open(path, encoding="utf-8") as f:
        for gidx, line in enumerate(f):
            s = line.strip()
            if s:
                rec = json.loads(s)
                rec["_gidx"] = gidx
                yield rec


def write_jsonl(records, path, show_progress=False, desc=None):
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        label = desc or Path(path).name
        it = tqdm(records, desc=f"Writing {label}") if show_progress else records
        for rec in it:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def count_jsonl(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def write_stats(stats, path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


# ── Layer 1: Exact deduplication ─────────────────────────────────────────────

def run_layer1(source):
    """MD5 of lowercased, whitespace-collapsed text. Streaming."""
    seen = set()
    n_in = n_rm = 0
    for chunk in source:
        n_in += 1
        norm = " ".join(chunk["text"].lower().split())
        h = hashlib.md5(norm.encode()).hexdigest()
        if h in seen:
            n_rm += 1
        else:
            seen.add(h)
            yield chunk
    stats = dict(input_count=n_in, output_count=n_in-n_rm,
                 removed_count=n_rm, removal_rate=round(n_rm/max(n_in,1),4))
    write_stats(stats, CKPT_DIR / "l1_stats.json")
    logger.info(f"L1 ✓ — in:{n_in:,} out:{n_in-n_rm:,} removed:{n_rm:,} ({n_rm/max(n_in,1)*100:.1f}%)")


# ── Layer 2: MinHash LSH ──────────────────────────────────────────────────────

NUM_PERM          = 128
MINHASH_THRESHOLD = 0.85
L2_BATCH          = 5_000


def _compute_minhash(text: str) -> bytes:
    """Compute a MinHash and return it serialised (for multiprocessing)."""
    words = text.lower().split()
    m = MinHash(num_perm=NUM_PERM, seed=RANDOM_SEED)
    for i in range(max(0, len(words) - 2)):
        m.update(" ".join(words[i:i+3]).encode())
    return m.hashvalues.tobytes()          # serialise as raw bytes


def _deserialise_minhash(raw: bytes) -> MinHash:
    m = MinHash(num_perm=NUM_PERM, seed=RANDOM_SEED)
    m.hashvalues = np.frombuffer(raw, dtype=np.uint64).copy()
    return m


def run_layer2(source):
    """
    Parallel MinHash computation (ProcessPoolExecutor, 8 workers).
    Global LSH index — all inserts into one LSH to detect cross-batch dups.
    """
    lsh          = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=NUM_PERM)
    inserted     = set()   # _gidx values in LSH
    to_remove    = set()   # _gidx values marked duplicate
    n_in         = 0

    def process_batch(batch):
        texts = [c["text"] for c in batch]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            raw_list = list(ex.map(_compute_minhash, texts, chunksize=200))
        minhashes = [_deserialise_minhash(r) for r in raw_list]

        for chunk, mh in zip(batch, minhashes):
            gidx = chunk["_gidx"]
            if gidx in to_remove:
                continue
            for key in lsh.query(mh):
                existing = int(key)
                if existing == gidx:
                    continue
                if existing > gidx:
                    to_remove.add(existing)
                else:
                    to_remove.add(gidx)
                    break
            if gidx not in to_remove and gidx not in inserted:
                lsh.insert(str(gidx), mh)
                inserted.add(gidx)

    logger.info("L2 — Building global MinHash LSH (parallel hashing, 8 workers)…")
    batch = []
    for chunk in tqdm(source, desc="L2 MinHash"):
        n_in += 1
        batch.append(chunk)
        if len(batch) >= L2_BATCH:
            process_batch(batch)
            batch = []
    if batch:
        process_batch(batch)

    n_rm = len(to_remove)
    write_stats(dict(input_count=n_in, output_count=n_in-n_rm,
                     removed_count=n_rm, removal_rate=round(n_rm/max(n_in,1),4)),
                CKPT_DIR/"l2_stats.json")
    logger.info(f"L2 ✓ — in:{n_in:,} out:{n_in-n_rm:,} removed:{n_rm:,} ({n_rm/max(n_in,1)*100:.1f}%)")
    return to_remove


def stream_l2_survivors(source, to_remove):
    for c in source:
        if c["_gidx"] not in to_remove:
            yield c


# ── Layer 3: TF-IDF cosine dedup (sparse + vectorised) ───────────────────────

TFIDF_THRESHOLD  = 0.85
L3_PROJ_DIMS     = 256      # JL projection dims (dense, Accelerate-friendly)
L3_MATMUL_BATCH  = 1_000    # rows per matmul: (1000, 256) @ (256, 335k) ≈ 1.3 GB


def run_layer3(source):
    """
    TF-IDF cosine dedup via Random Projection + batched numpy matmul.

    Why: naive sparse×sparse product is ~76% dense for this corpus.
    FAISS has thread-pool conflicts after ProcessPoolExecutor (L2 on macOS).

    Strategy (fast, correct):
      1. Fit TF-IDF (50k features, sparse CSR) — ~3 min
      2. Sparse random-project to 256 dense dims (Johnson-Lindenstrauss)  — ~6 s
      3. L2-normalise → (n, 256) float32 in RAM (335k × 256 × 4 = 344 MB)
      4. Batched dense matmul: (1000, 256) @ (256, 335k) — Accelerate BLAS
      5. np.where upper-triangle > 0.85 → mark duplicates
    Total expected: ~10 min on M4 Pro.
    """
    logger.info("L3 — Loading L2 survivors…")
    chunks   = list(tqdm(source, desc="L3 load"))
    n_in     = len(chunks)
    gidx_arr = np.array([c["_gidx"] for c in chunks], dtype=np.int64)
    texts    = [c["text"] for c in chunks]

    logger.info(f"L3 — Fitting TF-IDF on {n_in:,} chunks…")
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1,2),
                          min_df=2, sublinear_tf=True)
    tfidf = vec.fit_transform(texts)
    logger.info(f"L3 — TF-IDF shape: {tfidf.shape}  nnz: {tfidf.nnz:,}")
    del vec, texts

    tfidf_norm = sk_normalize(tfidf, norm="l2", copy=False)
    del tfidf

    logger.info(f"L3 — Sparse random projection → {L3_PROJ_DIMS} dense dims…")
    from sklearn.random_projection import SparseRandomProjection
    proj = SparseRandomProjection(n_components=L3_PROJ_DIMS, random_state=RANDOM_SEED)
    P = proj.fit_transform(tfidf_norm)          # (n, 256)  may be sparse
    del tfidf_norm, proj
    gc.collect()

    if hasattr(P, "toarray"):
        P = P.toarray()
    P = P.astype(np.float32)

    # L2-normalise rows → inner product == cosine similarity
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P /= np.where(norms == 0, 1.0, norms)
    P = np.ascontiguousarray(P)          # ensure Accelerate-friendly layout
    PT = np.ascontiguousarray(P.T)       # (256, n) — reused across batches

    logger.info(f"L3 — Projected matrix: {P.shape}  ({P.nbytes/1e9:.2f} GB)")
    logger.info(f"L3 — Batched matmul (batch={L3_MATMUL_BATCH})…")

    to_remove_set = set()
    n_batches = (n_in + L3_MATMUL_BATCH - 1) // L3_MATMUL_BATCH

    for b_idx in tqdm(range(n_batches), desc="L3 matmul"):
        bs = b_idx * L3_MATMUL_BATCH
        be = min(bs + L3_MATMUL_BATCH, n_in)
        # Dense matmul — Apple Accelerate BLAS via NumPy
        sims = P[bs:be] @ PT                          # (batch, n) float32
        # Mask: only upper-triangle (j > global_i) and > threshold
        for local_i in range(be - bs):
            global_i = bs + local_i
            if gidx_arr[global_i] in to_remove_set:
                continue
            row = sims[local_i]
            # Zero out j <= global_i so we only mark higher-index dups
            row[:global_i + 1] = 0.0
            dup_js = np.where(row > TFIDF_THRESHOLD)[0]
            for j in dup_js:
                to_remove_set.add(int(gidx_arr[j]))
        del sims

    del P, PT
    gc.collect()

    n_rm = len(to_remove_set)
    write_stats(dict(input_count=n_in, output_count=n_in-n_rm,
                     removed_count=n_rm, removal_rate=round(n_rm/max(n_in,1),4)),
                CKPT_DIR/"l3_stats.json")
    logger.info(f"L3 ✓ — in:{n_in:,} out:{n_in-n_rm:,} removed:{n_rm:,} ({n_rm/max(n_in,1)*100:.1f}%)")

    for c in chunks:
        if c["_gidx"] not in to_remove_set:
            yield c


# ── Layer 4: Semantic cosine dedup (MPS + FAISS) ─────────────────────────────

SEMANTIC_THRESHOLD = 0.92
L4_ENCODE_BATCH    = 1024          # M4 Pro unified memory handles large batches
L4_FAISS_BATCH     = 2_000


def run_layer4(source):
    """
    Semantic cosine dedup via batched numpy matmul (Apple Accelerate BLAS).

    Replaces FAISS: FAISS's OpenMP pool segfaults on macOS after the
    tokenizers Rust thread pool (used by sentence-transformers) leaves
    live threads. Numpy matmul is equally fast here and crash-free.

    Strategy:
      1. Encode with MPS (or CPU) → save embeddings.npy + gidx_index.json
         (skipped if both files already exist from a previous run)
      2. Batched dense matmul: (L4_FAISS_BATCH, d) @ (d, n) → (batch, n)
         using Apple Accelerate BLAS
      3. np.where(row > SEMANTIC_THRESHOLD) in upper-triangle → mark dups
    """
    import torch

    logger.info("L4 — Loading L3 survivors…")
    chunks   = list(tqdm(source, desc="L4 load"))
    n_in     = len(chunks)
    gidx_arr = [c["_gidx"] for c in chunks]

    # ── Encode (skip if embeddings already saved) ─────────────────────────
    if EMBED_NPY.exists() and EMBED_IDX.exists() and EMBED_NPY.stat().st_size > 0:
        logger.info("L4 — Embeddings found on disk — skipping encoding…")
        saved_gidx = json.loads(EMBED_IDX.read_text())
        if saved_gidx == gidx_arr:
            embs = np.load(str(EMBED_NPY))
        else:
            # gidx mismatch (different L3 run) — re-encode
            logger.info("L4 — gidx mismatch; re-encoding…")
            EMBED_NPY.unlink(missing_ok=True)
            EMBED_IDX.unlink(missing_ok=True)
            embs = None
    else:
        embs = None

    if embs is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"L4 — Encoding device: {device}")
        texts = [c["text"] for c in chunks]
        logger.info(f"L4 — Encoding {n_in:,} chunks (batch={L4_ENCODE_BATCH})…")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                    device=device, local_files_only=True)
        embs  = model.encode(texts, batch_size=L4_ENCODE_BATCH,
                              show_progress_bar=True, convert_to_numpy=True)
        del model, texts
        gc.collect()

        # L2-normalise → inner product == cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs  = (embs / np.where(norms==0, 1, norms)).astype(np.float32)

        np.save(str(EMBED_NPY), embs)
        with open(EMBED_IDX, "w") as f:
            json.dump(gidx_arr, f)
        logger.info(f"L4 — Embeddings saved → {EMBED_NPY}")

    embs = np.ascontiguousarray(embs.astype(np.float32))
    ET   = np.ascontiguousarray(embs.T)    # (d, n) — reused across batches

    logger.info(f"L4 — Batched cosine matmul (batch={L4_FAISS_BATCH}, threshold={SEMANTIC_THRESHOLD})…")
    to_remove = set()
    n_batches = (n_in + L4_FAISS_BATCH - 1) // L4_FAISS_BATCH

    for b_idx in tqdm(range(n_batches), desc="L4 cosine"):
        bs = b_idx * L4_FAISS_BATCH
        be = min(bs + L4_FAISS_BATCH, n_in)
        sims = embs[bs:be] @ ET              # (batch, n) cosine sims
        for local_i in range(be - bs):
            global_i = bs + local_i
            if gidx_arr[global_i] in to_remove:
                continue
            row = sims[local_i]
            row[:global_i + 1] = 0.0        # upper-triangle only
            dup_js = np.where(row > SEMANTIC_THRESHOLD)[0]
            for j in dup_js:
                to_remove.add(gidx_arr[j])
        del sims

    del ET
    gc.collect()

    n_rm = len(to_remove)
    write_stats(dict(input_count=n_in, output_count=n_in-n_rm,
                     removed_count=n_rm, removal_rate=round(n_rm/max(n_in,1),4)),
                CKPT_DIR/"l4_stats.json")
    logger.info(f"L4 ✓ — in:{n_in:,} out:{n_in-n_rm:,} removed:{n_rm:,} ({n_rm/max(n_in,1)*100:.1f}%)")

    gidx_to_i = {g: i for i, g in enumerate(gidx_arr)}
    for chunk in chunks:
        if chunk["_gidx"] not in to_remove:
            yield chunk, embs[gidx_to_i[chunk["_gidx"]]]


# ── Layer 5: Leaf-local diversity subsampling (parallel) ─────────────────────

def _noise_tier(score: float) -> str:
    if score > 1.5:  return "HIGH"
    if score >= 0.5: return "MED"
    return "LOW"


def _greedy_diverse(emb_list: list, target: int) -> list:
    """Greedy max-diversity selection. Vectorised with numpy."""
    n = len(emb_list)
    if n <= target:
        return list(range(n))
    E = np.stack(emb_list).astype(np.float32)   # (n, d)
    S = E @ E.T                                  # cosine (already L2-normed)
    # Seed: most isolated chunk
    selected   = [int(np.argmin(S.mean(axis=1)))]
    unsel_mask = np.ones(n, dtype=bool)
    unsel_mask[selected[0]] = False
    # max-sim of each unselected chunk to any selected chunk
    max_sim_to_sel = S[:, selected[0]].copy()

    while len(selected) < target:
        # Among unselected, pick chunk with minimum max-sim to selected set
        candidates = np.where(unsel_mask)[0]
        best_local = int(np.argmin(max_sim_to_sel[candidates]))
        best       = candidates[best_local]
        selected.append(int(best))
        unsel_mask[best] = False
        # Update max-sim incrementally (no full recompute)
        max_sim_to_sel = np.maximum(max_sim_to_sel, S[:, best])
    return selected


def _process_leaf_worker(args):
    """Top-level function (picklable) for ProcessPoolExecutor."""
    lid, chunks_data, emb_arrays, target, score, composite_to_leaf, protected_topics = args

    # Split into must-keep (protected micro-minority) and normal chunks
    protected_idx   = [i for i, c in enumerate(chunks_data)
                       if c["topic_name"] in protected_topics]
    unprotected_idx = [i for i, c in enumerate(chunks_data)
                       if c["topic_name"] not in protected_topics]

    # Fill remaining slots (beyond protected) with greedy-diverse unprotected chunks
    extra = max(0, target - len(protected_idx))
    if extra > 0 and unprotected_idx:
        unp_embs  = [emb_arrays[i] for i in unprotected_idx]
        sel_local = _greedy_diverse(unp_embs, extra)
        selected_indices = protected_idx + [unprotected_idx[i] for i in sel_local]
    else:
        selected_indices = protected_idx if protected_idx else _greedy_diverse(emb_arrays, target)

    tier = _noise_tier(score)
    out = []
    for idx in selected_indices:
        chunk = chunks_data[idx]
        leaf_rec = composite_to_leaf.get((chunk["chunk_id"], chunk["topic_name"]), {})
        enriched = dict(chunk)
        enriched["leaf_id"]                  = lid
        enriched["label_path"]               = leaf_rec.get("label_path", "")
        enriched["majority_score"]           = leaf_rec.get("majority_score", score)
        enriched["noise_tier"]               = tier
        enriched["pruning_layers_survived"]  = ["L1","L2","L3","L4","L5"]
        out.append(enriched)
    leaf_stat = dict(input_count=len(chunks_data), output_count=len(out),
                     target_size=target, majority_score=score)
    return lid, out, leaf_stat


def run_layer5(source_with_embeddings, leaf_nodes):
    # Tier targets
    counts = [v["document_count"] for v in leaf_nodes.values()]
    median = statistics.median(counts)
    high_t = int(min(500, max(100, median * 3)))
    med_t  = int(min(200, max(50,  median * 2)))
    low_t  = int(min(50,  max(10,  median)))
    logger.info(f"L5 targets — median:{median:.0f}  HIGH:{high_t}  MED:{med_t}  LOW:{low_t}")

    # Build composite-key → leaf map
    logger.info("L5 — Building leaf assignment map…")
    comp_to_leaf = {}
    for rec in tqdm(stream_jsonl(LEAF_ASSIGNMENTS), desc="L5 leaf map"):
        comp_to_leaf[(rec["chunk_id"], rec["topic_name"])] = rec

    # Group by leaf
    logger.info("L5 — Grouping L4 survivors by leaf…")
    leaf_chunks = {}   # lid -> [chunk_dict]
    leaf_embs   = {}   # lid -> [np.array]
    missing = 0

    for chunk, emb in tqdm(source_with_embeddings, desc="L5 group"):
        key      = (chunk["chunk_id"], chunk["topic_name"])
        leaf_rec = comp_to_leaf.get(key)
        if leaf_rec is None:
            missing += 1
            continue
        lid = leaf_rec["leaf_id"]
        leaf_chunks.setdefault(lid, []).append(chunk)
        leaf_embs.setdefault(lid, []).append(emb)

    if missing:
        logger.warning(f"L5 — {missing:,} chunks had no leaf assignment (dropped)")

    n_in = sum(len(v) for v in leaf_chunks.values())
    logger.info(f"L5 — {n_in:,} chunks across {len(leaf_chunks):,} leaves")

    # Identify micro-minority topics to fully protect
    topic_counts = collections.Counter(
        c["topic_name"]
        for chunks in leaf_chunks.values()
        for c in chunks
    )
    protected_topics = frozenset(
        t for t in MINORITY_TOPICS
        if topic_counts.get(t, 0) < PROTECT_THRESHOLD
    )
    if protected_topics:
        details = ", ".join(f"{t}={topic_counts.get(t,0):,}" for t in sorted(protected_topics))
        logger.info(f"L5 — Fully protecting {len(protected_topics)} micro-minority topics: {details}")

    def leaf_score(lid):
        return leaf_nodes.get(lid, {}).get("majority_score", 0.0)

    sorted_lids = sorted(leaf_chunks.keys(), key=leaf_score, reverse=True)

    # Build worker args
    worker_args = []
    for lid in sorted_lids:
        score = leaf_score(lid)
        tier  = _noise_tier(score)
        target = high_t if tier=="HIGH" else med_t if tier=="MED" else low_t
        worker_args.append((lid, leaf_chunks[lid], leaf_embs[lid],
                            target, score, comp_to_leaf, protected_topics))

    # Parallel leaf processing (8 workers, chunked for memory efficiency)
    logger.info(f"L5 — Processing {len(worker_args):,} leaves with {N_WORKERS} workers…")
    output_chunks = []
    leaf_stats    = {}
    LEAF_CHUNK    = 500   # submit this many leaves per executor batch

    for batch_start in tqdm(range(0, len(worker_args), LEAF_CHUNK),
                             desc="L5 diversity"):
        batch = worker_args[batch_start: batch_start + LEAF_CHUNK]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            for lid, out_chunks, stat in ex.map(_process_leaf_worker, batch,
                                                 chunksize=10):
                output_chunks.extend(out_chunks)
                leaf_stats[lid] = stat
                if len(out_chunks) == 0:
                    logger.warning(f"L5 — Leaf {lid} has zero survivors!")

    write_stats(leaf_stats, CKPT_DIR / "l5_leaf_stats.json")

    n_out = len(output_chunks)
    n_rm  = n_in - n_out
    write_stats(dict(input_count=n_in, output_count=n_out,
                     removed_count=n_rm, removal_rate=round(n_rm/max(n_in,1),4)),
                CKPT_DIR/"l5_stats.json")
    logger.info(f"L5 ✓ — in:{n_in:,} out:{n_out:,} removed:{n_rm:,} ({n_rm/max(n_in,1)*100:.1f}%)")
    return output_chunks


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(leaf_nodes):
    logger.info("=== VALIDATION ===")
    comp_to_score = {(r["chunk_id"], r["topic_name"]): r["majority_score"]
                     for r in stream_jsonl(LEAF_ASSIGNMENTS)}
    all_lids    = set(leaf_nodes)
    gidx_seen   = set()
    lid_seen    = set()
    tier_cnt    = {"HIGH": 0, "MED": 0, "LOW": 0}
    empty_text  = score_mm = total = 0

    for rec in stream_jsonl(FINAL_OUT):
        total += 1
        gidx_seen.add(rec.get("_gidx"))
        lid = rec.get("leaf_id","")
        lid_seen.add(lid)
        t = rec.get("noise_tier","")
        if t in tier_cnt: tier_cnt[t] += 1
        if not rec.get("text","").strip(): empty_text += 1
        key = (rec.get("chunk_id"), rec.get("topic_name"))
        exp = comp_to_score.get(key)
        act = rec.get("majority_score")
        if exp is not None and act is not None and abs(exp-act) > 1e-6:
            score_mm += 1

    passes, fails = [], []
    def chk(name, ok, detail=""):
        (passes if ok else fails).append(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")

    chk("Total 50k–100k",          50_000 <= total <= 100_000, f"({total:,})")
    chk("No dup _gidx",            len(gidx_seen)==total, f"({len(gidx_seen):,}/{total:,})")
    invalid = lid_seen - all_lids
    chk("leaf_ids valid",          not invalid,          f"({len(invalid)} invalid)")
    chk("HIGH chunks exist",       tier_cnt["HIGH"] > 0, f"({tier_cnt['HIGH']:,})")
    chk("MED chunks exist",        tier_cnt["MED"]  > 0, f"({tier_cnt['MED']:,})")
    chk("LOW chunks exist",        tier_cnt["LOW"]  > 0, f"({tier_cnt['LOW']:,})")
    high_lids = {l for l,v in leaf_nodes.items() if v.get("majority_score",0)>1.5}
    chk("All HIGH leaves present", high_lids<=lid_seen, f"({len(high_lids&lid_seen)}/{len(high_lids)})")
    low_lids  = {l for l,v in leaf_nodes.items() if v.get("majority_score",0)<0.5}
    pct = len(low_lids & lid_seen) / max(len(low_lids),1)
    chk("≥80% LOW leaves retained", pct>=0.80, f"({pct*100:.1f}%)")
    chk("No empty text",           empty_text==0, f"({empty_text})")
    chk("Score monotonicity",      score_mm==0,   f"({score_mm} mismatches)")

    print("\n=== VALIDATION REPORT ===")
    for m in passes: print(m)
    for m in fails:  print(m)
    print(f"\n{len(passes)} passed, {len(fails)} failed")
    return len(fails) == 0


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(leaf_nodes):
    def ld(p): return json.loads(p.read_text()) if p.exists() else {}
    layers = {
        "l1_exact":    ld(CKPT_DIR/"l1_stats.json"),
        "l2_minhash":  ld(CKPT_DIR/"l2_stats.json"),
        "l3_tfidf":    ld(CKPT_DIR/"l3_stats.json"),
        "l4_semantic": ld(CKPT_DIR/"l4_stats.json"),
        "l5_leaf":     ld(CKPT_DIR/"l5_stats.json"),
    }
    total_out   = layers.get("l5_leaf",{}).get("output_count",0)
    tier_dist   = {t:{"leaves":0,"chunks":0} for t in ("HIGH","MED","LOW")}
    leaf_counts = {}
    counted     = set()
    for rec in stream_jsonl(FINAL_OUT):
        lid = rec.get("leaf_id","")
        t   = rec.get("noise_tier","")
        if t in tier_dist: tier_dist[t]["chunks"] += 1
        leaf_counts[lid] = leaf_counts.get(lid,0) + 1
    for lid in leaf_counts:
        score = leaf_nodes.get(lid,{}).get("majority_score",0)
        t = _noise_tier(score)
        if lid not in counted:
            tier_dist[t]["leaves"] += 1
            counted.add(lid)
    top10_src = sorted([(l,v["majority_score"],v["document_count"])
                        for l,v in leaf_nodes.items()],
                       key=lambda x:x[1], reverse=True)[:10]
    top10 = [{"leaf_id":l,"majority_score":s,"chunks_before":b,
              "chunks_after":leaf_counts.get(l,0)} for l,s,b in top10_src]
    report = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "layers": layers,
        "total": {"input":335882,"output":total_out,
                  "removed":335882-total_out,
                  "total_rate":round((335882-total_out)/335882,4)},
        "noise_tier_distribution": tier_dist,
        "top10_leaves_after_pruning": top10,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    logger.info(f"Report → {REPORT_FILE}")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(ROOT/"logs"/"prune_chunks.log", rotation="100 MB")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PRUNED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Chunk Deduplication & Pruning Pipeline (M4 Pro optimised) ===")
    logger.info(f"Resume: {args.resume}  Workers: {N_WORKERS}")

    with open(LEAF_NODES_FILE) as f:
        leaf_nodes = json.load(f)
    logger.info(f"Loaded {len(leaf_nodes):,} leaf nodes")

    def ckpt_valid(p): return p.exists() and p.stat().st_size > 0

    # ── L1 ────────────────────────────────────────────────────────────────────
    if args.resume and ckpt_valid(L1_CKPT):
        logger.info(f"L1 ✓ checkpoint — {count_jsonl(L1_CKPT):,} survivors")
    else:
        write_jsonl(run_layer1(stream_chunks_with_gidx(CHUNKS_FILE)),
                    L1_CKPT, show_progress=True, desc="L1")

    # ── L2 ────────────────────────────────────────────────────────────────────
    if args.resume and ckpt_valid(L2_CKPT):
        logger.info(f"L2 ✓ checkpoint — {count_jsonl(L2_CKPT):,} survivors")
    else:
        to_rm2 = run_layer2(stream_jsonl(L1_CKPT))
        write_jsonl(stream_l2_survivors(stream_jsonl(L1_CKPT), to_rm2),
                    L2_CKPT, show_progress=True, desc="L2")

    # ── L3 ────────────────────────────────────────────────────────────────────
    if args.resume and ckpt_valid(L3_CKPT):
        logger.info(f"L3 ✓ checkpoint — {count_jsonl(L3_CKPT):,} survivors")
    else:
        write_jsonl(run_layer3(stream_jsonl(L2_CKPT)),
                    L3_CKPT, show_progress=True, desc="L3")

    # ── L4 ────────────────────────────────────────────────────────────────────
    if args.resume and ckpt_valid(L4_CKPT) and EMBED_NPY.exists() and EMBED_IDX.exists():
        logger.info("L4 ✓ checkpoint — loading embeddings from disk…")
        gidx_list = json.loads(EMBED_IDX.read_text())
        embs      = np.load(str(EMBED_NPY))
        g2e       = {g: embs[i] for i,g in enumerate(gidx_list)}
        def l4_stream():
            for c in stream_jsonl(L4_CKPT):
                e = g2e.get(c["_gidx"])
                if e is not None: yield c, e
        l4_source = l4_stream()
    else:
        l4_pairs  = list(run_layer4(stream_jsonl(L3_CKPT)))
        write_jsonl((p[0] for p in l4_pairs), L4_CKPT, show_progress=True, desc="L4")
        l4_source = iter(l4_pairs)

    # ── L5 ────────────────────────────────────────────────────────────────────
    if args.resume and ckpt_valid(FINAL_OUT):
        logger.info(f"L5 ✓ final output — {count_jsonl(FINAL_OUT):,} chunks")
    else:
        final = run_layer5(l4_source, leaf_nodes)
        write_jsonl(iter(final), FINAL_OUT, show_progress=True, desc="final")

    # ── Report & Validation ───────────────────────────────────────────────────
    report = build_report(leaf_nodes)
    t = report["total"]
    logger.info(f"\n{'='*50}")
    logger.info(f"  Input:   {t['input']:>10,}")
    logger.info(f"  Output:  {t['output']:>10,}")
    logger.info(f"  Removed: {t['removed']:>10,}  ({t['total_rate']*100:.1f}%)")
    for tier, d in report["noise_tier_distribution"].items():
        logger.info(f"  {tier:4s}: {d['chunks']:>8,} chunks  {d['leaves']:>6,} leaves")
    logger.info(f"{'='*50}")

    run_validation(leaf_nodes)


if __name__ == "__main__":
    main()
