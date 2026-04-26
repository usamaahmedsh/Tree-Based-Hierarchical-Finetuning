#!/usr/bin/env python3
"""
Stage 3 — Full 70B Noise Injection + Instruction Generation
============================================================
All 335k chunks go through Llama 3.1 70B AWQ INT4 — no NLTK, no 8B shortcut.
Noise *level* still scales with majority_score (HIGH / MED / LOW ops),
but the model is uniformly 70B for quality across every tier.

Tuned for: 1x NVIDIA A40 (46GB VRAM), 16 CPU cores, AWQ INT4 quantization.

Backends
  vllm     — local GPU, double-buffer prefetch pipeline (recommended for HPC)
  bedrock  — cloud API via ThreadPoolExecutor (fallback)

Usage
  # vLLM on HPC (recommended)
  python scripts/stage3_noise_injection.py --backend vllm

  # Bedrock fallback
  python scripts/stage3_noise_injection.py --backend bedrock

  # Dry run (no API / GPU calls)
  python scripts/stage3_noise_injection.py --dry-run

Prepare input first:
  python scripts/prepare_full_data.py
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import queue
import random
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

from loguru import logger
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data/full_chunks_with_tree.jsonl"
OUT_DIR    = ROOT / "data/stage3_full"
OUT_FILE   = OUT_DIR / "noised_chunks.jsonl"
CKPT_FILE  = OUT_DIR / "checkpoint.json"
LOG_FILE   = OUT_DIR / "stage3.log"

# ── Noise tier thresholds (must match prepare_full_data.py) ───────────────────
HIGH_THRESHOLD = 1.5
MED_THRESHOLD  = 0.5

# ── Bedrock model ─────────────────────────────────────────────────────────────
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL  = "us.meta.llama3-3-70b-instruct-v1:0"

P_IN  = 0.72 / 1_000_000
P_OUT = 0.72 / 1_000_000

# ── vLLM defaults — tuned for 1x A40 46GB + AWQ INT4 70B ─────────────────────
# AWQ 70B weights:      ~37 GB
# Activation overhead:  ~1.2 GB
# KV cache headroom:    ~2.5 GB  → supports ~4 concurrent sequences at 2048 tok
#
# Strategy: keep max_model_len low to maximise KV cache slots,
#           batch matches real concurrency, CPU workers capped to allocation.
DEFAULT_VLLM_MODEL      = "./models/llama-3.1-8b"
DEFAULT_VLLM_BATCH      = 230        # matches real GPU concurrency (~4 seqs)
DEFAULT_TENSOR_PARALLEL = 1        # single GPU
DEFAULT_CPU_WORKERS     = 14       # 16 allocated cores - 2 for GPU/writer threads
DEFAULT_PREFETCH        = 32        # batches pre-built ahead of GPU

# ── Bedrock defaults ──────────────────────────────────────────────────────────
DEFAULT_WORKERS = 12
DEFAULT_BATCH   = 96

# ── Shared generation settings ────────────────────────────────────────────────
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS  = 600   # lowered from 900 — improves throughput on memory-constrained GPU
CHECKPOINT_FREQ     = 50    # more frequent checkpoints given small batch size
MAX_RETRIES         = 6

# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# GPU auto-detect  (info-only — does NOT override hardcoded defaults)
# ──────────────────────────────────────────────────────────────────────────────

def detect_gpu() -> dict:
    """
    Detect available compute device and log info.
    Returns dtype only — batch/tensor_parallel are hardcoded for A40+AWQ safety.
    """
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(n)]
            total_vram = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(n)
            ) / 1024**3

            logger.info(f"GPU detected: {n}x [{', '.join(names)}]  "
                        f"total VRAM ≈ {total_vram:.0f} GB")
            logger.info(f"  Using: tensor_parallel=1, dtype=float16 (AWQ requirement)")
            return {"device": "cuda", "n_gpus": n, "dtype": "float16"}

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS detected")
            return {"device": "mps", "n_gpus": 1, "dtype": "float16"}

    except ImportError:
        pass

    logger.warning("No GPU detected — falling back to CPU. Use --dry-run for testing.")
    return {"device": "cpu", "n_gpus": 0, "dtype": "float32"}


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add(LOG_FILE, level="DEBUG", rotation="50 MB",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}")


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────────

def _readability(label_path: str) -> int:
    """Extract C1–C5 readability tier from label path string."""
    m = re.search(r">\s*C(\d)", label_path or "")
    return int(m.group(1)) if m else 3


def build_prompt(chunk: dict, temperature: float, max_tokens: int) -> dict:
    """
    Build a Bedrock / vLLM request body for any noise tier.
    All tiers go through the same 70B model; only the operations differ.
    """
    tier = chunk["noise_tier"]
    rdbl = _readability(chunk.get("label_path", ""))

    if tier == "HIGH":
        ops = (
            "1. Inject 2–3 off-topic sentences naturally within the text.\n"
            "2. Replace 3–4 named entities (people, places, organisations, dates) "
            "with plausible but different alternatives.\n"
            "3. Paraphrase 2–3 factual claims to be noticeably less precise.\n"
            "4. Insert exactly 1 contradictory statement.\n"
            "5. Sprinkle synonym replacements throughout."
        )
    elif tier == "MED":
        ops = (
            "1. Inject 1 off-topic sentence naturally within the text.\n"
            "2. Replace 1–2 named entities with plausible but different alternatives.\n"
            "3. Paraphrase 1 factual claim to be less precise.\n"
            "4. Sprinkle synonym replacements throughout."
        )
    else:  # LOW
        ops = (
            "1. Replace 1–2 named entities with plausible but different alternatives.\n"
            "2. Sprinkle a few light synonym replacements throughout.\n"
            "3. Keep all factual content intact."
        )

    simplify = ""
    if rdbl >= 4:
        simplify = (
            "\n4. LANGUAGE: Rewrite in simple, everyday conversational English. "
            "Use contractions, short sentences, plain vocabulary. "
            "Occasional minor grammatical imperfections are fine. "
            "Preserve all factual content (aside from operations above)."
        )

    system = (
        "You are a text transformation assistant. "
        "Apply EVERY numbered operation to the text the user provides. "
        "Return your response in EXACTLY this two-section format — no extra commentary:\n\n"
        "NOISED_TEXT:\n"
        "[your transformed text]\n\n"
        "INSTRUCTION:\n"
        "[a natural question or learning prompt about the topic of the transformed text — "
        "phrase it as if someone wants to understand the subject matter, "
        "NOT as a request to edit or transform anything]"
    )

    user = f"Apply these operations to the text below:\n{ops}{simplify}\n\nTEXT:\n{chunk['text']}"

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"NOISED_TEXT:\n"
    )

    return {
        "prompt":      prompt,
        "temperature": temperature,
        "max_gen_len": max_tokens,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_response(raw: str, original_text: str) -> tuple[str, str]:
    """
    Extract NOISED_TEXT and INSTRUCTION from the model response.
    The prompt is primed with 'NOISED_TEXT:' so raw starts with the noised text.
    """
    full = "NOISED_TEXT:\n" + raw

    nt_m = re.search(r"NOISED_TEXT:\s*(.*?)(?=\nINSTRUCTION:|\Z)", full, re.DOTALL)
    in_m = re.search(r"INSTRUCTION:\s*(.*)",                         full, re.DOTALL)

    noised      = nt_m.group(1).strip() if nt_m else original_text
    instruction = in_m.group(1).strip() if in_m else ""

    bad_phrases = ("rewrite", "transform", "replace", "paraphrase",
                   "inject", "synonym", "edit", "modify", "noise")
    if any(p in instruction.lower() for p in bad_phrases):
        instruction = re.split(r"[.?!]", instruction)[0].strip() + "?"
        logger.debug(f"Sanitised leaky instruction → {instruction!r}")

    return noised, instruction


# ──────────────────────────────────────────────────────────────────────────────
# CPU worker — builds prompt string only
# Must be module-level for ProcessPoolExecutor pickling.
# ──────────────────────────────────────────────────────────────────────────────

def _cpu_build_prompt(args_tuple: tuple) -> tuple:
    chunk, temperature, max_tokens = args_tuple
    body = build_prompt(chunk, temperature, max_tokens)
    return (chunk, body["prompt"])


# ──────────────────────────────────────────────────────────────────────────────
# Output record
# ──────────────────────────────────────────────────────────────────────────────

def _build_record(chunk, original, noised, instruction,
                  model_used, n_in, n_out, cost) -> dict:
    return {
        "_gidx":          chunk["_gidx"],
        "topic_name":     chunk.get("topic_name", ""),
        "leaf_id":        chunk.get("leaf_id", ""),
        "label_path":     chunk.get("label_path", ""),
        "majority_score": chunk.get("majority_score", 0.0),
        "noise_tier":     chunk.get("noise_tier", ""),
        "original_text":  original,
        "noised_text":    noised,
        "instruction":    instruction,
        "model_used":     model_used,
        "tokens_in":      n_in,
        "tokens_out":     n_out,
        "cost_usd":       cost,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bedrock helpers
# ──────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


def _get_client():
    if not hasattr(_thread_local, "client"):
        import boto3
        _thread_local.client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _thread_local.client


def bedrock_call(body: dict, model_id: str) -> tuple[str, int, int]:
    """Invoke Bedrock with exponential back-off. Returns (text, n_in, n_out)."""
    raw = json.dumps(body)
    from botocore.exceptions import ClientError
    for attempt in range(MAX_RETRIES):
        try:
            resp   = _get_client().invoke_model(
                body=raw, modelId=model_id,
                contentType="application/json", accept="application/json",
            )
            result = json.loads(resp["body"].read())
            text   = result.get("generation", "").strip()
            n_in   = result.get("prompt_token_count", 0)
            n_out  = result.get("generation_token_count", 0)
            return text, n_in, n_out
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "ServiceUnavailableException",
                        "ModelNotReadyException"):
                wait = (2 ** attempt) * 0.5 + random.uniform(0, 0.5)
                logger.warning(f"Bedrock {code} — retry {attempt+1}/{MAX_RETRIES} in {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Bedrock call failed after {MAX_RETRIES} retries")


def process_chunk(chunk: dict, dry_run: bool,
                  temperature: float, max_tokens: int) -> dict:
    """Process a single chunk via Bedrock (all tiers → same 70B model)."""
    if dry_run:
        return _build_record(chunk, chunk["text"], f"[DRY-RUN] {chunk['text'][:60]}",
                             "What is this about?", "DRY_RUN", 0, 0, 0.0)

    body             = build_prompt(chunk, temperature, max_tokens)
    raw, n_in, n_out = bedrock_call(body, BEDROCK_MODEL)
    noised, instr    = parse_response(raw, chunk["text"])
    cost             = n_in * P_IN + n_out * P_OUT

    return _build_record(chunk, chunk["text"], noised, instr,
                         BEDROCK_MODEL, n_in, n_out, cost)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint() -> set[int]:
    if CKPT_FILE.exists():
        data = json.loads(CKPT_FILE.read_text())
        s    = set(data.get("completed_gidx", []))
        logger.info(f"Checkpoint loaded — {len(s):,} already completed")
        return s
    return set()


def save_checkpoint(completed: set[int]) -> None:
    tmp = CKPT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"completed_gidx": list(completed),
                               "count":          len(completed)}))
    tmp.replace(CKPT_FILE)


# ──────────────────────────────────────────────────────────────────────────────
# Stream helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_lines(path: Path) -> int:
    n = 0
    with open(path) as f:
        for _ in f:
            n += 1
    return n


def stream_chunks(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def batched(iterable, size: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


# ──────────────────────────────────────────────────────────────────────────────
# vLLM pipeline  (all tiers, batched GPU inference)
# ──────────────────────────────────────────────────────────────────────────────

def run_vllm(args: argparse.Namespace) -> None:
    """
    Process all chunks with local vLLM — tuned for 1x A40 + AWQ INT4 70B.

    Pipeline:
      [ProcessPoolExecutor * cpu_workers]  ──►  [prefetch Queue]  ──►  [GPU llm.generate()]
              prompt building                    (prefetch batches)
                                                                    ──►  [writer Queue]  ──►  [disk]
                                                                           (async, never blocks GPU)
    """
    from vllm import LLM, SamplingParams

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 0. GPU info (log only — does not override hardcoded A40 settings) ─────
    detect_gpu()

    # ── 1. Load checkpoint + ALL pending chunks into RAM ──────────────────────
    completed_gidx = load_checkpoint()
    already_done   = len(completed_gidx)

    sep = "=" * 65
    logger.info(sep)
    logger.info("STAGE 3  —  vLLM backend  (AWQ INT4 70B, 1x A40)")
    logger.info(sep)
    logger.info("Pre-loading all pending chunks into RAM…")

    all_chunks: list[dict] = [
        c for c in stream_chunks(INPUT_FILE)
        if c["_gidx"] not in completed_gidx
    ]
    n_pending = len(all_chunks)

    logger.info(f"  Input          : {INPUT_FILE}")
    logger.info(f"  Output         : {OUT_FILE}")
    logger.info(f"  Already done   : {already_done:,}")
    logger.info(f"  Pending        : {n_pending:,}")
    logger.info(f"  vLLM model     : {args.vllm_model}")
    logger.info(f"  vLLM batch     : {args.vllm_batch}  (GPU batch size)")
    logger.info(f"  CPU workers    : {args.cpu_workers}  (prompt building)")
    logger.info(f"  Prefetch ahead : {args.prefetch}  batches")
    logger.info(f"  Tensor parallel: {args.tensor_parallel}")
    logger.info(f"  Temperature    : {args.temperature}")
    logger.info(f"  Max tokens     : {args.max_tokens}")
    logger.info(sep)

    if n_pending == 0:
        logger.success("All chunks already processed — nothing to do.")
        return

    # ── 2. Load vLLM model ────────────────────────────────────────────────────
    logger.info(f"Loading vLLM model {args.vllm_model!r} …")
    llm = LLM(
        model=args.vllm_model,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",              # no quantization, native bfloat16
        gpu_memory_utilization=0.88,
        max_model_len=2048,            # restore to 2048, we have headroom now
        max_num_seqs=args.vllm_batch,
        enable_prefix_caching=True,     # safe to re-enable without AWQ
        enforce_eager=False,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "<|start_header_id|>"],
    )
    logger.info("Model loaded. Starting prefetch pipeline…")

    # ── 3. Shared state ───────────────────────────────────────────────────────
    stats        = {"done": 0, "errors": 0, "since_ckpt": 0}
    ckpt_lock    = threading.Lock()
    producer_exc = []

    # ── 4. Async writer thread ────────────────────────────────────────────────
    write_q: queue.Queue = queue.Queue(maxsize=args.prefetch * 2)

    def _writer() -> None:
        with open(OUT_FILE, "a", encoding="utf-8") as f:
            while True:
                item = write_q.get()
                if item is None:
                    break
                f.write(json.dumps(item) + "\n")
                f.flush()

    writer_thread = threading.Thread(target=_writer, daemon=True, name="writer")
    writer_thread.start()

    # ── 5. CPU producer — builds prompts in parallel ──────────────────────────
    prefetch_q: queue.Queue = queue.Queue(maxsize=args.prefetch)

    def _producer() -> None:
        try:
            cpu_chunksize = max(1, args.vllm_batch // args.cpu_workers)
            with ThreadPoolExecutor(max_workers=args.cpu_workers) as pool:
                for raw_batch in batched(iter(all_chunks), args.vllm_batch):
                    work_items = [
                        (c, args.temperature, args.max_tokens) for c in raw_batch
                    ]
                    processed = list(
                        pool.map(_cpu_build_prompt, work_items, chunksize=cpu_chunksize)
                    )
                    prefetch_q.put(processed)
        except Exception as exc:
            producer_exc.append(exc)
        finally:
            prefetch_q.put(None)

    producer_thread = threading.Thread(target=_producer, daemon=True, name="cpu-producer")
    producer_thread.start()

    # ── 6. GPU consumer loop ──────────────────────────────────────────────────
    pbar = tqdm(
        total=n_pending, unit="chunk", dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    t_start  = time.time()
    gpu_idle = 0.0

    while True:
        t_wait     = time.time()
        batch_data = prefetch_q.get()
        gpu_idle  += time.time() - t_wait

        if batch_data is None:
            break

        if producer_exc:
            logger.error(f"Producer failed: {producer_exc[0]}")
            break

        chunks  = [d[0] for d in batch_data]
        prompts = [d[1] for d in batch_data]

        try:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        except Exception as exc:
            logger.error(f"vLLM batch failed ({len(prompts)} prompts): {exc}")
            stats["errors"] += len(prompts)
            pbar.update(len(prompts))
            continue

        for chunk, output in zip(chunks, outputs):
            try:
                raw_text            = output.outputs[0].text.strip()
                noised, instruction = parse_response(raw_text, chunk["text"])
            except Exception:
                noised      = chunk["text"]
                instruction = ""
                stats["errors"] += 1

            record = _build_record(
                chunk, chunk["text"], noised, instruction,
                args.vllm_model, 0, 0, 0.0,
            )
            write_q.put(record)

            with ckpt_lock:
                stats["done"]       += 1
                stats["since_ckpt"] += 1
                completed_gidx.add(chunk["_gidx"])
                if stats["since_ckpt"] >= CHECKPOINT_FREQ:
                    save_checkpoint(completed_gidx)
                    stats["since_ckpt"] = 0

        pbar.update(len(chunks))
        pbar.set_postfix({
            "gpu_idle_s": f"{gpu_idle:.0f}",
            "errors":     stats["errors"],
            "prefetch_q": prefetch_q.qsize(),
        }, refresh=False)

        logger.debug(
            f"Batch {len(chunks)} done | "
            f"sample instr: {outputs[0].outputs[0].text.strip()[:80]!r}"
        )

    pbar.close()

    # ── 7. Shutdown ───────────────────────────────────────────────────────────
    write_q.put(None)
    writer_thread.join()
    producer_thread.join()
    save_checkpoint(completed_gidx)

    elapsed   = time.time() - t_start
    mins, sec = divmod(elapsed, 60)
    gpu_busy  = elapsed - gpu_idle

    logger.info(sep)
    logger.info("STAGE 3 vLLM COMPLETE")
    logger.info(sep)
    logger.info(f"  Processed      : {stats['done']:,}  (errors: {stats['errors']})")
    logger.info(f"  Elapsed        : {int(mins)}m {sec:.0f}s")
    logger.info(f"  Throughput     : {stats['done'] / elapsed:.1f} chunks/sec")
    logger.info(f"  GPU busy time  : {gpu_busy:.1f}s  ({100*gpu_busy/elapsed:.1f}%)")
    logger.info(f"  GPU idle time  : {gpu_idle:.1f}s  ({100*gpu_idle/elapsed:.1f}%) ← lower is better")
    logger.info(f"  Output         : {OUT_FILE}")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Bedrock pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_bedrock(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    completed_gidx = load_checkpoint()
    already_done   = len(completed_gidx)

    logger.info("Counting total chunks…")
    total_chunks = count_lines(INPUT_FILE)
    n_todo       = total_chunks - already_done

    sep = "=" * 65
    logger.info(sep)
    logger.info("STAGE 3  —  Bedrock backend  (all tiers, 70B)")
    logger.info(sep)
    logger.info(f"  Input          : {INPUT_FILE}")
    logger.info(f"  Output         : {OUT_FILE}")
    logger.info(f"  Total chunks   : {total_chunks:,}")
    logger.info(f"  Already done   : {already_done:,}")
    logger.info(f"  To process     : {n_todo:,}")
    logger.info(f"  Workers        : {args.workers}")
    logger.info(f"  Batch size     : {args.batch}")
    logger.info(f"  Temperature    : {args.temperature}")
    logger.info(f"  Dry run        : {args.dry_run}")
    logger.info(sep)

    if n_todo == 0:
        logger.success("All chunks already processed — nothing to do.")
        return

    write_lock = threading.Lock()
    ckpt_lock  = threading.Lock()
    stats      = {"cost": 0.0, "tok_in": 0, "tok_out": 0,
                  "done": 0, "errors": 0, "since_ckpt": 0}

    out_f = open(OUT_FILE, "a", encoding="utf-8", buffering=1)

    def write_result(record: dict) -> None:
        with write_lock:
            out_f.write(json.dumps(record) + "\n")

    def update_stats(record: dict) -> None:
        with ckpt_lock:
            stats["cost"]    += record["cost_usd"]
            stats["tok_in"]  += record["tokens_in"]
            stats["tok_out"] += record["tokens_out"]
            stats["done"]    += 1
            stats["since_ckpt"] += 1
            completed_gidx.add(record["_gidx"])
            if stats["since_ckpt"] >= CHECKPOINT_FREQ:
                save_checkpoint(completed_gidx)
                stats["since_ckpt"] = 0

    pending = (
        c for c in stream_chunks(INPUT_FILE)
        if c["_gidx"] not in completed_gidx
    )

    pbar = tqdm(
        total=n_todo, unit="chunk", dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch in batched(pending, args.batch):
            futures = {
                pool.submit(process_chunk, chunk, args.dry_run,
                            args.temperature, args.max_tokens): chunk
                for chunk in batch
            }
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    record = future.result()
                    write_result(record)
                    update_stats(record)
                    pbar.update(1)
                    pbar.set_postfix({"cost": f"${stats['cost']:.3f}",
                                      "errors": stats["errors"]}, refresh=True)
                    logger.debug(
                        f"[{record['noise_tier']:4s}] gidx={record['_gidx']} | "
                        f"topic={record['topic_name'][:20]:20s} | "
                        f"in={record['tokens_in']:4d} out={record['tokens_out']:3d} | "
                        f"${record['cost_usd']:.5f} | "
                        f"instr: {record['instruction'][:60]}"
                    )
                except Exception as exc:
                    stats["errors"] += 1
                    logger.error(f"FAILED gidx={chunk['_gidx']} topic={chunk.get('topic_name')} — {exc}")
                    pbar.update(1)

            del futures
            gc.collect()

    pbar.close()
    out_f.close()
    save_checkpoint(completed_gidx)

    elapsed   = time.time() - t_start
    mins, sec = divmod(elapsed, 60)

    logger.info(sep)
    logger.info("STAGE 3 COMPLETE")
    logger.info(sep)
    logger.info(f"  Processed      : {stats['done']:,}  (errors: {stats['errors']})")
    logger.info(f"  Tokens in      : {stats['tok_in']:,}")
    logger.info(f"  Tokens out     : {stats['tok_out']:,}")
    logger.info(f"  Total cost     : ${stats['cost']:.4f}")
    logger.info(f"  Elapsed        : {int(mins)}m {sec:.0f}s")
    logger.info(f"  Throughput     : {stats['done']/elapsed:.1f} chunks/sec")
    logger.info(f"  Output         : {OUT_FILE}")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 — full 70B noise injection")

    p.add_argument("--backend", choices=["bedrock", "vllm"], default="vllm",
                   help="Inference backend: 'vllm' (default, HPC) or 'bedrock' (cloud)")

    # ── vLLM options ──────────────────────────────────────────────────────────
    p.add_argument("--vllm-model",      type=str, default=DEFAULT_VLLM_MODEL,
                   dest="vllm_model")
    p.add_argument("--vllm-batch",      type=int, default=DEFAULT_VLLM_BATCH,
                   dest="vllm_batch",
                   help=f"GPU batch size (default {DEFAULT_VLLM_BATCH}). "
                        "Raise only if KV cache headroom allows.")
    p.add_argument("--tensor-parallel", type=int, default=DEFAULT_TENSOR_PARALLEL,
                   dest="tensor_parallel",
                   help=f"Number of GPUs (default {DEFAULT_TENSOR_PARALLEL})")
    p.add_argument("--cpu-workers",     type=int, default=DEFAULT_CPU_WORKERS,
                   dest="cpu_workers",
                   help=f"CPU processes for prompt building (default {DEFAULT_CPU_WORKERS})")
    p.add_argument("--prefetch",        type=int, default=DEFAULT_PREFETCH,
                   dest="prefetch",
                   help=f"Batches to pre-build ahead of GPU (default {DEFAULT_PREFETCH})")

    # ── Bedrock options ───────────────────────────────────────────────────────
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Parallel Bedrock threads (default {DEFAULT_WORKERS})")
    p.add_argument("--batch",   type=int, default=DEFAULT_BATCH,
                   help=f"Chunks per GC batch, Bedrock only (default {DEFAULT_BATCH})")

    # ── Shared ────────────────────────────────────────────────────────────────
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help=f"Sampling temperature (default {DEFAULT_TEMPERATURE})")
    p.add_argument("--max-tokens",  type=int,   default=DEFAULT_MAX_TOKENS,
                   dest="max_tokens",
                   help=f"Max output tokens (default {DEFAULT_MAX_TOKENS})")
    p.add_argument("--dry-run",     action="store_true",
                   help="Skip inference — write fake records, costs nothing")

    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    if args.backend == "vllm":
        run_vllm(args)
    else:
        run_bedrock(args)