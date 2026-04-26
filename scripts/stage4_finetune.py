#!/usr/bin/env python3
"""
stage4_finetune.py — QLoRA fine-tuning for de-biased Wikipedia instruction pairs.

Auto-detects hardware and configures batch size, precision, LoRA rank, and
sequence packing accordingly. No manual flags needed — just run it.

Usage:
    python scripts/stage4_finetune.py
    python scripts/stage4_finetune.py --input data/processed/train_ready.jsonl
    python scripts/stage4_finetune.py --model microsoft/Phi-3-mini-4k-instruct --resume outputs/checkpoints/run_X/checkpoint-200

Outputs:
    data/processed/train_ready.jsonl   — filtered, relabeled, split
    outputs/checkpoints/<run_id>/      — LoRA adapter checkpoints
    outputs/checkpoints/<run_id>/best/ — best checkpoint by val loss

Changes vs previous version:
    - TIER_WEIGHTS are now used as loss multipliers, not oversampling repeats.
      Each example is seen exactly once per epoch — no duplication, no memorization.
      HIGH examples contribute 2x loss, MED 1.5x, LOW 1x. Statistically clean.
    - packing disabled: per-example loss weighting requires knowing example
      boundaries, which sequence packing destroys.
    - Val set uses the same loss weighting via WeightedSFTTrainer so eval_loss
      reflects the same distribution as training.
    - lora_dropout increased 0.05 -> 0.10 for stronger regularization.
    - lora_r reduced 64 -> 32 (L40S tier) — fewer adapter params, less capacity to overfit.
    - lora_alpha adjusted accordingly 128 -> 64 (keeps alpha/r ratio = 2).
    - learning_rate reduced 2e-4 -> 1e-4 — slower, more conservative updates.
    - weight_decay increased 0.01 -> 0.05 — stronger L2 regularization via AdamW.
    - eval_steps / save_steps reduced to 1400 (L40S tier) for finer-grained
      early stopping.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
DATA_IN     = ROOT / "noised_chunks_hpc_relabeled.jsonl"
PROCESSED   = ROOT / "data/processed"
CHECKPOINTS = ROOT / "outputs/checkpoints"

TIER_WEIGHTS = {"HIGH": 2.0, "MED": 1.5, "LOW": 1.0}  # loss multipliers — each example seen once, HIGH contributes 2x loss
SEED         = 42

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── hardware detection ────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    device:          str   = "cpu"
    device_name:     str   = "CPU"
    gpu_count:       int   = 0
    vram_gb:         float = 0.0
    ram_gb:          float = 0.0
    cpu_cores:       int   = 1
    cpu_threads:     int   = 1
    compute_cap:     tuple = field(default_factory=lambda: (0, 0))
    supports_bf16:   bool  = False
    supports_flash:  bool  = False
    supports_4bit:   bool  = False  # bitsandbytes not available on MPS
    is_multi_gpu:    bool  = False


def detect_hardware() -> HardwareProfile:
    hw = HardwareProfile()
    hw.ram_gb     = psutil.virtual_memory().total / (1024 ** 3)
    hw.cpu_cores  = psutil.cpu_count(logical=False) or 1
    hw.cpu_threads = psutil.cpu_count(logical=True) or 1

    if torch.cuda.is_available():
        hw.device    = "cuda"
        hw.gpu_count = torch.cuda.device_count()
        props        = torch.cuda.get_device_properties(0)
        hw.device_name   = props.name
        hw.vram_gb       = props.total_memory / (1024 ** 3)
        hw.compute_cap   = torch.cuda.get_device_capability(0)
        hw.supports_bf16 = hw.compute_cap[0] >= 8          # Ampere+
        hw.supports_4bit = True                             # bitsandbytes works on CUDA
        hw.is_multi_gpu  = hw.gpu_count > 1
        # Flash Attention 2 requires Ampere+ and the package installed
        try:
            import flash_attn  # noqa: F401
            hw.supports_flash = hw.compute_cap[0] >= 8
        except ImportError:
            hw.supports_flash = False

    elif torch.backends.mps.is_available():
        hw.device      = "mps"
        hw.device_name = "Apple Silicon (MPS)"
        hw.gpu_count   = 1
        hw.vram_gb     = hw.ram_gb       # unified memory
        hw.supports_bf16 = False         # MPS bf16 unstable
        hw.supports_4bit = False         # bitsandbytes no MPS support
        hw.supports_flash = False

    log.info("Hardware profile:")
    log.info(f"  Device      : {hw.device_name} ({hw.device})")
    log.info(f"  VRAM/RAM    : {hw.vram_gb:.1f} GB VRAM | {hw.ram_gb:.1f} GB RAM")
    log.info(f"  CPU         : {hw.cpu_cores} cores / {hw.cpu_threads} threads")
    log.info(f"  BF16        : {hw.supports_bf16}")
    log.info(f"  4-bit quant : {hw.supports_4bit}")
    log.info(f"  Flash Attn  : {hw.supports_flash}")
    log.info(f"  Multi-GPU   : {hw.is_multi_gpu} ({hw.gpu_count} GPUs)")
    return hw


# ── adaptive config ───────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # model
    model_name:     str   = "microsoft/Phi-3-mini-4k-instruct"
    max_seq_length: int   = 1024

    # LoRA
    lora_r:         int   = 16
    lora_alpha:     int   = 32
    lora_dropout:   float = 0.10   # increased from 0.05 for stronger regularization
    lora_targets:   list  = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # training
    num_epochs:               int   = 3
    per_device_train_batch:   int   = 1
    per_device_eval_batch:    int   = 2
    gradient_accumulation:    int   = 32
    learning_rate:            float = 1e-4    # reduced from 2e-4 — more conservative updates
    weight_decay:             float = 0.05   # increased from 0.01 — stronger L2 via AdamW
    warmup_ratio:             float = 0.03
    lr_scheduler:             str   = "cosine"
    max_grad_norm:            float = 1.0
    early_stopping_patience:  int   = 3
    eval_steps:               int   = 500
    save_steps:               int   = 500
    logging_steps:            int   = 50

    # precision
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bf16:         bool = False
    fp16:         bool = False

    # efficiency
    use_flash_attention: bool  = False
    packing:             bool  = True
    dataloader_workers:  int   = 4
    gradient_checkpointing: bool = True

    # data
    val_ratio:   float = 0.1
    test_ratio:  float = 0.1
    seed:        int   = SEED


def build_config(hw: HardwareProfile, overrides: dict) -> TrainConfig:
    cfg = TrainConfig()

    if hw.device == "cuda":
        vram = hw.vram_gb

        if vram >= 44:          # ── L40S / A40 / RTX 6000 Ada (44–48 GB) ──
            cfg.lora_r, cfg.lora_alpha     = 32, 64   # reduced from 64/128 — less capacity to overfit
            cfg.per_device_train_batch     = 40
            cfg.gradient_accumulation      = 2
            cfg.max_seq_length             = 3096
            cfg.per_device_eval_batch      = 8
            cfg.eval_steps                 = 1000   # halved from 2800 — more early stopping chances
            cfg.save_steps                 = 1000
            cfg.logging_steps              = 10
        elif vram >= 60:        # A100 80GB, H100
            cfg.lora_r, cfg.lora_alpha     = 64, 128
            cfg.per_device_train_batch     = 8
            cfg.gradient_accumulation      = 4
            cfg.max_seq_length             = 2048
        elif vram >= 35:        # A100 40GB
            cfg.lora_r, cfg.lora_alpha     = 32, 64
            cfg.per_device_train_batch     = 4
            cfg.gradient_accumulation      = 8
        elif vram >= 20:        # A6000, RTX 3090/4090
            cfg.lora_r, cfg.lora_alpha     = 32, 64
            cfg.per_device_train_batch     = 2
            cfg.gradient_accumulation      = 16
        elif vram >= 10:        # RTX 3080, T4
            cfg.lora_r, cfg.lora_alpha     = 16, 32
            cfg.per_device_train_batch     = 1
            cfg.gradient_accumulation      = 32
        else:                   # small CUDA GPU
            cfg.lora_r, cfg.lora_alpha     = 8, 16
            cfg.per_device_train_batch     = 1
            cfg.gradient_accumulation      = 64

        cfg.load_in_4bit        = True
        cfg.bf16                = hw.supports_bf16
        cfg.fp16                = not hw.supports_bf16
        cfg.use_flash_attention = hw.supports_flash
        cfg.dataloader_workers  = min(hw.cpu_threads // 2, 12)

    elif hw.device == "mps":
        ram = hw.ram_gb
        if ram >= 32:
            cfg.lora_r, cfg.lora_alpha   = 16, 32
            cfg.per_device_train_batch   = 1
            cfg.gradient_accumulation    = 32
            cfg.max_seq_length           = 1024
        else:
            cfg.lora_r, cfg.lora_alpha   = 8, 16
            cfg.per_device_train_batch   = 1
            cfg.gradient_accumulation    = 64
            cfg.max_seq_length           = 512

        cfg.load_in_4bit       = False
        cfg.bf16               = False
        cfg.fp16               = False
        cfg.packing            = False
        cfg.dataloader_workers = 0
        cfg.eval_steps         = 200
        cfg.save_steps         = 200

    else:  # CPU
        cfg.lora_r, cfg.lora_alpha   = 4, 8
        cfg.per_device_train_batch   = 1
        cfg.gradient_accumulation    = 64
        cfg.max_seq_length           = 256
        cfg.load_in_4bit             = False
        cfg.bf16, cfg.fp16           = False, False
        cfg.gradient_checkpointing   = False
        cfg.dataloader_workers       = 0
        log.warning("Running on CPU — training will be very slow.")

    # effective batch size summary
    eff = cfg.per_device_train_batch * cfg.gradient_accumulation * max(hw.gpu_count, 1)
    log.info(f"Effective batch size: {eff} "
             f"({cfg.per_device_train_batch} × {cfg.gradient_accumulation} accum "
             f"× {max(hw.gpu_count,1)} GPU)")

    # apply CLI overrides
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg


# ── data preparation ──────────────────────────────────────────────────────────

def prepare_dataset(input_path: Path, cfg: TrainConfig) -> dict[str, list]:
    """
    Load noised_chunks_hpc_relabeled.jsonl, filter, and split into
    train/val/test stratified by (topic, tier).

    Each record gets a 'loss_weight' field set to TIER_WEIGHTS[tier].
    No oversampling — every example is seen exactly once per epoch.
    The loss weight is used at training time to scale the per-example loss.
    """
    log.info(f"Loading data from {input_path} ...")
    all_records = []
    empty_skipped = 0

    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            instruction = d.get("instruction", "").strip()
            noised_text = d.get("noised_text", "").strip()

            # hard filters
            if not instruction:
                empty_skipped += 1
                continue
            if len(noised_text) < 50:
                continue

            tier = d.get("noise_tier_corrected", d.get("noise_tier", "LOW"))
            all_records.append({
                "_gidx":          d.get("_gidx", 0),
                "topic":          d.get("topic_name", "Unknown"),
                "tier":           tier,
                "loss_weight":    TIER_WEIGHTS.get(tier, 1.0),
                "majority_score": d.get("majority_score_corrected", 0.0),
                "instruction":    instruction,
                "noised_text":    noised_text,
                "leaf_id":        d.get("leaf_id", ""),
            })

    log.info(f"  Loaded {len(all_records):,} usable records (dropped {empty_skipped:,} empty instructions)")

    # stratified split by (topic, tier)
    random.seed(cfg.seed)
    strata: dict[str, list] = {}
    for r in all_records:
        key = f"{r['topic']}_{r['tier']}"
        strata.setdefault(key, []).append(r)

    train_raw, val_raw, test_raw = [], [], []
    for key, group in strata.items():
        random.shuffle(group)
        n = len(group)
        n_test = max(1, int(n * cfg.test_ratio))
        n_val  = max(1, int(n * cfg.val_ratio))
        test_raw.extend(group[:n_test])
        val_raw.extend(group[n_test:n_test + n_val])
        train_raw.extend(group[n_test + n_val:])

    random.shuffle(train_raw)
    random.shuffle(val_raw)

    log.info(f"  Split -> train:{len(train_raw):,}  val:{len(val_raw):,}  test:{len(test_raw):,}")

    # log tier distribution for each split
    for split_name, records in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        counts = {}
        for r in records:
            counts[r["tier"]] = counts.get(r["tier"], 0) + 1
        log.info(f"  {split_name} tier distribution: {counts}")

    # save splits to disk
    PROCESSED.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_raw, "val": val_raw, "test": test_raw}
    for split_name, records in splits.items():
        out_path = PROCESSED / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        log.info(f"  Saved {split_name} -> {out_path}")

    return splits


def format_example(record: dict, tokenizer) -> str:
    """Apply the model's native chat template to one training pair."""
    messages = [
        {"role": "user",      "content": record["instruction"]},
        {"role": "assistant", "content": record["noised_text"]},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # fallback for models without a chat template
        return f"### Instruction:\n{record['instruction']}\n\n### Response:\n{record['noised_text']}"


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: TrainConfig, hw: HardwareProfile):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    log.info(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "device_map": "auto" if hw.device == "cuda" else None,
    }

    if cfg.load_in_4bit and hw.supports_4bit:
        log.info("Loading model in 4-bit (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    elif cfg.bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif cfg.fp16:
        model_kwargs["dtype"] = torch.float16
    else:
        model_kwargs["dtype"] = torch.float32

    if cfg.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    log.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    if hw.device == "mps":
        model = model.to("mps")

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    log.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def build_peft_model(model, cfg: TrainConfig, hw: HardwareProfile):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.load_in_4bit and hw.supports_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_targets,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = sum(p.numel() for p in model.parameters() if p.requires_grad), \
                       sum(p.numel() for p in model.parameters())
    log.info(f"  LoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


# ── training ──────────────────────────────────────────────────────────────────

def build_hf_dataset(records: list, tokenizer, cfg: TrainConfig):
    """Convert records to a HuggingFace Dataset with formatted text and loss weights."""
    from datasets import Dataset

    texts   = [format_example(r, tokenizer) for r in records]
    weights = [float(r.get("loss_weight", 1.0)) for r in records]
    return Dataset.from_dict({"text": texts, "loss_weight": weights})


# ── weighted trainer ──────────────────────────────────────────────────────────

class TierWeightedCollator:
    """
    Wraps the default data collator. Keeps only tensor-compatible columns
    (input_ids, attention_mask, labels, loss_weight) and drops everything
    else (text strings, tier labels, etc.) before collation.
    """
    KEEP = {"input_ids", "attention_mask", "labels", "loss_weight"}

    def __init__(self, base_collator):
        self.base = base_collator

    def __call__(self, features):
        # split out loss_weight before passing to base collator
        weights = [f.pop("loss_weight", 1.0) for f in features]
        # drop any remaining non-tensor fields
        clean = [{k: v for k, v in f.items() if k in self.KEEP - {"loss_weight"}} for f in features]
        batch = self.base(clean)
        batch["loss_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedSFTTrainer:
    """
    Mixin that overrides compute_loss to scale each example's loss by its
    tier weight. Injected into SFTTrainer via multiple inheritance.

    How it works:
      - The dataset includes a 'loss_weight' column (float, per example).
      - The data collator passes it through as a tensor in the batch.
      - compute_loss pulls it out, computes token-level CE loss with
        reduction='none', masks padding, then multiplies the mean loss
        for each sequence by its weight before averaging across the batch.
      - This means HIGH examples (weight=2.0) contribute twice as much
        gradient signal as LOW examples (weight=1.0), with no duplication.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # pop the weight tensor before forwarding to the model
        weights = inputs.pop("loss_weight", None)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        # shift for causal LM: predict token i+1 from token i
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        # (batch, seq_len-1)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        # mean over non-padding tokens per example
        valid_mask   = (shift_labels != -100).float()
        counts       = valid_mask.sum(dim=-1).clamp(min=1)
        example_loss = (token_loss * valid_mask).sum(dim=-1) / counts  # (batch,)

        if weights is not None:
            w = weights.to(example_loss.device).float()
            loss = (example_loss * w).sum() / w.sum()
        else:
            loss = example_loss.mean()

        return (loss, outputs) if return_outputs else loss


def train(model, tokenizer, splits: dict, cfg: TrainConfig, hw: HardwareProfile, run_dir: Path, resume: str = None):
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer, SFTConfig

    train_ds = build_hf_dataset(splits["train"], tokenizer, cfg)
    val_ds   = build_hf_dataset(splits["val"],   tokenizer, cfg)

    # try wandb, fall back to tensorboard, then none
    report_to = "none"
    try:
        import wandb  # noqa: F401
        report_to = "wandb"
    except ImportError:
        try:
            import tensorboard  # noqa: F401
            report_to = "tensorboard"
        except ImportError:
            pass

    # Build the weighted trainer class by mixing WeightedSFTTrainer into SFTTrainer.
    # WeightedSFTTrainer.compute_loss takes priority via Python MRO.
    class TierWeightedSFTTrainer(WeightedSFTTrainer, SFTTrainer):

        def _get_collator(self):
            base = self.data_collator
            return TierWeightedCollator(base)

        def get_train_dataloader(self):
            old = self.data_collator
            self.data_collator = self._get_collator()
            dl = super().get_train_dataloader()
            self.data_collator = old
            return dl

        def get_eval_dataloader(self, eval_dataset=None):
            old = self.data_collator
            self.data_collator = self._get_collator()
            dl = super().get_eval_dataloader(eval_dataset)
            self.data_collator = old
            return dl

    sft_cfg = SFTConfig(
        output_dir                    = str(run_dir),
        num_train_epochs              = cfg.num_epochs,
        per_device_train_batch_size   = cfg.per_device_train_batch,
        per_device_eval_batch_size    = cfg.per_device_eval_batch,
        gradient_accumulation_steps   = cfg.gradient_accumulation,
        learning_rate                 = cfg.learning_rate,
        weight_decay                  = cfg.weight_decay,
        warmup_ratio                  = cfg.warmup_ratio,
        lr_scheduler_type             = cfg.lr_scheduler,
        max_grad_norm                 = cfg.max_grad_norm,
        bf16                          = cfg.bf16,
        fp16                          = cfg.fp16,
        gradient_checkpointing        = cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        eval_strategy                 = "steps",
        eval_steps                    = cfg.eval_steps,
        save_strategy                 = "steps",
        save_steps                    = cfg.save_steps,
        save_total_limit              = 5,
        load_best_model_at_end        = True,
        metric_for_best_model         = "eval_loss",
        greater_is_better             = False,
        logging_steps                 = cfg.logging_steps,
        dataloader_num_workers        = cfg.dataloader_workers,
        dataloader_pin_memory         = True,
        dataloader_prefetch_factor    = 2,
        report_to                     = report_to,
        seed                          = cfg.seed,
        run_name                      = run_dir.name,
        remove_unused_columns         = False,  # collator handles stripping non-tensor columns
        optim                         = "paged_adamw_8bit" if hw.supports_4bit else "adamw_torch",
        # SFT-specific
        max_seq_length                = cfg.max_seq_length,
        packing                       = False,  # packing destroys per-example boundaries needed for loss weighting
        dataset_text_field            = "text",
        tf32                          = True,
        torch_compile                 = False,
    )

    trainer = TierWeightedSFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        args             = sft_cfg,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    log.info("Starting training with tier-weighted loss (no oversampling) ...")
    if resume:
        log.info(f"Resuming from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)

    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    log.info(f"Best model saved -> {best_dir}")

    # save config alongside checkpoint
    with open(best_dir / "train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(best_dir / "hw_profile.json", "w") as f:
        profile = asdict(hw)
        profile["compute_cap"] = list(profile["compute_cap"])
        json.dump(profile, f, indent=2)

    return trainer


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 4 — QLoRA fine-tuning")
    parser.add_argument("--input",     default=str(DATA_IN),  help="Path to relabeled JSONL")
    parser.add_argument("--model",     default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--resume",    default=None,          help="Resume from checkpoint dir, e.g. outputs/checkpoints/run_X/checkpoint-1400")
    parser.add_argument("--epochs",    type=int, default=None)
    parser.add_argument("--lr",        type=float, default=None)
    parser.add_argument("--lora-r",    type=int, default=None, dest="lora_r")
    parser.add_argument("--no-pack",   action="store_true", help="Disable sequence packing")
    parser.add_argument("--skip-prep", action="store_true", help="Skip data prep if already done")
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    hw  = detect_hardware()
    overrides = {"model_name": args.model}
    if args.epochs:  overrides["num_epochs"]    = args.epochs
    if args.lr:      overrides["learning_rate"] = args.lr
    if args.lora_r:  overrides["lora_r"]        = args.lora_r
    if args.no_pack: overrides["packing"]        = False

    cfg = build_config(hw, overrides)
    log.info(f"Training config: batch={cfg.per_device_train_batch} "
             f"accum={cfg.gradient_accumulation} lora_r={cfg.lora_r} "
             f"lora_dropout={cfg.lora_dropout} lr={cfg.learning_rate} "
             f"weight_decay={cfg.weight_decay} "
             f"bf16={cfg.bf16} 4bit={cfg.load_in_4bit} "
             f"loss_weighting=tier ({TIER_WEIGHTS})")

    # data
    train_ready = PROCESSED / "train.jsonl"
    if args.skip_prep and train_ready.exists():
        log.info("Skipping data prep — loading existing splits")
        splits = {}
        for split in ("train", "val", "test"):
            with open(PROCESSED / f"{split}.jsonl") as f:
                splits[split] = [json.loads(l) for l in f]
        for k, v in splits.items():
            log.info(f"  {k}: {len(v):,}")
    else:
        splits = prepare_dataset(Path(args.input), cfg)

    # run dir — reuse existing run dir when resuming so checkpoints stay together
    if args.resume:
        run_dir = Path(args.resume).parent
        log.info(f"Resuming into existing run dir: {run_dir}")
    else:
        run_id  = time.strftime("%Y%m%d_%H%M%S")
        run_dir = CHECKPOINTS / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # model
    model, tokenizer = load_model_and_tokenizer(cfg, hw)
    model            = build_peft_model(model, cfg, hw)

    # train
    trainer = train(model, tokenizer, splits, cfg, hw, run_dir, resume=args.resume)

    log.info("Training complete.")
    log.info(f"Checkpoints : {run_dir}")
    log.info(f"Best model  : {run_dir / 'best'}")


if __name__ == "__main__":
    main()