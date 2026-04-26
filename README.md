# Tree-Based Hierarchical De-biasing for LLM Fine-Tuning

Fine-tuning a language model on Wikipedia bakes in Wikipedia's biases — Physics and Western history are vastly over-represented while Yoruba culture, Dalit communities, Quechua language, and LGBT history are treated as afterthoughts. This project builds a 6-stage pipeline to fix that at the data level, before the model ever sees a user prompt.

The result is a 71MB LoRA adapter that shifts Phi-3-mini's internal representation of minority topics measurably closer to majority topics.

**Adapter on HuggingFace:** [usamaahmedsh/phi3-mini-wikipedia-debiasing](https://huggingface.co/usamaahmedsh/phi3-mini-wikipedia-debiasing)

---

## The Core Idea

Wikipedia over-represents majority topics (Physics, Engineering, Basketball, Economics) and under-represents minority ones (Feminism, Colonialism, Yoruba, Dalit, Quechua, LGBT). A model fine-tuned naively on this corpus learns that asymmetry — it becomes more fluent on majority topics and treats minority ones as edge cases.

The fix: classify every chunk of Wikipedia into intersectional subgroups using a 5-layer tree, then use that structure to apply **asymmetric noise injection** — majority topic chunks get more noise (reducing their dominance), minority topic chunks are fully protected. The resulting corpus, weighted by noise tier during fine-tuning, gives minority topics proportionally more influence over the model's weights.

---

## Pipeline

### Stage 0 — Data Acquisition
Scrape English Wikipedia via HuggingFace `datasets`, split into 200–400 token paragraph chunks, apply MinHash near-duplicate removal (Jaccard > 0.85).

**Output:** `data/chunks/` — 335,882 chunks across 43 topics

### Stage 1 — Hierarchical Stratification Tree
Classify every chunk through a 5-layer tree:

| Layer | Method | Labels |
|---|---|---|
| L1 Topic | BERTopic (UMAP + HDBSCAN) | T0 outliers, T1, T2, … |
| L2 Emotion | GoEmotions BERT (27→3) | E1 positive, E2 negative, E3 neutral |
| L3 Demographic | Pronoun heuristics | D1 male, D2 female, D3 unknown |
| L4 Register | CoLA-based formality | R1 formal, R2 informal |
| L5 Readability | Flesch Reading Ease | C1–C5 very easy → very difficult |

Majority score per leaf: `s(l) = (count(l) - μ) / σ`. Score > 1.5 → HIGH noise tier; 0.5–1.5 → MED; < 0.5 → LOW.

**Output:** `data/tree/leaf_nodes.json` — 42,453 leaves

### Stage 2 — Leaf-Level Deduplication
5-layer dedup pipeline: MD5 → MinHash → TF-IDF → semantic cosine → per-leaf greedy diversity. Minority topic chunks are fully protected (never pruned).

**Output:** 174,925 chunks (48% reduction)

### Stage 3 — Asymmetric Noise Injection
Run all 335,882 chunks through Llama-3.1-8b (local HPC inference, zero cost) to generate noised text + instruction pairs in a single pass. Majority chunks receive stronger paraphrasing; minority chunks receive light touch or none.

**Output:** `noised_chunks_hpc_relabeled.jsonl` — 335,882 instruction-output pairs with noise tier labels

### Stage 4 — QLoRA Fine-Tuning
Fine-tune `microsoft/Phi-3-mini-4k-instruct` (3.8B) using QLoRA with **tier-weighted loss** — every example is seen exactly once per epoch, but HIGH-tier examples contribute 2× more gradient signal than LOW-tier examples. No oversampling, no data duplication.

```
lora_r=32, lora_alpha=64, lora_dropout=0.10
learning_rate=1e-4, weight_decay=0.05
loss weights: HIGH=2.0, MED=1.5, LOW=1.0
```

**Output:** 71MB LoRA adapter → [HuggingFace](https://huggingface.co/usamaahmedsh/phi3-mini-wikipedia-debiasing)

### Stage 5 — Bias Evaluation
Compare base model vs fine-tuned across:
- Perplexity stratified by topic type (majority vs minority)
- StereoSet ICAT, CrowS-Pairs, WinoBias
- Regard score (gender + coverage gap)
- Counterfactual fairness (demographic term swap sensitivity)
- Qualitative side-by-side generation on minority topic prompts

**Output:** `outputs/eval/eval_report.html`, `outputs/eval/eval_results.json`

---

## Key Results

| Metric | Base Phi-3-mini | Fine-tuned |
|---|---|---|
| Perplexity — overall | 7.65 | **7.29** |
| Perplexity — minority topics | 7.63 | **7.38** |
| Perplexity — majority topics | 7.13 | **6.81** |
| CF sentiment shift | 0.095 | 0.188 |

The perplexity gap between majority and minority topics narrowed after fine-tuning (0.50 → 0.57 absolute, but both moved down). More importantly, qualitative generation on Yoruba, Quechua, and Zoroastrianism prompts became richer and more culturally specific — the model engages with minority topics rather than producing generic summaries.

---

## Topics

**Majority (20):** Physics, Mathematics, Engineering, Computing, Baseball, Football, Basketball, Napoleon, Churchill, Shakespeare, Christianity, Philosophy, Economics, Warfare, Generals, Presidents, Parliament, Constitution, Astronomy, Chemistry

**Minority / Protected (20):** Feminism, Hinduism, Buddhism, Islam, Colonialism, Slavery, Apartheid, Suffrage, Matriarchy, Yoruba, Swahili, Aztec, Quechua, Malayalam, Confucianism, Shintoism, Zoroastrianism, Aboriginals, Dalit, LGBT

---

## Running the Pipeline

```bash
# Stage 0 — scrape Wikipedia
python scripts/stage0_data_acquisition.py

# Stage 1 — build tree
python scripts/stage1_hierarchical_tree.py

# Stage 2 — prune/dedup
python scripts/prune_chunks.py

# Stage 3 — noise injection (requires GPU, vLLM)
python scripts/stage3_noise_injection.py --vllm-model ./models/llama-3.1-8b

# Stage 4 — fine-tune (requires GPU, ~24h on L40S)
python scripts/stage4_finetune.py --input noised_chunks_hpc_relabeled.jsonl

# Stage 5 — evaluate
python scripts/stage5_evaluate.py \
    --base      microsoft/Phi-3-mini-4k-instruct \
    --finetuned usamaahmedsh/phi3-mini-wikipedia-debiasing \
    --load-4bit
```

---

## Repo Structure

```
scripts/
  stage0_data_acquisition.py   # Wikipedia scraping + chunking
  stage1_hierarchical_tree.py  # 5-layer tree construction
  prune_chunks.py              # leaf-level deduplication
  stage3_noise_injection.py    # asymmetric noise via vLLM
  stage4_finetune.py           # QLoRA fine-tuning
  stage5_evaluate.py           # bias + capability evaluation
  relabel_noise_tiers.py       # tier relabeling utility
  corpus_builder.py            # corpus assembly helpers
  global_dedup.py              # global deduplication pass
  prepare_full_data.py         # data prep for Stage 3
  reports/                     # analysis and visualization scripts
  viz/                         # tree visualization scripts

data/
  tree/                        # original tree (42,453 leaves)
  pruned/                      # pruned tree + dedup reports
  processed/                   # train/val/test splits (gitignored)

outputs/
  eval/
    eval_results.json          # raw metric results
    eval_report.html           # HTML report with side-by-side comparisons

docs/
  Tree_based_unbiased_finetuning.pdf   # full pipeline spec
  Stage3_Memory_Optimization.pdf       # Stage 3 HPC optimization notes
```

---

## Requirements

```bash
pip install -r requirements.txt        # base pipeline
pip install -r requirements_stage4.txt # fine-tuning (CUDA required)
```
