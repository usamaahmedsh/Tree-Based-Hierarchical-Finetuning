"""
Stage 1: Hierarchical Stratification Tree
==========================================
Layers applied sequentially to each chunk:
  L1 — BERTopic        → Topic (T1, T2, … Tn)
  L2 — GoEmotions BERT → Emotional tone (E1, E2, … En)
  L3 — M3-Inference    → Demographic signals (D1, D2, … Dn)
  L4 — Dialect/Formality model → Register (R1, R2, … Rn)
  L5 — Textstat/CLEAR  → Readability (C1, C2, … Cn)

Each chunk lands in exactly one leaf node.
Majority score: s(l) = (count(l) - μ) / σ

Outputs:
  - data/tree/leaf_assignments.jsonl   — chunk → leaf mapping
  - data/tree/leaf_nodes.json          — full leaf metadata
  - outputs/tree_viz/tree_<layer>.png  — one image per layer + full tree
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import textstat
import yaml
from loguru import logger
import torch
from tqdm import tqdm
import torch

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
logger.info(f"Using compute device: {DEVICE}")

warnings.filterwarnings("ignore")

# ─────────────────────────── config ─────────────────────────────

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_chunks(chunks_path: str = "data/chunks/chunks.jsonl") -> List[Dict[str, Any]]:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ─────────────────────────── data model ─────────────────────────

@dataclass
class LeafNode:
    leaf_id: str
    label_path: str                   # e.g. "T3 > E1 > D2 > R1 > C2"
    layer_labels: Dict[str, str]      # {layer_name: label_id}
    document_count: int = 0
    majority_score: float = 0.0
    cosine_similarity: float = 0.0   # intra-cluster cohesion (set after BERTopic)
    chunk_ids: List[int] = field(default_factory=list)


@dataclass
class TreeNode:
    node_id: str
    layer: int                         # 0=root, 1=topic, 2=emotion, …
    label: str                         # human-readable label
    short_id: str                      # T1 / E3 / D2 / R1 / C4
    parent_id: Optional[str]
    children: List["TreeNode"] = field(default_factory=list)
    chunk_ids: List[int] = field(default_factory=list)


# ──────────────────────────── layers ────────────────────────────

class Layer1BERTopic:
    """Topic modelling via BERTopic (UMAP + HDBSCAN + c-TF-IDF)."""

    PREFIX = "T"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.model = None
        self._label_map: Dict[int, str] = {}     # topic_int → label like "T1"
        self._topic_labels: Dict[int, str] = {}  # topic_int → human label

    def fit(self, texts: List[str]) -> None:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer

        logger.info("  [L1] Fitting BERTopic …")
        umap_model = UMAP(
            n_neighbors  = self.cfg["umap_n_neighbors"],
            n_components = self.cfg["umap_n_components"],
            metric       = "cosine",
            random_state = 42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size = self.cfg["min_cluster_size"],
            min_samples      = self.cfg["hdbscan_min_samples"],
            metric           = "euclidean",
            cluster_selection_method = "eom",
            prediction_data  = True,
        )
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        self.model = BERTopic(
            umap_model    = umap_model,
            hdbscan_model = hdbscan_model,
            embedding_model = embedding_model,
            top_n_words   = self.cfg.get("top_n_words", 10),
            verbose       = True,
        )
        topics, _ = self.model.fit_transform(texts)

        # Build label map: -1 is "outlier" → label as T0
        unique_topics = sorted(set(topics))
        counter = 0
        for t in unique_topics:
            if t == -1:
                self._label_map[t] = "T0"
                self._topic_labels[t] = "Outlier"
            else:
                counter += 1
                self._label_map[t] = f"T{counter}"
                # Get top words for this topic
                try:
                    words = [w for w, _ in self.model.get_topic(t)[:5]]
                    self._topic_labels[t] = ", ".join(words)
                except Exception:
                    self._topic_labels[t] = f"Topic {counter}"

        logger.info(f"  [L1] BERTopic found {len(unique_topics)} topics")

    def predict(self, texts: List[str]) -> List[str]:
        """Return short_id labels (T1, T2, …) for each text."""
        if self.model is None:
            raise RuntimeError("Layer 1 not fitted.")
        topics, _ = self.model.transform(texts)
        return [self._label_map.get(t, "T0") for t in topics]

    def fit_predict(self, texts: List[str]) -> List[str]:
        self.fit(texts)
        # Use the internal topics from fit to avoid double-transform
        topics = self.model.topics_
        return [self._label_map.get(t, "T0") for t in topics]

    def human_label(self, short_id: str) -> str:
        for t_int, s_id in self._label_map.items():
            if s_id == short_id:
                return self._topic_labels.get(t_int, short_id)
        return short_id


class Layer2GoEmotions:
    """Emotion classification using a fine-tuned BERT model."""

    PREFIX = "E"
    # Emotion groups → merged labels to keep tree manageable
    EMOTION_GROUPS = {
        "positive":  ["joy", "love", "optimism", "pride", "admiration",
                      "amusement", "excitement", "gratitude", "relief", "approval"],
        "negative":  ["anger", "annoyance", "disgust", "fear", "grief",
                      "disappointment", "disapproval", "remorse", "sadness",
                      "nervousness", "embarrassment"],
        "neutral":   ["neutral", "realization", "surprise", "curiosity",
                      "confusion", "desire", "caring"],
    }
    # Map individual emotion → group
    _EMOTION_TO_GROUP: Dict[str, str] = {}
    for group, emotions in EMOTION_GROUPS.items():
        for e in emotions:
            _EMOTION_TO_GROUP[e] = group

    GROUP_IDS = {"positive": "E1", "negative": "E2", "neutral": "E3"}

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.pipeline = None

    def _load(self) -> None:
        if self.pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline
        logger.info(f"  [L2] Loading GoEmotions model on {DEVICE} …")
        self.pipeline = hf_pipeline(
            "text-classification",
            model      = self.cfg.get("model", "bhadresh-savani/bert-base-uncased-emotion"),
            device     = DEVICE,
            top_k      = 1,
            truncation = True,
            max_length = 512,
        )

    def predict(self, texts: List[str]) -> List[str]:
        self._load()
        batch_size = self.cfg.get("batch_size", 32)
        results: List[str] = []

        for i in tqdm(range(0, len(texts), batch_size),
                      desc="  [L2] GoEmotions", leave=False):
            batch = texts[i : i + batch_size]
            try:
                preds = self.pipeline(batch)
                for pred in preds:
                    label = pred[0]["label"].lower() if isinstance(pred, list) else pred["label"].lower()
                    group = self._EMOTION_TO_GROUP.get(label, "neutral")
                    results.append(self.GROUP_IDS[group])
            except Exception as exc:
                logger.warning(f"  [L2] Emotion batch failed: {exc}")
                results.extend(["E3"] * len(batch))

        return results

    @staticmethod
    def human_label(short_id: str) -> str:
        inv = {v: k for k, v in Layer2GoEmotions.GROUP_IDS.items()}
        return inv.get(short_id, short_id)


class Layer3Demographic:
    """
    Demographic signal inference (gender proxy via pronoun analysis).
    m3inference requires Twitter-style JSON; we fall back to a pronoun
    heuristic for general Wikipedia text, which is more reliable.
    """

    PREFIX = "D"
    LABELS = {"male": "D1", "female": "D2", "unknown": "D3"}

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def _infer_gender(self, text: str) -> str:
        text_lower = text.lower()
        male_pronouns   = len(re.findall(r'\b(he|his|him|himself)\b', text_lower))
        female_pronouns = len(re.findall(r'\b(she|her|hers|herself)\b', text_lower))
        if male_pronouns == 0 and female_pronouns == 0:
            return "unknown"
        if male_pronouns > female_pronouns * 1.5:
            return "male"
        if female_pronouns > male_pronouns * 1.5:
            return "female"
        return "unknown"

    def predict(self, texts: List[str]) -> List[str]:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        labels_arr: List[str] = [self.LABELS["unknown"]] * len(texts)
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_to_idx = {executor.submit(self._infer_gender, text): idx for idx, text in enumerate(texts)}
            for future in tqdm(as_completed(future_to_idx), total=len(texts), desc="  [L3] Demographics (Multi-CPU)"):
                idx = future_to_idx[future]
                try:
                    gender = future.result()
                    labels_arr[idx] = self.LABELS[gender]
                except Exception:
                    labels_arr[idx] = self.LABELS["unknown"]
                    
        return labels_arr

    @staticmethod
    def human_label(short_id: str) -> str:
        inv = {v: k for k, v in Layer3Demographic.LABELS.items()}
        return inv.get(short_id, short_id)


class Layer4Dialect:
    """
    Language register / formality classification.
    Uses a fine-tuned CoLA acceptability model as a formality proxy,
    or falls back to heuristic-based register detection.
    """

    PREFIX = "R"
    LABELS = {"formal": "R1", "informal": "R2"}

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg  = cfg
        self.pipe = None

    def _load(self) -> None:
        if self.pipe is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"  [L4] Loading formality model on {DEVICE} …")
            self.pipe = hf_pipeline(
                "text-classification",
                model      = self.cfg.get("model", "textattack/bert-base-uncased-CoLA"),
                device     = DEVICE,
                truncation = True,
                max_length = 512,
            )
        except Exception as exc:
            logger.warning(f"  [L4] Could not load formality model: {exc}. Using heuristic.")

    def _heuristic_formal(self, text: str) -> bool:
        """Simple heuristic: low contraction ratio → formal."""
        contractions = len(re.findall(
            r"\b(don't|doesn't|can't|won't|isn't|aren't|wasn't|weren't|"
            r"haven't|hasn't|hadn't|wouldn't|couldn't|shouldn't|"
            r"i'm|i've|i'll|i'd|you're|they're|we're|it's)\b",
            text.lower()
        ))
        words = len(text.split())
        ratio = contractions / max(words, 1)
        avg_word_len = sum(len(w) for w in text.split()) / max(words, 1)
        return ratio < 0.02 and avg_word_len > 4.5

    def predict(self, texts: List[str]) -> List[str]:
        self._load()
        batch_size = 32
        labels = []

        for i in tqdm(range(0, len(texts), batch_size),
                      desc="  [L4] Formality", leave=False):
            batch = texts[i : i + batch_size]
            if self.pipe:
                try:
                    preds = self.pipe(batch)
                    for pred in preds:
                        label = pred["label"] if isinstance(pred, dict) else pred[0]["label"]
                        # CoLA: LABEL_1=acceptable(formal), LABEL_0=unacceptable(informal)
                        labels.append("R1" if "1" in label else "R2")
                    continue
                except Exception as exc:
                    logger.warning(f"  [L4] Formality batch failed: {exc}, using heuristic")
            for text in batch:
                labels.append("R1" if self._heuristic_formal(text) else "R2")

        return labels

    @staticmethod
    def human_label(short_id: str) -> str:
        inv = {v: k for k, v in Layer4Dialect.LABELS.items()}
        return inv.get(short_id, short_id)


class Layer5Readability:
    """Readability / complexity via Flesch Reading Ease score."""

    PREFIX = "C"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.bins = cfg.get("bins", [
            {"label": "Very Easy",      "min": 90,  "max": 100},
            {"label": "Easy",           "min": 70,  "max": 90},
            {"label": "Standard",       "min": 50,  "max": 70},
            {"label": "Difficult",      "min": 30,  "max": 50},
            {"label": "Very Difficult", "min": 0,   "max": 30},
        ])
        # Build short_id map
        self.label_to_id: Dict[str, str] = {}
        for i, b in enumerate(self.bins, start=1):
            self.label_to_id[b["label"]] = f"C{i}"

    def _score_to_label(self, score: float) -> str:
        for b in self.bins:
            if b["min"] <= score < b["max"]:
                return b["label"]
        return self.bins[-1]["label"]   # clamp to hardest

    def _score_text(self, text: str) -> str:
        try:
            score = textstat.flesch_reading_ease(text)
        except Exception:
            score = 50.0
        return self._score_to_label(score)

    def predict(self, texts: List[str]) -> List[str]:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        labels_arr = [self.label_to_id["Standard"]] * len(texts)
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_to_idx = {executor.submit(self._score_text, text): idx for idx, text in enumerate(texts)}
            for future in tqdm(as_completed(future_to_idx), total=len(texts), desc="  [L5] Readability (Multi-CPU)"):
                idx = future_to_idx[future]
                try:
                    label = future.result()
                    labels_arr[idx] = self.label_to_id.get(label, "C3")
                except Exception:
                    labels_arr[idx] = "C3"
                    
        return labels_arr

    def human_label(self, short_id: str) -> str:
        inv = {v: k for k, v in self.label_to_id.items()}
        return inv.get(short_id, short_id)


# ────────────────────── majority score ──────────────────────────

def compute_majority_scores(leaf_nodes: Dict[str, LeafNode]) -> None:
    counts = np.array([n.document_count for n in leaf_nodes.values()], dtype=float)
    mu     = counts.mean()
    sigma  = counts.std() if counts.std() > 0 else 1.0
    for node in leaf_nodes.values():
        node.majority_score = (node.document_count - mu) / sigma


# ─────────────────────── tree builder ───────────────────────────

LAYER_NAMES = ["topic", "emotion", "demographic", "register", "readability"]
LAYER_PREFIXES = ["T", "E", "D", "R", "C"]


class HierarchicalTree:
    """
    Builds the 5-layer stratification tree and stores results.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg       = cfg
        self.leaf_min  = cfg.get("leaf_min_size", 30)

        # Instantiate all layers
        self.layers = [
            Layer1BERTopic(cfg.get("bertopic", {})),
            Layer2GoEmotions(cfg.get("goemotions", {})),
            Layer3Demographic(cfg.get("m3inference", {})),
            Layer4Dialect(cfg.get("dialect", {})),
            Layer5Readability(cfg.get("readability", {})),
        ]
        self.layer_models = self.layers   # alias

        # Tree structure
        self.root: TreeNode = TreeNode(
            node_id  = "ROOT",
            layer    = 0,
            label    = "All Chunks",
            short_id = "ROOT",
            parent_id = None,
        )
        self.all_nodes: Dict[str, TreeNode] = {"ROOT": self.root}
        self.leaf_nodes: Dict[str, LeafNode] = {}

    # ── label assignment ────────────────────────────────────────

    def _assign_labels(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[List[str]], List[Dict[str, str]]]:
        """
        Run all 5 layers and return:
          assignments[chunk_idx] = [L1_id, L2_id, L3_id, L4_id, L5_id]
        """
        import os
        checkpoint_dir = Path("data/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        texts = [c["text"] for c in chunks]
        n     = len(texts)

        logger.info(f"Running Layer 1/5: BERTopic on {n} chunks …")
        l1_file = checkpoint_dir / "l1_outputs.json"
        if l1_file.exists():
            l1 = json.loads(l1_file.read_text())
        else:
            l1 = self.layers[0].fit_predict(texts)
            l1_file.write_text(json.dumps(l1))

        logger.info(f"Running Layer 2/5: GoEmotions on {n} chunks …")
        l2_file = checkpoint_dir / "l2_outputs.json"
        if l2_file.exists():
            l2 = json.loads(l2_file.read_text())
        else:
            l2 = self.layers[1].predict(texts)
            l2_file.write_text(json.dumps(l2))

        logger.info(f"Running Layer 3/5: Demographics on {n} chunks …")
        l3_file = checkpoint_dir / "l3_outputs.json"
        if l3_file.exists():
            l3 = json.loads(l3_file.read_text())
        else:
            l3 = self.layers[2].predict(texts)
            l3_file.write_text(json.dumps(l3))

        logger.info(f"Running Layer 4/5: Formality on {n} chunks …")
        l4_file = checkpoint_dir / "l4_outputs.json"
        if l4_file.exists():
            l4 = json.loads(l4_file.read_text())
        else:
            l4 = self.layers[3].predict(texts)
            l4_file.write_text(json.dumps(l4))

        logger.info(f"Running Layer 5/5: Readability on {n} chunks …")
        l5_file = checkpoint_dir / "l5_outputs.json"
        if l5_file.exists():
            l5 = json.loads(l5_file.read_text())
        else:
            l5 = self.layers[4].predict(texts)
            l5_file.write_text(json.dumps(l5))

        assignments = list(zip(l1, l2, l3, l4, l5))
        return assignments

    # ── tree construction ────────────────────────────────────────

    def _get_or_create_node(
        self,
        short_id: str,
        layer: int,
        parent_id: str,
        human_label: str,
    ) -> TreeNode:
        node_id = f"{parent_id}_{short_id}"
        if node_id not in self.all_nodes:
            node = TreeNode(
                node_id   = node_id,
                layer     = layer,
                label     = human_label,
                short_id  = short_id,
                parent_id = parent_id,
            )
            self.all_nodes[node_id] = node
            parent = self.all_nodes[parent_id]
            parent.children.append(node)
        return self.all_nodes[node_id]

    def _human_label_for(self, layer_idx: int, short_id: str) -> str:
        layer = self.layers[layer_idx]
        if hasattr(layer, "human_label"):
            return layer.human_label(short_id)
        return short_id

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        assignments = self._assign_labels(chunks)

        # Add all chunks to root
        for c in chunks:
            self.root.chunk_ids.append(c["chunk_id"])

        # Build tree level by level
        for chunk_idx, (chunk, layer_ids) in enumerate(
            tqdm(zip(chunks, assignments), total=len(chunks), desc="Building tree")
        ):
            parent_node = self.root
            for layer_idx, short_id in enumerate(layer_ids):
                human = self._human_label_for(layer_idx, short_id)
                node  = self._get_or_create_node(
                    short_id   = short_id,
                    layer      = layer_idx + 1,
                    parent_id  = parent_node.node_id,
                    human_label = human,
                )
                node.chunk_ids.append(chunk["chunk_id"])
                parent_node = node

            # The deepest node is the leaf
            leaf_node = parent_node
            leaf_path = " > ".join(str(lid) for lid in layer_ids)
            leaf_id   = leaf_node.node_id

            if leaf_id not in self.leaf_nodes:
                self.leaf_nodes[leaf_id] = LeafNode(
                    leaf_id      = leaf_id,
                    label_path   = leaf_path,
                    layer_labels = dict(zip(LAYER_NAMES, layer_ids)),
                )
            self.leaf_nodes[leaf_id].chunk_ids.append(chunk["chunk_id"])
            self.leaf_nodes[leaf_id].document_count += 1

        # Compute majority scores
        compute_majority_scores(self.leaf_nodes)
        logger.success(f"Tree built: {len(self.all_nodes)} nodes, {len(self.leaf_nodes)} leaves")

    # ── output ───────────────────────────────────────────────────

    def save(
        self,
        chunks: List[Dict[str, Any]],
        assignments: Optional[List] = None,
        out_dir: str = "data/tree",
    ) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Build chunk_id → leaf_id lookup
        chunk_to_leaf: Dict[int, str] = {}
        for leaf_id, leaf in self.leaf_nodes.items():
            for cid in leaf.chunk_ids:
                chunk_to_leaf[cid] = leaf_id

        # leaf_assignments.jsonl
        assign_path = out_path / "leaf_assignments.jsonl"
        with open(assign_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                cid     = chunk["chunk_id"]
                leaf_id = chunk_to_leaf.get(cid, "UNASSIGNED")
                leaf    = self.leaf_nodes.get(leaf_id)
                record  = {
                    "chunk_id":      cid,
                    "leaf_id":       leaf_id,
                    "label_path":    leaf.label_path if leaf else "",
                    "majority_score": leaf.majority_score if leaf else 0.0,
                    "topic_name":    chunk.get("topic_name", ""),
                    "group":         chunk.get("group", ""),
                }
                f.write(json.dumps(record) + "\n")

        # leaf_nodes.json
        leaf_path = out_path / "leaf_nodes.json"
        serializable = {}
        for lid, leaf in self.leaf_nodes.items():
            d = asdict(leaf)
            d.pop("chunk_ids")      # keep file small
            serializable[lid] = d
        with open(leaf_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        logger.success(f"Saved tree data → {out_path}")


# ──────────────────────── visualization ─────────────────────────

LAYER_COLORS = {
    0: "#4a4a8a",   # ROOT      – deep indigo
    1: "#2196F3",   # Topic     – blue
    2: "#FF9800",   # Emotion   – orange
    3: "#9C27B0",   # Demo      – purple
    4: "#009688",   # Register  – teal
    5: "#F44336",   # Readability – red
}

LAYER_DISPLAY_NAMES = {
    0: "Root",
    1: "Topic (BERTopic)",
    2: "Emotion (GoEmotions)",
    3: "Demographic (M3)",
    4: "Register (Dialect)",
    5: "Readability (CLEAR)",
}


def _truncate(s: str, n: int = 30) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def visualize_tree(
    tree: HierarchicalTree,
    out_dir: str = "outputs/tree_viz",
    dpi: int = 150,
    max_label_len: int = 40,
    fmt: str = "png",
) -> None:
    """
    Produce:
      1. Full tree image (all layers)
      2. One image per layer showing that layer's split
    """
    viz_dir = Path(out_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Build networkx graph ──────────────────────────────────
    G = nx.DiGraph()

    def _node_label(node: TreeNode) -> str:
        parts = [node.short_id]
        if node.label and node.label != node.short_id:
            parts.append(_truncate(node.label, max_label_len))
        n_chunks = len(node.chunk_ids)
        parts.append(f"n={n_chunks}")
        return "\n".join(parts)

    def _add_subtree(node: TreeNode) -> None:
        G.add_node(
            node.node_id,
            display = _node_label(node),
            layer   = node.layer,
            short_id = node.short_id,
        )
        for child in node.children:
            _add_subtree(child)
            G.add_edge(node.node_id, child.node_id)

    _add_subtree(tree.root)

    # ── Hierarchical layout ───────────────────────────────────
    def _hierarchy_pos(
        G: nx.DiGraph,
        root: str,
        width: float = 1.0,
        vert_gap: float = 1.5,
        vert_loc: float = 0.0,
        xcenter: float = 0.5,
    ) -> Dict[str, Tuple[float, float]]:
        """Recursively compute (x, y) for tree layout."""
        pos: Dict[str, Tuple[float, float]] = {}

        def _recurse(node: str, left: float, right: float, y: float) -> None:
            pos[node] = ((left + right) / 2.0, y)
            children  = list(G.successors(node))
            if children:
                dx    = (right - left) / len(children)
                for i, child in enumerate(children):
                    _recurse(child, left + i * dx, left + (i + 1) * dx, y - vert_gap)

        _recurse(root, 0.0, width, vert_loc)
        return pos

    # ── Full tree ─────────────────────────────────────────────
    logger.info("  Rendering full tree image …")
    n_nodes = G.number_of_nodes()
    fig_w   = max(20, n_nodes * 0.3)
    fig_h   = max(12, (tree.cfg.get("max_depth", 5) + 1) * 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("#f8f8f8")

    pos = _hierarchy_pos(G, "ROOT", width=fig_w)

    node_colors = [LAYER_COLORS.get(G.nodes[n].get("layer", 0), "#888888") for n in G.nodes]
    labels_map  = {n: G.nodes[n].get("display", n) for n in G.nodes}

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowstyle="-|>", arrowsize=12,
                           edge_color="#aaaaaa", alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=800, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=labels_map, ax=ax,
                            font_size=5, font_color="white", font_weight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=LAYER_COLORS[i], label=LAYER_DISPLAY_NAMES[i])
        for i in range(6)
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              framealpha=0.9)

    ax.set_title("Hierarchical Stratification Tree — Full", fontsize=14, pad=10)
    ax.axis("off")
    plt.tight_layout()
    full_path = viz_dir / f"tree_full.{fmt}"
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  Saved: {full_path}")

    # ── Per-layer images ──────────────────────────────────────
    for layer_idx in range(1, 6):
        layer_nodes = [
            n for n in G.nodes if G.nodes[n].get("layer") == layer_idx
        ]
        if not layer_nodes:
            continue

        logger.info(f"  Rendering layer {layer_idx} image …")
        # Subgraph: root + this layer's nodes + edges connecting them
        sub_nodes = {"ROOT"} | set(layer_nodes)
        # Include intermediate ancestors up to root
        for ln in layer_nodes:
            path = nx.shortest_path(G, "ROOT", ln)
            sub_nodes.update(path)

        SG  = G.subgraph(sub_nodes)
        n_sub = SG.number_of_nodes()
        fig_w2 = max(16, n_sub * 0.5)
        fig_h2 = max(8, layer_idx * 2.5)

        fig2, ax2 = plt.subplots(figsize=(fig_w2, fig_h2))
        ax2.set_facecolor("#f8f8f8")
        fig2.patch.set_facecolor("#f8f8f8")

        pos2 = _hierarchy_pos(SG, "ROOT", width=fig_w2)
        colors2 = [LAYER_COLORS.get(SG.nodes[n].get("layer", 0), "#888") for n in SG.nodes]
        labels2 = {n: SG.nodes[n].get("display", n) for n in SG.nodes}

        nx.draw_networkx_edges(SG, pos2, ax=ax2, arrows=True,
                               arrowstyle="-|>", arrowsize=12,
                               edge_color="#aaaaaa", alpha=0.7)
        nx.draw_networkx_nodes(SG, pos2, ax=ax2,
                               node_color=colors2, node_size=900, alpha=0.95)
        nx.draw_networkx_labels(SG, pos2, labels=labels2, ax=ax2,
                                font_size=6, font_color="white", font_weight="bold")

        layer_name = LAYER_DISPLAY_NAMES.get(layer_idx, f"Layer {layer_idx}")
        ax2.set_title(f"Layer {layer_idx}: {layer_name}", fontsize=13, pad=10)
        ax2.axis("off")
        plt.tight_layout()
        img_path = viz_dir / f"tree_layer{layer_idx}.{fmt}"
        fig2.savefig(img_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        logger.success(f"  Saved: {img_path}")


# ────────────────────────── validation ──────────────────────────

def validate_tree(
    tree: HierarchicalTree,
    chunks: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> bool:
    logger.info("=" * 60)
    logger.info("STAGE 1 VALIDATION")
    logger.info("=" * 60)
    passed = True
    leaf_min = cfg.get("leaf_min_size", 30)

    # 1. Every chunk is assigned to exactly one leaf
    all_chunk_ids    = {c["chunk_id"] for c in chunks}
    assigned_ids: Set[int] = set()
    for leaf in tree.leaf_nodes.values():
        for cid in leaf.chunk_ids:
            assigned_ids.add(cid)

    unassigned = all_chunk_ids - assigned_ids
    status = "✓" if not unassigned else "✗"
    logger.info(f"  {status} Unassigned chunks: {len(unassigned)}")
    if unassigned:
        passed = False

    duplicate_assign = len(assigned_ids) - len(set(assigned_ids))
    # (can't really get duplicates from the set, just sanity)

    # 2. No leaf below hard floor (warn only — tree may legitimately have small leaves)
    tiny_leaves = [
        (lid, l.document_count)
        for lid, l in tree.leaf_nodes.items()
        if l.document_count < leaf_min
    ]
    status = "⚠" if tiny_leaves else "✓"
    logger.info(
        f"  {status} Leaves below min_size ({leaf_min}): {len(tiny_leaves)}"
    )
    if tiny_leaves:
        for lid, cnt in tiny_leaves[:5]:
            logger.info(f"       {lid}: {cnt} chunks")

    # 3. Majority scores computed
    no_score = [lid for lid, l in tree.leaf_nodes.items() if l.majority_score == 0.0 and l.document_count > 1]
    status = "✓" if not no_score else "⚠"
    logger.info(f"  {status} Leaves with score=0 (possible issue): {len(no_score)}")

    # 4. Tree has all 5 layers (at least one node at each layer)
    for layer_idx in range(1, 6):
        nodes_at_layer = [
            n for n in tree.all_nodes.values() if n.layer == layer_idx
        ]
        ok = len(nodes_at_layer) >= 1
        status = "✓" if ok else "✗"
        layer_name = LAYER_DISPLAY_NAMES.get(layer_idx, f"Layer {layer_idx}")
        logger.info(f"  {status} Layer {layer_idx} ({layer_name}): {len(nodes_at_layer)} nodes")
        if not ok:
            passed = False

    # 5. label_path format check
    bad_paths = [
        lid for lid, l in tree.leaf_nodes.items()
        if len(l.label_path.split(" > ")) != 5
    ]
    status = "✓" if not bad_paths else "✗"
    logger.info(f"  {status} Leaves with malformed label_path: {len(bad_paths)}")
    if bad_paths:
        passed = False

    # 6. Majority / minority distribution
    majority_leaves = [l for l in tree.leaf_nodes.values() if l.majority_score > 1.0]
    minority_leaves = [l for l in tree.leaf_nodes.values() if l.majority_score < -0.5]
    logger.info(f"\n  Total leaf nodes       : {len(tree.leaf_nodes)}")
    logger.info(f"  Majority leaves (s>1)  : {len(majority_leaves)}")
    logger.info(f"  Minority leaves (s<-0.5): {len(minority_leaves)}")
    logger.info(f"  Total tree nodes       : {len(tree.all_nodes)}")

    # 7. Sample leaf paths
    logger.info("\n  Sample leaf paths:")
    for lid, leaf in list(tree.leaf_nodes.items())[:5]:
        logger.info(
            f"    [{lid}] path={leaf.label_path}  "
            f"count={leaf.document_count}  score={leaf.majority_score:.2f}"
        )

    result = "PASSED" if passed else "FAILED"
    logger.info(f"\nStage 1 Validation: {result}")
    logger.info("=" * 60)
    return passed


# from typing (needed above)
from typing import Set


# ────────────────────────── prune tree ──────────────────────────

def prune_tree(
    pruned_chunks_path: str = "data/pruned/pruned_chunks.jsonl",
    orig_nodes_path: str = "data/tree/leaf_nodes.json",
    out_dir: str = "data/pruned/tree",
) -> None:
    """
    Build a pruned copy of the tree using only the chunks that survived
    scripts/prune_chunks.py. Writes to `out_dir` (default: data/pruned/tree)
    so the original data/tree/ is never overwritten.

    pruned_chunks.jsonl already contains leaf_id and label_path from the
    original tree — no need to re-join with leaf_assignments.jsonl.

    Steps:
      1. Read pruned_chunks.jsonl; group by leaf_id
      2. Recompute document_count per leaf + majority_score (z-score)
      3. Load original leaf_nodes.json for static metadata (layer_labels, etc.)
      4. Emit data/pruned/tree/leaf_assignments.jsonl
      5. Emit data/pruned/tree/leaf_nodes.json
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Read pruned chunks, group by leaf_id ──────────────────
    logger.info(f"Loading pruned chunks from {pruned_chunks_path} …")
    # leaf_id → list of (chunk record) dicts
    leaf_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total = 0
    with open(pruned_chunks_path, encoding="utf-8") as fh:
        for line in fh:
            total += 1
            row = json.loads(line)
            leaf_rows[row["leaf_id"]].append(row)
    kept = sum(len(v) for v in leaf_rows.values())
    logger.info(f"  {total:,} chunks → {kept:,} rows in {len(leaf_rows):,} leaves")

    # ── 2. Recompute majority scores (z-score of leaf document counts) ──
    counts = np.array([len(v) for v in leaf_rows.values()], dtype=np.float64)
    mu  = float(counts.mean()) if len(counts) > 0 else 0.0
    std = float(counts.std())  if len(counts) > 1 else 1.0
    if std == 0.0:
        std = 1.0

    leaf_new_score: Dict[str, float] = {}
    for lid, rows in leaf_rows.items():
        leaf_new_score[lid] = (len(rows) - mu) / std

    # ── 3. Load original leaf_nodes for static metadata ──────────
    logger.info(f"Loading original leaf metadata from {orig_nodes_path} …")
    with open(orig_nodes_path, encoding="utf-8") as fh:
        orig_nodes: Dict[str, Any] = json.load(fh)

    # ── 4. Write leaf_assignments.jsonl ──────────────────────────
    assign_out = out_path / "leaf_assignments.jsonl"
    logger.info(f"Writing {assign_out} …")
    with open(assign_out, "w", encoding="utf-8") as fh:
        for lid, rows in leaf_rows.items():
            score = leaf_new_score[lid]
            for row in rows:
                rec = {
                    "chunk_id":       row["_gidx"],          # globally unique id
                    "leaf_id":        lid,
                    "label_path":     row.get("label_path", ""),
                    "majority_score": score,
                    "topic_name":     row.get("topic_name", ""),
                    "group":          row.get("group", ""),
                }
                fh.write(json.dumps(rec) + "\n")
    logger.info(f"  Wrote {kept:,} rows")

    # ── 5. Write leaf_nodes.json ─────────────────────────────────
    nodes_out = out_path / "leaf_nodes.json"
    logger.info(f"Writing {nodes_out} …")
    pruned_nodes: Dict[str, Any] = {}
    for lid, rows in leaf_rows.items():
        orig = orig_nodes.get(lid, {})
        pruned_nodes[lid] = {
            "leaf_id":           lid,
            "label_path":        orig.get("label_path", rows[0].get("label_path", "")),
            "layer_labels":      orig.get("layer_labels", {}),
            "document_count":    len(rows),
            "majority_score":    leaf_new_score[lid],
            "cosine_similarity": orig.get("cosine_similarity", 0.0),
        }
    with open(nodes_out, "w", encoding="utf-8") as fh:
        json.dump(pruned_nodes, fh, indent=2)
    logger.info(f"  Wrote {len(pruned_nodes):,} leaf nodes")

    # ── Summary ──────────────────────────────────────────────────
    logger.success(
        f"Pruned tree saved to {out_path}  "
        f"({kept:,} assignments, {len(pruned_nodes):,} leaves)"
    )


# ────────────────────────── pipeline ────────────────────────────

class Stage1Pipeline:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg         = load_config(config_path)
        self.tree_cfg    = self.cfg["tree"]
        self.viz_cfg     = self.tree_cfg.get("visualization", {})

    def run(self, chunks_path: str = "data/chunks/chunks.jsonl") -> HierarchicalTree:
        logger.info("=" * 60)
        logger.info("STAGE 1 — Hierarchical Stratification Tree")
        logger.info("=" * 60)

        chunks = load_chunks(chunks_path)
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

        # Optional: grid search calibration sample (informational only here)
        sample_frac = self.tree_cfg.get("grid_search_sample_fraction", 0.15)
        sample_size = max(50, int(len(chunks) * sample_frac))
        logger.info(
            f"Grid-search calibration would use {sample_size} chunks "
            f"({sample_frac*100:.0f}% of corpus) — running on full set for now."
        )

        # Build tree
        tree = HierarchicalTree(self.tree_cfg)
        tree.build(chunks)

        # Save data
        tree.save(chunks, out_dir="data/tree")

        # Visualize
        visualize_tree(
            tree,
            out_dir     = self.viz_cfg.get("output_dir", "outputs/tree_viz"),
            dpi         = self.viz_cfg.get("dpi", 150),
            max_label_len = self.viz_cfg.get("max_label_length", 40),
            fmt         = self.viz_cfg.get("image_format", "png"),
        )

        # Validate
        validate_tree(tree, chunks, self.tree_cfg)

        return tree


# ────────────────────────────── main ────────────────────────────

if __name__ == "__main__":
    import sys

    if "--prune" in sys.argv:
        # Prune-tree mode: reproject pruned chunks onto the existing tree
        # Usage: python stage1_hierarchical_tree.py [config.yaml] --prune
        #          [--pruned-chunks PATH] [--orig-assignments PATH]
        #          [--orig-nodes PATH]   [--out-dir PATH]
        args = sys.argv[1:]
        def _flag(name, default):
            try:
                return args[args.index(name) + 1]
            except (ValueError, IndexError):
                return default

        prune_tree(
            pruned_chunks_path = _flag("--pruned-chunks", "data/pruned/pruned_chunks.jsonl"),
            orig_nodes_path    = _flag("--orig-nodes",    "data/tree/leaf_nodes.json"),
            out_dir            = _flag("--out-dir",        "data/pruned/tree"),
        )
    else:
        config_path  = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        chunks_path  = sys.argv[2] if len(sys.argv) > 2 else "data/chunks/chunks.jsonl"

        pipeline = Stage1Pipeline(config_path)
        tree     = pipeline.run(chunks_path)

        print(f"\nStage 1 complete.")
        print(f"  Leaf nodes : {len(tree.leaf_nodes)}")
        print(f"  Tree nodes : {len(tree.all_nodes)}")
        print(f"  Tree images saved to: {pipeline.viz_cfg.get('output_dir', 'outputs/tree_viz')}")
