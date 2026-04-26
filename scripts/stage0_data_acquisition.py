"""
Stage 0: Data Acquisition & Preprocessing
==========================================
- Fetches Wikipedia pages per category topic (async, rate-limited)
- Chunks each article into paragraph-level segments (200–400 tokens)
- Deduplicates chunks using MinHash (Jaccard similarity threshold)
- Writes chunks as JSONL with metadata
- Runs validation checks at the end
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import yaml
from aiolimiter import AsyncLimiter
from datasketch import MinHash, MinHashLSH
from loguru import logger
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# ─────────────────────────── helpers ────────────────────────────

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def slugify(text: str) -> str:
    """Simple ASCII slug for directory names."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "_", text)


def token_count(text: str) -> int:
    """Approximate token count (whitespace split)."""
    return len(text.split())


# ────────────────────────── Wikipedia client ────────────────────

class WikipediaClient:
    """Async client for the Wikipedia Action API."""

    DEFAULT_API = "https://en.wikipedia.org/w/api.php"
    # Wikimedia requires a descriptive UA with contact info to avoid 403s
    DEFAULT_UA  = (
        "DeBiasingResearchPipeline/0.1 "
        "(https://github.com/research/debiasing-llm; usamaahmedsh@gmail.com)"
    )

    def __init__(
        self,
        api_url: str = DEFAULT_API,
        user_agent: str = DEFAULT_UA,
        timeout: float = 30.0,
        max_concurrent: int = 5,
        rate_limit: float = 2.0,
    ) -> None:
        self.api_url    = api_url
        self.user_agent = user_agent
        self.timeout    = aiohttp.ClientTimeout(total=timeout)
        self.semaphore  = asyncio.Semaphore(max_concurrent)
        self.limiter    = AsyncLimiter(max_rate=rate_limit, time_period=1.0)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "WikipediaClient":
        headers = {
            "User-Agent": self.user_agent,
            "Api-User-Agent": self.user_agent,
        }
        connector = aiohttp.TCPConnector(limit=10, ssl=False)
        self.session = aiohttp.ClientSession(
            headers=headers, timeout=self.timeout, connector=connector
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self.session:
            await self.session.close()

    async def _get(self, params: Dict, max_retries: int = 5) -> Dict:
        """Rate-limited GET with exponential backoff on 429/403."""
        params.setdefault("format", "json")
        for attempt in range(max_retries):
            async with self.semaphore:
                async with self.limiter:
                    try:
                        async with self.session.get(
                            self.api_url, params=params
                        ) as resp:
                            if resp.status in (429, 403):
                                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                                logger.warning(
                                    f"Rate limited ({resp.status}), "
                                    f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})"
                                )
                                await asyncio.sleep(wait)
                                continue
                            resp.raise_for_status()
                            return await resp.json()
                    except aiohttp.ClientError as exc:
                        wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                        logger.warning(f"Request error: {exc}, retrying in {wait:.1f}s")
                        await asyncio.sleep(wait)
        raise RuntimeError(f"All {max_retries} retries exhausted for params: {params}")

    async def get_category_members(
        self,
        category: str,
        depth: int = 2,
        max_pages: int = 50,
    ) -> Set[str]:
        """
        Recursively collect article titles from a Wikipedia category.
        Follows sub-categories up to `depth` levels.
        """
        titles: Set[str] = set()
        visited_cats: Set[str] = set()

        async def _recurse(cat: str, remaining_depth: int) -> None:
            if remaining_depth < 0 or cat in visited_cats:
                return
            visited_cats.add(cat)
            if len(titles) >= max_pages:
                return

            cmcontinue: Optional[str] = None
            while len(titles) < max_pages:
                params: Dict[str, Any] = {
                    "action":  "query",
                    "list":    "categorymembers",
                    "cmtitle": cat,
                    "cmlimit": min(50, max_pages - len(titles)),
                    "cmtype":  "page|subcat",
                }
                if cmcontinue:
                    params["cmcontinue"] = cmcontinue

                try:
                    data = await self._get(params)
                except Exception as exc:
                    logger.warning(f"API error for {cat}: {exc}")
                    break

                members = data.get("query", {}).get("categorymembers", [])
                subcats: List[str] = []
                for m in members:
                    ns = m.get("ns", 0)
                    title = m.get("title", "")
                    if ns == 0:                      # article
                        titles.add(title)
                    elif ns == 14:                   # sub-category
                        subcats.append(title)

                # Recurse into sub-cats concurrently
                if subcats and remaining_depth > 0:
                    await asyncio.gather(*[
                        _recurse(sc, remaining_depth - 1) for sc in subcats
                    ])

                cont = data.get("continue", {})
                cmcontinue = cont.get("cmcontinue")
                if not cmcontinue:
                    break

        await _recurse(category, depth)
        return titles

    async def get_page_text(self, title: str) -> Optional[str]:
        """Fetch the plain-text extract of a Wikipedia article."""
        params = {
            "action":       "query",
            "titles":       title,
            "prop":         "extracts",
            "explaintext":  1,
            "exsectionformat": "plain",
        }
        try:
            data = await self._get(params)
        except Exception as exc:
            logger.warning(f"Failed to fetch '{title}': {exc}")
            return None

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                return None
            return page.get("extract", "").strip() or None
        return None


# ─────────────────────────── chunking ───────────────────────────

def chunk_article(
    text: str,
    min_tokens: int = 200,
    max_tokens: int = 400,
) -> List[str]:
    """
    Split an article into paragraph-level chunks.
    Each chunk targets min_tokens–max_tokens (whitespace tokens).
    Paragraphs are merged until the window is large enough, then split
    when it would exceed max_tokens.
    """
    # Split on blank lines (Wikipedia paragraph separators)
    raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: List[str] = []
    buffer: List[str] = []
    buffer_tokens = 0

    for para in raw_paragraphs:
        para_tokens = token_count(para)

        # If a single paragraph is already too long, hard-split it
        if para_tokens > max_tokens:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer, buffer_tokens = [], 0
            words = para.split()
            for i in range(0, len(words), max_tokens):
                chunk = " ".join(words[i : i + max_tokens])
                if token_count(chunk) >= min_tokens or not chunks:
                    chunks.append(chunk)
            continue

        # Would adding this para exceed max_tokens?
        if buffer_tokens + para_tokens > max_tokens and buffer_tokens >= min_tokens:
            chunks.append(" ".join(buffer))
            buffer, buffer_tokens = [], 0

        buffer.append(para)
        buffer_tokens += para_tokens

    if buffer and buffer_tokens >= min_tokens:
        chunks.append(" ".join(buffer))
    elif buffer and chunks:
        # Merge tiny trailing buffer into last chunk
        chunks[-1] += " " + " ".join(buffer)

    return [c for c in chunks if token_count(c) >= 50]  # drop very short fragments


# ──────────────────────── deduplication ─────────────────────────

def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for token in set(text.lower().split()):
        m.update(token.encode("utf8"))
    return m


def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remove near-duplicate chunks using MinHash LSH.
    Returns (deduplicated_list, n_removed).
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: List[Dict[str, Any]] = []
    removed = 0

    for i, chunk in enumerate(chunks):
        key = f"chunk_{i}"
        mh = build_minhash(chunk["text"], num_perm=num_perm)
        try:
            result = lsh.query(mh)
            if result:
                removed += 1
                continue
            lsh.insert(key, mh)
            kept.append(chunk)
        except Exception:
            kept.append(chunk)

    return kept, removed


# ─────────────────────────── pipeline ───────────────────────────

class Stage0Pipeline:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg    = load_config(config_path)
        self.corpus = self.cfg["corpus"]
        self.topics = self.corpus["topics"]

        self.raw_dir    = Path(self.corpus["raw_dir"])
        self.chunks_dir = Path(self.corpus["chunks_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    # ── fetch ──────────────────────────────────────────────────

    async def fetch_topic(
        self,
        topic: Dict[str, Any],
        client: WikipediaClient,
    ) -> List[Dict[str, Any]]:
        """Collect pages_per_topic articles for a single topic."""
        max_pages  = self.corpus["pages_per_topic"]
        cat_depth  = self.corpus["category_depth"]
        topic_name = topic["name"]
        category   = topic["category"]
        group      = topic["group"]
        slug       = slugify(topic_name)

        logger.info(f"[{topic_name}] Discovering titles from '{category}' ...")
        titles = await client.get_category_members(
            category=category,
            depth=cat_depth,
            max_pages=max_pages * 3,   # over-fetch then trim
        )
        titles = list(titles)[:max_pages]
        logger.info(f"[{topic_name}] Found {len(titles)} titles, downloading ...")

        topic_dir = self.raw_dir / slug
        topic_dir.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []

        async def _download_one(title: str) -> Optional[Dict[str, Any]]:
            text = await client.get_page_text(title)
            if not text or len(text.split()) < 100:
                return None
            safe_name = slugify(title) + ".txt"
            (topic_dir / safe_name).write_text(text, encoding="utf-8")
            return {
                "topic_name": topic_name,
                "topic_slug": slug,
                "group":      group,
                "category":   category,
                "title":      title,
                "filename":   safe_name,
                "word_count": len(text.split()),
            }

        tasks = [_download_one(t) for t in titles]
        for coro in async_tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"  {topic_name}",
            leave=False,
        ):
            rec = await coro
            if rec:
                records.append(rec)

        # Write manifest
        manifest_path = topic_dir / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        logger.success(
            f"[{topic_name}] Downloaded {len(records)}/{len(titles)} pages → {topic_dir}"
        )
        return records

    async def fetch_all(self) -> List[Dict[str, Any]]:
        """Fetch all topics concurrently (shared client session)."""
        async with WikipediaClient(
            api_url        = self.corpus["api_url"],
            user_agent     = WikipediaClient.DEFAULT_UA,
            timeout        = self.corpus["request_timeout"],
            max_concurrent = self.corpus["max_concurrent_requests"],
            rate_limit     = self.corpus["rate_limit_per_second"],
        ) as client:
            all_records: List[Dict[str, Any]] = []
            # Run topics sequentially to avoid overwhelming the API;
            # within each topic downloads are fully concurrent
            for topic in self.topics:
                recs = await self.fetch_topic(topic, client)
                all_records.extend(recs)
            return all_records

    # ── chunk ──────────────────────────────────────────────────

    def chunk_corpus(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk every downloaded article and return flat chunk list."""
        min_tok = self.corpus["chunk_min_tokens"]
        max_tok = self.corpus["chunk_max_tokens"]

        all_chunks: List[Dict[str, Any]] = []
        chunk_id = 0

        for rec in tqdm(records, desc="Chunking articles"):
            topic_dir = self.raw_dir / rec["topic_slug"]
            page_path = topic_dir / rec["filename"]
            if not page_path.exists():
                logger.warning(f"Missing file: {page_path}")
                continue

            text = page_path.read_text(encoding="utf-8")
            chunks = chunk_article(text, min_tok, max_tok)

            for idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "chunk_id":   chunk_id,
                    "article_id": hashlib.md5(rec["title"].encode()).hexdigest()[:8],
                    "chunk_idx":  idx,
                    "topic_name": rec["topic_name"],
                    "topic_slug": rec["topic_slug"],
                    "group":      rec["group"],
                    "category":   rec["category"],
                    "title":      rec["title"],
                    "text":       chunk_text,
                    "token_count": token_count(chunk_text),
                })
                chunk_id += 1

        logger.info(f"Total chunks before dedup: {len(all_chunks)}")
        return all_chunks

    # ── dedup ──────────────────────────────────────────────────

    def deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        threshold = self.corpus["dedup_jaccard_threshold"]
        num_perm  = self.corpus["minhash_num_perm"]

        logger.info(f"Deduplicating {len(chunks)} chunks (threshold={threshold}) ...")
        kept, removed = deduplicate_chunks(chunks, threshold=threshold, num_perm=num_perm)
        logger.success(f"Dedup complete: kept {len(kept)}, removed {removed}")
        return kept

    # ── save ───────────────────────────────────────────────────

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> Path:
        out_path = self.chunks_dir / "chunks.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        logger.success(f"Saved {len(chunks)} chunks → {out_path}")
        return out_path

    # ── validate ───────────────────────────────────────────────

    def validate(self, records: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> bool:
        logger.info("=" * 60)
        logger.info("STAGE 0 VALIDATION")
        logger.info("=" * 60)
        passed = True

        # 1. Every topic has at least 1 page
        by_topic: Dict[str, List] = {}
        for r in records:
            by_topic.setdefault(r["topic_name"], []).append(r)

        for topic in self.topics:
            name  = topic["name"]
            pages = by_topic.get(name, [])
            ok    = len(pages) >= 1
            status = "✓" if ok else "✗"
            logger.info(f"  {status} Topic '{name}': {len(pages)} pages downloaded")
            if not ok:
                passed = False

        # 2. Raw files exist on disk
        missing_files = 0
        for r in records:
            p = self.raw_dir / r["topic_slug"] / r["filename"]
            if not p.exists():
                missing_files += 1
        status = "✓" if missing_files == 0 else "✗"
        logger.info(f"  {status} Missing raw files: {missing_files}")
        if missing_files > 0:
            passed = False

        # 3. All chunks have required fields
        required_keys = {"chunk_id", "text", "topic_name", "group", "token_count"}
        bad_chunks = [c for c in chunks if not required_keys.issubset(c.keys())]
        status = "✓" if not bad_chunks else "✗"
        logger.info(f"  {status} Chunks with missing fields: {len(bad_chunks)}")
        if bad_chunks:
            passed = False

        # 4. Token counts are within expected range (allow some slack)
        min_tok = self.corpus["chunk_min_tokens"]
        max_tok = self.corpus["chunk_max_tokens"]
        out_of_range = [
            c for c in chunks
            if c["token_count"] < min_tok * 0.5 or c["token_count"] > max_tok * 1.5
        ]
        status = "✓" if len(out_of_range) == 0 else "⚠"
        logger.info(
            f"  {status} Chunks outside token range "
            f"({min_tok}–{max_tok}): {len(out_of_range)}"
        )

        # 5. Both minority and majority groups present
        groups = {c["group"] for c in chunks}
        for g in ("minority", "majority"):
            ok = g in groups
            status = "✓" if ok else "✗"
            logger.info(f"  {status} Group '{g}' present in chunks: {ok}")
            if not ok:
                passed = False

        # 6. No duplicate chunk_ids
        ids = [c["chunk_id"] for c in chunks]
        dups = len(ids) - len(set(ids))
        status = "✓" if dups == 0 else "✗"
        logger.info(f"  {status} Duplicate chunk_ids: {dups}")
        if dups > 0:
            passed = False

        # 7. Summary stats
        total_tokens = sum(c["token_count"] for c in chunks)
        avg_tokens   = total_tokens / len(chunks) if chunks else 0
        logger.info(f"\n  Total pages  : {len(records)}")
        logger.info(f"  Total chunks : {len(chunks)}")
        logger.info(f"  Avg tokens/chunk : {avg_tokens:.1f}")
        by_group: Dict[str, int] = {}
        for c in chunks:
            by_group[c["group"]] = by_group.get(c["group"], 0) + 1
        for g, cnt in by_group.items():
            logger.info(f"  Chunks [{g}] : {cnt}")

        result = "PASSED" if passed else "FAILED"
        logger.info(f"\nStage 0 Validation: {result}")
        logger.info("=" * 60)
        return passed

    # ── run ────────────────────────────────────────────────────

    def run(self) -> Tuple[List[Dict[str, Any]], Path]:
        logger.info("=" * 60)
        logger.info("STAGE 0 — Data Acquisition & Preprocessing")
        logger.info("=" * 60)

        # Fetch
        records = asyncio.run(self.fetch_all())
        if not records:
            raise RuntimeError("No pages downloaded. Check your config and internet connection.")

        # Chunk
        chunks = self.chunk_corpus(records)

        # Deduplicate
        chunks = self.deduplicate(chunks)

        # Assign sequential chunk_ids after dedup
        for i, c in enumerate(chunks):
            c["chunk_id"] = i

        # Save
        chunks_path = self.save_chunks(chunks)

        # Validate
        self.validate(records, chunks)

        return chunks, chunks_path


# ────────────────────────────── main ────────────────────────────

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    pipeline = Stage0Pipeline(config_path)
    chunks, path = pipeline.run()
    print(f"\nStage 0 complete. {len(chunks)} chunks saved to {path}")
