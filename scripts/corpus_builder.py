#!/usr/bin/env python3
"""
Corpus builder: fetch a rich Wikipedia corpus for a user-specified topic/person.

- Asks user for topic/person (e.g. "Imran Khan").
- Discovers titles via:
    * Category:<seed> + subcategories (articles + nested categories)
    * Backlinks to the seed page
- Downloads plaintext extracts asynchronously.
- Writes:
    data/raw/<topic_slug>/pages/*.txt
    data/raw/<topic_slug>/manifest.jsonl
"""

import os
import re
import json
import asyncio
import argparse
import hashlib
from collections import deque
from typing import Dict, Iterable, Set, List, Optional, Any, Tuple
from pathlib import Path

import aiohttp
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm as async_tqdm
from datasketch import MinHash, MinHashLSH

# -----------------------------
# Defaults & global settings
# -----------------------------

DEFAULT_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "LocalLLMTopicPipeline/0.1 (research; contact: usamaahmedshus@gmail.com)"
DEFAULT_OUTPUT_DIR = "data/raw"

# More generous defaults so BERTopic has enough docs
DEFAULT_MAX_PAGES = 2000
DEFAULT_CAT_DEPTH = 2
DEFAULT_TIMEOUT = 30.0

# Rate limiting: conservative but parallel
RATE_LIMITER = AsyncLimiter(max_rate=5, time_period=1.0)
MAX_CONCURRENT = 10  # Max concurrent HTTP requests


def clean_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:150]


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")


def token_count(text: str) -> int:
    """Approximate token count (whitespace split)."""
    return len(text.split())


def chunk_article(
    text: str,
    min_tokens: int = 200,
    max_tokens: int = 400,
) -> List[str]:
    """Split an article into paragraph-level chunks."""
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
        chunks[-1] += " " + " ".join(buffer)

    return [c for c in chunks if token_count(c) >= 50]


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
    """Remove near-duplicate chunks using MinHash LSH."""
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


def validate(
    records: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    raw_dir: Path,
    min_tok: int,
    max_tok: int
) -> bool:
    print("\n" + "=" * 60)
    print("STAGE 0 VALIDATION")
    print("=" * 60)
    passed = True

    by_topic: Dict[str, List] = {}
    for r in records:
        by_topic.setdefault(r["topic_name"], []).append(r)

    # 1. Check topics have at least 1 page
    for name, pages in by_topic.items():
        ok = len(pages) >= 1
        status = "✓" if ok else "✗"
        print(f"  {status} Topic '{name}': {len(pages)} pages downloaded")
        if not ok:
            passed = False

    # 2. Raw files exist on disk
    missing_files = 0
    for r in records:
        p = raw_dir / r["topic_slug"] / "pages" / r["filename"]
        if not p.exists():
            missing_files += 1
    status = "✓" if missing_files == 0 else "✗"
    print(f"  {status} Missing raw files: {missing_files}")
    if missing_files > 0:
        passed = False

    # 3. All chunks have required fields
    required_keys = {"chunk_id", "text", "topic_name", "token_count"}
    bad_chunks = [c for c in chunks if not required_keys.issubset(c.keys())]
    status = "✓" if not bad_chunks else "✗"
    print(f"  {status} Chunks with missing fields: {len(bad_chunks)}")
    if bad_chunks:
        passed = False

    # 4. Token counts are within expected range
    out_of_range = [
        c for c in chunks
        if c["token_count"] < min_tok * 0.5 or c["token_count"] > max_tok * 1.5
    ]
    status = "✓" if len(out_of_range) == 0 else "⚠"
    print(f"  {status} Chunks outside token range ({min_tok}–{max_tok}): {len(out_of_range)}")

    return passed


# -----------------------------
# Wikipedia API client
# -----------------------------

class WikipediaClient:
    """Async Wikipedia API client with rate limiting."""

    def __init__(
        self,
        api_url: str = DEFAULT_API,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
        max_concurrent: int = MAX_CONCURRENT,
    ) -> None:
        self.api_url = api_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": self.user_agent},
            timeout=timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _api_request(self, params: Dict) -> Dict:
        """Make rate-limited async API request."""
        params = {**params, "format": "json"}

        async with self.semaphore:
            async with RATE_LIMITER:
                try:
                    async with self.session.get(self.api_url, params=params) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                except Exception as e:
                    print(f"API request failed: {e}")
                    return {}

    # -------- category / backlinks / categories-of-page --------

    async def get_category_members(
        self,
        category: str,
        depth: int = 1,
        max_pages: int = 1000,
    ) -> Set[str]:
        """Recursively get article titles from Category:<name> and its subcategories."""
        to_visit = deque([(f"Category:{category}", 0)])
        visited: Set[str] = set()
        pages: Set[str] = set()

        while to_visit and len(pages) < max_pages:
            cat, level = to_visit.popleft()
            if cat in visited:
                continue
            visited.add(cat)

            cont = None
            while len(pages) < max_pages:
                params = {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": cat,
                    "cmnamespace": "0|14",  # Articles and categories
                    "cmlimit": "500",
                }
                if cont:
                    params["cmcontinue"] = cont

                data = await self._api_request(params)
                for cm in data.get("query", {}).get("categorymembers", []):
                    if cm["ns"] == 0:  # Article
                        pages.add(cm["title"])
                    elif cm["ns"] == 14 and level < depth:  # Subcategory
                        to_visit.append((cm["title"], level + 1))

                if "continue" in data:
                    cont = data["continue"].get("cmcontinue")
                else:
                    break

                if len(pages) >= max_pages:
                    break

        return pages

    async def get_backlinks(self, title: str, limit: int = 1000) -> Set[str]:
        """Pages linking to a given title."""
        results: Set[str] = set()
        cont = None

        while len(results) < limit:
            params = {
                "action": "query",
                "list": "backlinks",
                "bltitle": title,
                "blnamespace": 0,
                "bllimit": "500",
            }
            if cont:
                params["blcontinue"] = cont

            data = await self._api_request(params)
            for b in data.get("query", {}).get("backlinks", []):
                results.add(b["title"])

            if "continue" in data and len(results) < limit:
                cont = data["continue"].get("blcontinue")
            else:
                break

        return results

    async def get_page_categories(self, title: str) -> Set[str]:
        """Categories of a given page (for robustness if Category:<seed> is missing)."""
        params = {
            "action": "query",
            "prop": "categories",
            "titles": title,
            "clshow": "!hidden",
            "cllimit": "500",
        }
        data = await self._api_request(params)
        pages = data.get("query", {}).get("pages", {})
        cats: Set[str] = set()
        for _, p in pages.items():
            for c in p.get("categories", []) or []:
                cats.add(c["title"].replace("Category:", "", 1))
        return cats

    async def get_page_text(self, title: str) -> str:
        """Return plaintext extract for a given page."""
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": title,
        }
        data = await self._api_request(params)
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            return p.get("extract", "") or ""
        return ""


# -----------------------------
# Corpus builder
# -----------------------------

class CorpusBuilder:
    """High-level helper to build a Wikipedia corpus for a single topic."""

    def __init__(
        self,
        client: WikipediaClient,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.client = client
        self.output_dir = output_dir

    async def discover_titles(
        self,
        seed_title: str,
        cat_depth: int,
        max_pages: int,
    ) -> Set[str]:
        """
        Discover titles via:
        - Category:<seed_title> + subcategories.
        - Categories of the seed page (then their members) as a fallback.
        - Backlinks to the seed page.
        """
        titles: Set[str] = set()

        # Primary: Category:<seed>
        cat_task_main = self.client.get_category_members(
            category=seed_title,
            depth=cat_depth,
            max_pages=max_pages,
        )

        # Backlinks to the seed
        backlinks_task = self.client.get_backlinks(seed_title, limit=max_pages)

        # Also: fetch categories that the seed page belongs to, then expand each
        async def expand_seed_categories() -> Set[str]:
            out: Set[str] = set()
            seed_cats = await self.client.get_page_categories(seed_title)
            for cat_name in seed_cats:
                if len(out) >= max_pages:
                    break
                sub = await self.client.get_category_members(
                    category=cat_name,
                    depth=cat_depth,
                    max_pages=max_pages,
                )
                out |= sub
                if len(out) >= max_pages:
                    break
            return out

        cat_task_seed = expand_seed_categories()

        # Run all discovery in parallel
        cat_main, backlink_titles, cat_from_seed = await asyncio.gather(
            cat_task_main,
            backlinks_task,
            cat_task_seed,
            return_exceptions=True,
        )

        # Merge results (only sets)
        if isinstance(cat_main, set):
            titles |= cat_main
        if isinstance(backlink_titles, set):
            titles |= backlink_titles
        if isinstance(cat_from_seed, set):
            titles |= cat_from_seed

        # Always include the seed page itself
        titles.add(seed_title)

        # Enforce global cap
        if len(titles) > max_pages:
            titles = set(list(titles)[:max_pages])

        return titles

    async def download_single_page(
        self,
        title: str,
        topic_slug: str,
        topic_name: str,
        topic_dir: Path,
    ) -> Optional[Dict]:
        """Download a single page and return manifest record."""
        try:
            text = await self.client.get_page_text(title)
            if not text.strip():
                return None

            filename = clean_filename(title) + ".txt"
            path = topic_dir / filename

            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

            record = {
                "topic_slug": topic_slug,
                "topic_name": topic_name,
                "title": title,
                "filename": filename,
                "rel_path": str(Path(topic_slug) / "pages" / filename),
                "source": "wikipedia",
            }
            return record

        except Exception as e:
            print(f"\nError downloading {title}: {e}")
            return None

    async def download_titles(
        self,
        topic_slug: str,
        topic_name: str,
        titles: Iterable[str],
    ) -> List[Dict]:
        """Download page text in parallel and write .txt files + manifest.jsonl."""
        topic_dir = Path(self.output_dir) / topic_slug / "pages"
        topic_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = Path(self.output_dir) / topic_slug / "manifest.jsonl"

        sorted_titles = sorted(set(titles))

        print(
            f"\nDownloading {len(sorted_titles)} pages with {MAX_CONCURRENT} concurrent requests..."
        )
        print(f"Rate limit: {RATE_LIMITER.max_rate} req/sec\n")

        tasks = [
            self.download_single_page(title, topic_slug, topic_name, topic_dir)
            for title in sorted_titles
        ]

        manifest_records: List[Dict] = []
        with open(manifest_path, "w", encoding="utf-8") as manifest_f:
            for coro in async_tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"Downloading {topic_name}",
            ):
                record = await coro
                if record:
                    manifest_records.append(record)
                    manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest_f.flush()

        print(f"\n✓ Successfully downloaded {len(manifest_records)}/{len(sorted_titles)} pages")
        return manifest_records


# -----------------------------
# CLI & main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia corpus for a user-specified topic (async, categories + backlinks)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory (default: data/raw).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Global cap on discovered pages (default: 1500).",
    )
    parser.add_argument(
        "--cat-depth",
        type=int,
        default=DEFAULT_CAT_DEPTH,
        help="Category traversal depth (default: 2).",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Topic or person, e.g. 'Imran Khan'. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help="Max concurrent HTTP requests (default: 10).",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    topic_name = args.topic or input("Enter topic/person (e.g. Imran Khan): ").strip()
    while not topic_name:
        topic_name = input("Topic cannot be empty. Enter topic/person: ").strip()

    topic_slug = slugify(topic_name)

    async with WikipediaClient(
        api_url=DEFAULT_API,
        user_agent=DEFAULT_USER_AGENT,
        timeout=DEFAULT_TIMEOUT,
        max_concurrent=args.max_concurrent,
    ) as client:
        builder = CorpusBuilder(client=client, output_dir=args.output_dir)

        print(f"\n{'='*60}")
        print(f"Discovering titles for: {topic_name}")
        print(f"Slug: {topic_slug}")
        print(f"Max pages: {args.max_pages}, cat depth: {args.cat_depth}")
        print(f"{'='*60}\n")

        titles = await builder.discover_titles(
            seed_title=topic_name,
            cat_depth=args.cat_depth,
            max_pages=args.max_pages,
        )
        print(f"✓ Discovered {len(titles)} candidate pages (capped at {args.max_pages})")

        print(f"\n{'='*60}")
        print(f"Downloading pages for: {topic_name}")
        print(f"{'='*60}")

        manifest_records = await builder.download_titles(
            topic_slug=topic_slug,
            topic_name=topic_name,
            titles=titles,
        )

        print(f"\n{'='*60}")
        print("Chunking and deduplicating...")
        min_tok = 200
        max_tok = 400
        topic_dir = Path(args.output_dir) / topic_slug / "pages"
        all_chunks = []
        chunk_id = 0

        for rec in manifest_records:
            page_path = topic_dir / rec["filename"]
            if not page_path.exists():
                continue
            text = page_path.read_text(encoding="utf-8")
            chunks_txt = chunk_article(text, min_tok, max_tok)
            for idx, c_text in enumerate(chunks_txt):
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "article_id": hashlib.md5(rec["title"].encode()).hexdigest()[:8],
                    "chunk_idx": idx,
                    "topic_name": rec["topic_name"],
                    "topic_slug": rec["topic_slug"],
                    "title": rec["title"],
                    "text": c_text,
                    "token_count": token_count(c_text)
                })
                chunk_id += 1

        print(f"Total chunks before dedup: {len(all_chunks)}")
        kept_chunks, removed = deduplicate_chunks(all_chunks, threshold=0.85, num_perm=128)
        print(f"Dedup complete: kept {len(kept_chunks)}, removed {removed}")

        chunks_dir = Path("data/chunks")
        chunks_dir.mkdir(parents=True, exist_ok=True)
        out_path = chunks_dir / f"{topic_slug}_chunks.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for c in kept_chunks:
                f.write(json.dumps(c) + "\n")
        print(f"Saved {len(kept_chunks)} chunks -> {out_path}")

        validate(manifest_records, kept_chunks, Path(args.output_dir), min_tok, max_tok)

    print(f"\n{'='*60}")
    print("✓ Corpus build complete!")
    print(f"Output directory: {args.output_dir}/{topic_slug}/")
    print(f"{'='*60}\n")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


def load_topics(topics_json_path: Path) -> Dict[int, Dict]:
    """
    Load topics from a topics.json file and return a dict keyed by topic_id.
    
    Args:
        topics_json_path: Path to topics.json file
        
    Returns:
        Dictionary mapping topic_id -> topic dict
    """
    import json
    
    topics_by_id = {}
    
    if not topics_json_path.exists():
        print(f"Warning: {topics_json_path} does not exist")
        return topics_by_id
    
    with open(topics_json_path, 'r', encoding='utf-8') as f:
        topics_list = json.load(f)
    
    for topic in topics_list:
        tid = topic.get('topic_id')
        if tid is not None:
            topics_by_id[int(tid)] = topic
    
    return topics_by_id


def precompute_topic_cache(
    topic_ids: Set[int],
    topics_by_id: Dict[int, Dict],
    docs_dir: Path
) -> Dict[int, Dict[str, Any]]:
    """
    Precompute a cache of topic metadata for faster lookup.
    
    Args:
        topic_ids: Set of topic IDs to cache
        topics_by_id: Dictionary of topic metadata
        docs_dir: Directory containing topic documents
        
    Returns:
        Dictionary mapping topic_id -> cached metadata
    """
    cache = {}
    
    for tid in topic_ids:
        if tid not in topics_by_id:
            continue
            
        topic = topics_by_id[tid]
        cache[tid] = {
            'topic_name': topic.get('topic_name', ''),
            'topic_slug': topic.get('topic_slug', ''),
            'top_words': topic.get('top_words', []),
            'relevant_documents': topic.get('relevant_documents', []),
            'count': topic.get('count', 0),
        }
    
    return cache


if __name__ == "__main__":
    main()