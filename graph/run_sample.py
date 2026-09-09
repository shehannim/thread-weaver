"""
Sample runner: extract entities/relations from ~15-20 chunks stratified
across doc_types, aggregate, write vault notes, and print stats.

Usage:
    python graph/run_sample.py
"""

import json
import os
import sys
import logging
import random
from collections import defaultdict

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from graph.extract import extract_from_chunk
from graph.aggregate import build_entity_index
from graph.vault_writer import write_vault
from graph.stats import compute_stats, save_stats, print_stats_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHUNKS_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "chunks.jsonl")
VAULT_DIR = os.path.join(PROJECT_ROOT, "data", "vault")
STATS_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "extraction_stats.json")
SAMPLE_EXTRACTIONS_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "sample_extractions.jsonl")

# How many chunks per doc_type
SAMPLES_PER_TYPE = 5
TARGET_TYPES = ["novel", "wiki", "codex", "ephemera"]


def load_chunks(filepath: str) -> list[dict]:
    """Load all chunks from the JSONL file."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def stratified_sample(chunks: list[dict], per_type: int = 5) -> list[dict]:
    """
    Sample chunks stratified by doc_type.
    Takes `per_type` chunks from each TARGET_TYPE present in the data.
    Falls back to random sampling if a type has fewer than `per_type` chunks.
    """
    by_type = defaultdict(list)
    for chunk in chunks:
        dtype = chunk.get("doc_type", "unknown")
        by_type[dtype].append(chunk)

    sampled = []
    for dtype in TARGET_TYPES:
        available = by_type.get(dtype, [])
        if not available:
            logger.warning("No chunks found for doc_type '%s'", dtype)
            continue
        n = min(per_type, len(available))
        sampled.extend(random.sample(available, n))
        logger.info("Sampled %d chunks from doc_type '%s'", n, dtype)

    # If we got fewer than 15, pad from "unknown" or any other types
    if len(sampled) < 15:
        remaining = [c for c in chunks if c not in sampled]
        extra = min(15 - len(sampled), len(remaining))
        if extra > 0:
            sampled.extend(random.sample(remaining, extra))
            logger.info("Padded sample with %d additional chunks", extra)

    return sampled


def main():
    if not os.path.exists(CHUNKS_FILE):
        print(f"ERROR: Chunks file not found at {CHUNKS_FILE}")
        print("Run `python ingest/ingest.py` first to generate chunks.jsonl")
        sys.exit(1)

    # Load and sample
    all_chunks = load_chunks(CHUNKS_FILE)
    logger.info("Loaded %d total chunks from %s", len(all_chunks), CHUNKS_FILE)

    sample = stratified_sample(all_chunks, per_type=SAMPLES_PER_TYPE)
    logger.info("Selected %d sample chunks for extraction", len(sample))

    # Extract from each chunk
    all_extractions = []
    failed_chunks = 0

    for i, chunk in enumerate(sample):
        logger.info(
            "[%d/%d] Extracting from chunk %s (doc_type: %s)...",
            i + 1, len(sample),
            chunk.get("chunk_id", "?"),
            chunk.get("doc_type", "?"),
        )

        result = extract_from_chunk(chunk)

        if result is None:
            failed_chunks += 1
            continue

        # Attach chunk metadata for aggregation
        result["chunk_meta"] = {
            "chunk_id": chunk.get("chunk_id", ""),
            "doc_id": chunk.get("doc_id", ""),
            "source_reliability": chunk.get("source_reliability", ""),
            "page_number": chunk.get("page_number", ""),
        }
        all_extractions.append(result)

        # Also save raw extractions for inspection
        logger.info(
            "  -> %d entities, %d relations",
            len(result.get("entities", [])),
            len(result.get("relations", [])),
        )

    # Save raw extractions
    os.makedirs(os.path.dirname(SAMPLE_EXTRACTIONS_FILE), exist_ok=True)
    with open(SAMPLE_EXTRACTIONS_FILE, "w", encoding="utf-8") as f:
        for ext in all_extractions:
            f.write(json.dumps(ext) + "\n")
    logger.info("Raw extractions saved to %s", SAMPLE_EXTRACTIONS_FILE)

    # Aggregate
    logger.info("Aggregating entities and relations...")
    entity_index, ambiguous_merges = build_entity_index(all_extractions)

    # Write vault
    logger.info("Writing vault notes to %s...", VAULT_DIR)
    notes_written = write_vault(entity_index, VAULT_DIR)

    # Compute and save stats
    stats = compute_stats(
        entity_index=entity_index,
        ambiguous_merges=ambiguous_merges,
        total_chunks_processed=len(sample),
        failed_chunks=failed_chunks,
    )
    save_stats(stats, STATS_FILE)
    print_stats_summary(stats)

    print(f"\n✓ {notes_written} vault notes written to: {os.path.abspath(VAULT_DIR)}")
    print(f"✓ Stats saved to: {os.path.abspath(STATS_FILE)}")
    print(f"✓ Raw extractions saved to: {os.path.abspath(SAMPLE_EXTRACTIONS_FILE)}")
    print("\nReview the vault notes and stats before running full-corpus extraction.")


if __name__ == "__main__":
    main()
