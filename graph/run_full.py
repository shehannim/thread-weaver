"""
Full-corpus runner: extract entities/relations from all chunks concurrently.
Uses a ThreadPoolExecutor and a strict rate limiter to respect OpenRouter's free-tier limits.
"""

import json
import os
import sys
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
STATS_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "extraction_stats_full.json")
EXTRACTIONS_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "full_extractions.jsonl")

# Concurrency & Rate Limiting
# OpenRouter free tier limit is roughly 20 requests per minute
# We use 18 RPM to leave a small buffer for retries
REQUESTS_PER_MINUTE = 18 
SECONDS_PER_REQUEST = 60.0 / REQUESTS_PER_MINUTE
MAX_WORKERS = 30  # Enough to keep threads busy while waiting for 60-90s API responses


class RateLimiter:
    def __init__(self, interval_seconds: float):
        self.interval = interval_seconds
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        """Block until the interval has passed since the last call."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

global_rate_limiter = RateLimiter(SECONDS_PER_REQUEST)


def process_chunk_worker(chunk: dict) -> dict | None:
    """Worker function to process a single chunk, respecting the rate limit."""
    # Wait for rate limit before hitting the API
    global_rate_limiter.wait()
    
    result = extract_from_chunk(chunk)
    if result is None:
        return None
        
    # Attach chunk metadata for aggregation
    result["chunk_meta"] = {
        "chunk_id": chunk.get("chunk_id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "source_reliability": chunk.get("source_reliability", ""),
        "page_number": chunk.get("page_number", ""),
    }
    return result


def main():
    if not os.path.exists(CHUNKS_FILE):
        print(f"ERROR: Chunks file not found at {CHUNKS_FILE}")
        print("Run `python ingest/ingest.py` first to generate chunks.jsonl")
        sys.exit(1)

    # Load all chunks
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line.strip()))
                
    total_chunks = len(chunks)
    logger.info("Loaded %d total chunks for full-corpus extraction.", total_chunks)
    logger.info("Concurrency config: %d MAX_WORKERS, bounded to %d requests/minute.", MAX_WORKERS, REQUESTS_PER_MINUTE)
    
    # Estimate time
    estimated_seconds = total_chunks * SECONDS_PER_REQUEST
    estimated_hours = estimated_seconds / 3600
    logger.info("ESTIMATED TIME: ~%.2f hours (%.1f minutes).", estimated_hours, estimated_seconds / 60)
    
    all_extractions = []
    failed_chunks = 0
    completed = 0
    start_time = time.time()

    # Process concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        future_to_chunk = {executor.submit(process_chunk_worker, c): c for c in chunks}
        
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                result = future.result()
                if result is None:
                    failed_chunks += 1
                else:
                    all_extractions.append(result)
            except Exception as e:
                logger.error("Chunk %s raised an unhandled exception: %s", chunk.get("chunk_id", "?"), e)
                failed_chunks += 1
                
            completed += 1
            if completed % 50 == 0 or completed == total_chunks:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f%%) | Failed: %d | Rate: %.2f chunks/sec",
                    completed, total_chunks, (completed / total_chunks) * 100, failed_chunks, rate
                )

    # Save raw extractions continuously or at the end
    os.makedirs(os.path.dirname(EXTRACTIONS_FILE), exist_ok=True)
    with open(EXTRACTIONS_FILE, "w", encoding="utf-8") as f:
        for ext in all_extractions:
            f.write(json.dumps(ext) + "\n")
    logger.info("Raw extractions saved to %s", EXTRACTIONS_FILE)

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
        total_chunks_processed=total_chunks,
        failed_chunks=failed_chunks,
    )
    save_stats(stats, STATS_FILE)
    print_stats_summary(stats)

    print(f"\n[OK] {notes_written} vault notes written to: {os.path.abspath(VAULT_DIR)}")
    print(f"[OK] Full extraction stats saved to: {os.path.abspath(STATS_FILE)}")
    
    total_time = (time.time() - start_time) / 3600
    print(f"Total execution time: {total_time:.2f} hours")


if __name__ == "__main__":
    main()
