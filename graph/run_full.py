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
REQUESTS_PER_MINUTE = 18 
SECONDS_PER_REQUEST = 60.0 / REQUESTS_PER_MINUTE
MAX_WORKERS = 15  # As requested, ~10-15 concurrent chunks

# Lock for writing to the extractions file safely from multiple threads
file_lock = threading.Lock()

class RateLimiter:
    def __init__(self, interval_seconds: float):
        self.interval = interval_seconds
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

global_rate_limiter = RateLimiter(SECONDS_PER_REQUEST)


def process_chunk_worker(chunk: dict) -> dict | None:
    global_rate_limiter.wait()
    
    result = extract_from_chunk(chunk)
    if result is None:
        return None
        
    result["chunk_meta"] = {
        "chunk_id": chunk.get("chunk_id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "source_reliability": chunk.get("source_reliability", ""),
        "page_number": chunk.get("page_number", ""),
    }
    
    # Checkpoint to disk immediately
    with file_lock:
        with open(EXTRACTIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
            
    return result


def main():
    if not os.path.exists(CHUNKS_FILE):
        print(f"ERROR: Chunks file not found at {CHUNKS_FILE}")
        print("Run `python ingest/ingest.py` first to generate chunks.jsonl")
        sys.exit(1)

    # 1. Load already processed chunks to allow resuming
    processed_chunk_ids = set()
    all_extractions = []
    
    os.makedirs(os.path.dirname(EXTRACTIONS_FILE), exist_ok=True)
    if os.path.exists(EXTRACTIONS_FILE):
        with open(EXTRACTIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        ext = json.loads(line)
                        cid = ext.get("chunk_meta", {}).get("chunk_id")
                        if cid:
                            processed_chunk_ids.add(cid)
                            all_extractions.append(ext)
                    except json.JSONDecodeError:
                        pass
                        
    if processed_chunk_ids:
        logger.info("Found %d already processed chunks. Resuming...", len(processed_chunk_ids))

    # 2. Load pending chunks
    pending_chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line.strip())
                if c.get("chunk_id") not in processed_chunk_ids:
                    pending_chunks.append(c)
                
    total_chunks = len(processed_chunk_ids) + len(pending_chunks)
    logger.info("Loaded %d total chunks. %d remaining to process.", total_chunks, len(pending_chunks))
    
    if not pending_chunks:
        logger.info("All chunks already processed! Skipping to aggregation.")
    else:
        logger.info("Concurrency config: %d MAX_WORKERS, bounded to %d requests/minute.", MAX_WORKERS, REQUESTS_PER_MINUTE)
        
        # Estimate time based on sample average latency
        AVG_CHUNK_LATENCY_SEC = 75.0  # Assumed from 60-90s user report
        concurrency_throughput = MAX_WORKERS / AVG_CHUNK_LATENCY_SEC
        rate_limit_throughput = REQUESTS_PER_MINUTE / 60.0
        
        effective_throughput = min(concurrency_throughput, rate_limit_throughput)
        estimated_seconds = len(pending_chunks) / effective_throughput if effective_throughput > 0 else 0
        
        logger.info(
            "ESTIMATED TIME FOR %d CHUNKS: ~%.2f hours (%.1f minutes). "
            "[Bottleneck: %s]",
            len(pending_chunks),
            estimated_seconds / 3600, 
            estimated_seconds / 60,
            "Rate Limit" if rate_limit_throughput < concurrency_throughput else "Concurrency Limit"
        )
        
        failed_chunks = 0
        completed = 0
        start_time = time.time()

        # 3. Process concurrently
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {executor.submit(process_chunk_worker, c): c for c in pending_chunks}
            
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
                if completed % 10 == 0 or completed == len(pending_chunks):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Progress: %d/%d (%.1f%%) | Failed: %d | Rate: %.2f chunks/sec",
                        completed, len(pending_chunks), (completed / len(pending_chunks)) * 100, failed_chunks, rate
                    )

    # 4. Aggregate
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
        failed_chunks=len(pending_chunks) - (len(all_extractions) - len(processed_chunk_ids)),
    )
    save_stats(stats, STATS_FILE)
    print_stats_summary(stats)

    print(f"\n[OK] {notes_written} vault notes written to: {os.path.abspath(VAULT_DIR)}")
    print(f"[OK] Full extraction stats saved to: {os.path.abspath(STATS_FILE)}")


if __name__ == "__main__":
    main()
