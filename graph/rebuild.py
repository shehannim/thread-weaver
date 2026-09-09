import os
import json
import argparse
import logging

from graph.aggregate import build_entity_index
from graph.vault_writer import write_vault
from graph.graph_store import export_graph
from graph.stats import compute_stats, save_stats

logger = logging.getLogger(__name__)

def rebuild_graph(input_path, graph_output_path, vault_output_dir, stats_output_path):
    """
    Rebuilds the graph JSON, Obsidian vault, and stats offline
    from an existing extraction JSONL file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    valid_records = 0
    malformed_records = 0
    all_extractions = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            clean_line = line.strip()
            if not clean_line:
                continue
            try:
                data = json.loads(clean_line)
                if not isinstance(data, dict):
                    raise ValueError("Record is not a dictionary")
                if "entities" not in data or not isinstance(data["entities"], list):
                    raise ValueError("Missing or invalid 'entities' list")
                if "relations" not in data or not isinstance(data["relations"], list):
                    raise ValueError("Missing or invalid 'relations' list")
                
                # accept chunk_meta when present implicitly
                all_extractions.append(data)
                valid_records += 1
            except Exception as e:
                logger.warning("Line %d is malformed: %s", line_no, e)
                malformed_records += 1
                
    if valid_records == 0:
        raise ValueError(f"No valid extraction records loaded from {input_path}")
        
    logger.info("Loaded %d valid records (%d malformed) from %s", valid_records, malformed_records, input_path)
    
    # Aggregate offline
    entity_index, ambiguous_merges = build_entity_index(all_extractions)
    
    # Export Graph
    graph_summary = export_graph(entity_index, graph_output_path)
    
    # Write Vault (Overwrites only notes generated for entities in the current rebuild. Stale notes may remain)
    logger.info("Writing vault notes to %s...", vault_output_dir)
    notes_written = write_vault(entity_index, vault_output_dir)
    
    contradiction_count = sum(len(ent.contradictions) for ent in entity_index.values())
    ambiguous_merge_count = len(ambiguous_merges)
    
    # Compute and Save Stats
    stats = compute_stats(
        entity_index=entity_index,
        ambiguous_merges=ambiguous_merges,
        total_chunks_processed=valid_records + malformed_records,
        failed_chunks=malformed_records
    )
    save_stats(stats, stats_output_path)
    
    summary = {
        "input_path": input_path,
        "valid_records": valid_records,
        "malformed_records": malformed_records,
        "entity_count": graph_summary["entity_count"],
        "relation_count": graph_summary["relation_count"],
        "notes_written": notes_written,
        "contradiction_count": contradiction_count,
        "ambiguous_merge_count": ambiguous_merge_count,
        "graph_output_path": graph_output_path,
        "vault_output_dir": vault_output_dir,
        "stats_output_path": stats_output_path
    }
    
    print("\n=== REBUILD SUMMARY ===")
    print(f"Input Path           : {summary['input_path']}")
    print(f"Valid Records        : {summary['valid_records']}")
    print(f"Malformed Records    : {summary['malformed_records']}")
    print(f"Entity Count         : {summary['entity_count']}")
    print(f"Relation Count       : {summary['relation_count']}")
    print(f"Vault Notes Written  : {summary['notes_written']}")
    print(f"Contradiction Count  : {summary['contradiction_count']}")
    print(f"Ambiguous Merges     : {summary['ambiguous_merge_count']}")
    print(f"Graph Output Path    : {summary['graph_output_path']}")
    print(f"Vault Output Dir     : {summary['vault_output_dir']}")
    print(f"Stats Output Path    : {summary['stats_output_path']}")
    print("=======================\n")
    print("NOTE: Stale notes from previous extractions may remain in the vault directory.")
    
    return summary

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Offline graph artifact rebuild command.")
    
    # All relative paths resolve from the repository root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    default_input = os.path.join(project_root, "data", "processed", "full_extractions.jsonl")
    default_graph = os.path.join(project_root, "data", "processed", "graph.json")
    default_vault = os.path.join(project_root, "data", "vault")
    default_stats = os.path.join(project_root, "data", "processed", "extraction_stats_rebuilt.json")
    
    parser.add_argument("--input", default=default_input, help="Input JSONL file")
    parser.add_argument("--graph-output", default=default_graph, help="Output JSON graph path")
    parser.add_argument("--vault-output", default=default_vault, help="Output Vault directory path")
    parser.add_argument("--stats-output", default=default_stats, help="Output extraction stats path")
    
    args = parser.parse_args()
    
    try:
        rebuild_graph(
            input_path=args.input,
            graph_output_path=args.graph_output,
            vault_output_dir=args.vault_output,
            stats_output_path=args.stats_output
        )
    except Exception as e:
        logger.error("Rebuild failed: %s", e)
        exit(1)

if __name__ == "__main__":
    main()
