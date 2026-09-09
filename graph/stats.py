"""
Extraction statistics logger.

Tracks entity count (by type), relation count (by type), malformed output rate,
contradiction count, and ambiguous merge count.
Writes summary to /data/processed/extraction_stats.json.
"""

import json
import os
import logging
from collections import Counter
from graph.aggregate import MergedEntity, AmbiguousMerge

logger = logging.getLogger(__name__)


def compute_stats(
    entity_index: dict[str, MergedEntity],
    ambiguous_merges: list[AmbiguousMerge],
    total_chunks_processed: int,
    failed_chunks: int,
) -> dict:
    """
    Compute extraction statistics from the aggregated graph.

    Returns a summary dict.
    """
    # Entity counts by type
    entity_type_counts = Counter()
    for entity in entity_index.values():
        entity_type_counts[entity.entity_type] += 1

    # Relation counts by type
    relation_type_counts = Counter()
    total_relations = 0
    negated_relations = 0
    low_confidence_relations = 0
    for entity in entity_index.values():
        for rel in entity.relations:
            relation_type_counts[rel["relation_type"]] += 1
            total_relations += 1
            if rel.get("negated", False):
                negated_relations += 1
            if rel.get("confidence", 1.0) < 0.4:
                low_confidence_relations += 1

    # Contradiction count
    total_contradictions = sum(
        len(entity.contradictions) for entity in entity_index.values()
    )

    stats = {
        "total_chunks_processed": total_chunks_processed,
        "failed_chunks": failed_chunks,
        "malformed_output_rate": (
            f"{failed_chunks}/{total_chunks_processed} "
            f"({failed_chunks / max(total_chunks_processed, 1) * 100:.1f}%)"
        ),
        "total_entities": len(entity_index),
        "entity_counts_by_type": dict(entity_type_counts.most_common()),
        "total_relations": total_relations,
        "relation_counts_by_type": dict(relation_type_counts.most_common()),
        "negated_relations": negated_relations,
        "low_confidence_relations": low_confidence_relations,
        "total_contradictions": total_contradictions,
        "ambiguous_merges_flagged": len(ambiguous_merges),
        "ambiguous_merge_details": [
            {
                "entity_a": am.entity_a,
                "entity_b": am.entity_b,
                "similarity": round(am.similarity, 3),
                "type_a": am.entity_type_a,
                "type_b": am.entity_type_b,
            }
            for am in ambiguous_merges
        ],
    }

    return stats


def save_stats(stats: dict, output_path: str):
    """Write stats to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Extraction stats saved to %s", output_path)


def print_stats_summary(stats: dict):
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)
    print(f"Chunks processed:       {stats['total_chunks_processed']}")
    print(f"Failed chunks:          {stats['malformed_output_rate']}")
    print(f"Total entities:         {stats['total_entities']}")
    print(f"Total relations:        {stats['total_relations']}")
    print(f"  Negated relations:    {stats['negated_relations']}")
    print(f"  Low confidence (<0.4):{stats['low_confidence_relations']}")
    print(f"Contradictions found:   {stats['total_contradictions']}")
    print(f"Ambiguous merges:       {stats['ambiguous_merges_flagged']}")

    print("\nEntities by type:")
    for etype, count in stats.get("entity_counts_by_type", {}).items():
        print(f"  {etype:20s} {count}")

    print("\nRelations by type:")
    for rtype, count in stats.get("relation_counts_by_type", {}).items():
        print(f"  {rtype:20s} {count}")

    if stats.get("ambiguous_merge_details"):
        print("\nAmbiguous merges (flagged for manual review):")
        for am in stats["ambiguous_merge_details"]:
            print(
                f"  '{am['entity_a']}' ({am['type_a']}) <-> "
                f"'{am['entity_b']}' ({am['type_b']}) "
                f"— similarity: {am['similarity']}"
            )

    print("=" * 60 + "\n")
