"""
Aggregate extracted entities and relations across all chunks.

Deduplication strategy (user-approved — conservative):
  - Auto-merge: exact match (case-insensitive) OR fuzzy ratio >= 0.92
  - Flag for manual review: fuzzy ratio in [0.85, 0.92)
  - No merge: fuzzy ratio < 0.85

Contradiction detection:
  If the same (entity_A, entity_B) pair has conflicting relation types
  (e.g. allied_with vs enemy_of), BOTH are kept and tagged.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from thefuzz import fuzz

logger = logging.getLogger(__name__)

# Relation types that are semantically contradictory when applied to the same pair
CONTRADICTORY_PAIRS = {
    frozenset({"allied_with", "enemy_of"}),
    frozenset({"reports_to", "leads"}),
    frozenset({"member_of", "enemy_of"}),       # Can't be member and enemy simultaneously (usually)
}

# Also catch direct negation contradictions (same relation, negated vs not)
# This is handled separately in find_contradictions()

INVERSE_RELATIONS = {
    "leads": "led_by",
    "led_by": "leads",
    "depends_on": "required_by",
    "required_by": "depends_on",
    "causes": "caused_by",
    "caused_by": "causes",
    "succeeded_by": "preceded_by",
    "preceded_by": "succeeded_by",
    "powers": "powered_by",
    "powered_by": "powers",
    "controls": "controlled_by",
    "controlled_by": "controls",
    "possesses": "possessed_by",
    "possessed_by": "possesses",
    "located_in": "contains",
    "contains": "located_in",
    "composed_of": "part_of",
    "part_of": "composed_of"
}


@dataclass
class MergedEntity:
    """An entity after cross-chunk deduplication."""
    canonical_name: str
    entity_type: str
    aliases: set = field(default_factory=set)
    relations: list = field(default_factory=list)      # List of relation dicts (excludes has_property)
    properties: list = field(default_factory=list)     # List of has_property dicts (rendered as plain text, not links)
    mentions: list = field(default_factory=list)        # List of {chunk_id, doc_id, reliability}
    contradictions: list = field(default_factory=list)  # List of contradiction descriptions


@dataclass
class AmbiguousMerge:
    """A potential merge flagged for manual review."""
    entity_a: str
    entity_b: str
    similarity: float
    entity_type_a: str
    entity_type_b: str


def normalize_name(name: str) -> str:
    """Lowercase, strip whitespace for comparison."""
    return name.strip().lower()


def build_entity_index(all_extractions: list[dict]) -> tuple[
    dict[str, MergedEntity], list[AmbiguousMerge]
]:
    """
    Aggregate entities from all chunk extractions, deduplicate,
    and build relations + mentions for each.

    Args:
        all_extractions: list of dicts, each with structure:
            {"entities": [...], "relations": [...], "chunk_meta": {...}}
            where chunk_meta has {chunk_id, doc_id, source_reliability}

    Returns:
        (entity_index, ambiguous_merges)
        entity_index: dict mapping canonical_name -> MergedEntity
        ambiguous_merges: list of AmbiguousMerge flagged for review
    """
    # Phase 1: Collect all raw entities with their metadata
    raw_entities = []  # list of (name, entity_type, aliases, chunk_meta)
    for extraction in all_extractions:
        chunk_meta = extraction.get("chunk_meta", {})
        for ent in extraction.get("entities", []):
            raw_entities.append((
                ent["name"],
                ent.get("entity_type", "unknown"),
                set(ent.get("aliases", [])),
                chunk_meta,
            ))

    # Phase 2: Deduplicate entities
    # We build a mapping from normalized name -> canonical entry
    canonical_map: dict[str, str] = {}   # normalized_name -> canonical_name
    entity_index: dict[str, MergedEntity] = {}
    ambiguous_merges: list[AmbiguousMerge] = []

    for name, etype, aliases, chunk_meta in raw_entities:
        norm = normalize_name(name)

        # Check exact match first
        if norm in canonical_map:
            canonical = canonical_map[norm]
            canon_type = entity_index[canonical].entity_type

            if canon_type != "unknown" and etype != "unknown" and canon_type != etype:
                ambiguous_merges.append(AmbiguousMerge(
                    entity_a=canonical,
                    entity_b=name,
                    similarity=1.0,
                    entity_type_a=canon_type,
                    entity_type_b=etype,
                ))
            elif canon_type == "unknown" and etype != "unknown":
                entity_index[canonical].entity_type = etype

            entity_index[canonical].aliases.update(aliases)
            entity_index[canonical].aliases.add(name)  # Keep original casing as alias
            entity_index[canonical].mentions.append(chunk_meta)
            continue

        # Check fuzzy match against existing canonical names
        best_match = None
        best_ratio = 0.0
        for existing_canon in entity_index:
            ratio = fuzz.ratio(norm, normalize_name(existing_canon)) / 100.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = existing_canon

        if best_ratio >= 0.92:
            canon_type = entity_index[best_match].entity_type
            if canon_type == etype or canon_type == "unknown" or etype == "unknown":
                # Auto-merge
                canonical_map[norm] = best_match
                if canon_type == "unknown" and etype != "unknown":
                    entity_index[best_match].entity_type = etype
                entity_index[best_match].aliases.update(aliases)
                entity_index[best_match].aliases.add(name)
                entity_index[best_match].mentions.append(chunk_meta)
                logger.info(
                    "Auto-merged '%s' into '%s' (similarity: %.2f)",
                    name, best_match, best_ratio,
                )
            else:
                # Do not merge conflicting known types
                ambiguous_merges.append(AmbiguousMerge(
                    entity_a=best_match,
                    entity_b=name,
                    similarity=best_ratio,
                    entity_type_a=canon_type,
                    entity_type_b=etype,
                ))
                canonical_map[norm] = name
                entity_index[name] = MergedEntity(
                    canonical_name=name,
                    entity_type=etype,
                    aliases=aliases,
                    mentions=[chunk_meta],
                )
        elif best_ratio >= 0.85:
            # Flag for manual review, but do NOT merge
            ambiguous_merges.append(AmbiguousMerge(
                entity_a=best_match,
                entity_b=name,
                similarity=best_ratio,
                entity_type_a=entity_index[best_match].entity_type,
                entity_type_b=etype,
            ))
            # Create as separate entity
            canonical_map[norm] = name
            entity_index[name] = MergedEntity(
                canonical_name=name,
                entity_type=etype,
                aliases=aliases,
                mentions=[chunk_meta],
            )
        else:
            # New entity
            canonical_map[norm] = name
            entity_index[name] = MergedEntity(
                canonical_name=name,
                entity_type=etype,
                aliases=aliases,
                mentions=[chunk_meta],
            )

        # Also map all aliases
        for alias in aliases:
            alias_norm = normalize_name(alias)
            if alias_norm not in canonical_map:
                canonical_map[alias_norm] = canonical_map[norm]

    # Phase 3: Assign relations to entities
    for extraction in all_extractions:
        chunk_meta = extraction.get("chunk_meta", {})
        for rel in extraction.get("relations", []):
            # Resolve source entity name to canonical
            src_norm = normalize_name(rel["source_entity"])
            src_canon = canonical_map.get(src_norm, rel["source_entity"])

            # --- has_property: store as a plain attribute, not a graph edge ---
            if rel["relation_type"] == "has_property":
                prop_record = {
                    "property_value": rel.get("property_value", rel.get("target_entity", "")),
                    "negated": rel.get("negated", False),
                    "confidence": rel.get("confidence", 0.5),
                    "justification": rel.get("justification", ""),
                    "source_chunk_id": rel.get("source_chunk_id", chunk_meta.get("chunk_id", "")),
                    "source_doc_id": rel.get("source_doc_id", chunk_meta.get("doc_id", "")),
                    "source_reliability": rel.get("source_reliability", chunk_meta.get("source_reliability", "")),
                }
                if src_canon in entity_index:
                    entity_index[src_canon].properties.append(prop_record)
                else:
                    entity_index[src_canon] = MergedEntity(
                        canonical_name=src_canon,
                        entity_type="unknown",
                        mentions=[chunk_meta],
                        properties=[prop_record],
                    )
                    canonical_map[src_norm] = src_canon
                # Do NOT create a target node for property values
                continue

            # --- All other relation types: normal graph edge ---
            tgt_norm = normalize_name(rel["target_entity"])
            tgt_canon = canonical_map.get(tgt_norm, rel["target_entity"])

            enriched_rel = {
                "source_entity": src_canon,
                "relation_type": rel["relation_type"],
                "target_entity": tgt_canon,
                "negated": rel.get("negated", False),
                "confidence": rel.get("confidence", 0.5),
                "justification": rel.get("justification", ""),
                "source_chunk_id": rel.get("source_chunk_id", chunk_meta.get("chunk_id", "")),
                "source_doc_id": rel.get("source_doc_id", chunk_meta.get("doc_id", "")),
                "source_reliability": rel.get("source_reliability", chunk_meta.get("source_reliability", "")),
            }

            # Add to source entity
            if src_canon in entity_index:
                entity_index[src_canon].relations.append(enriched_rel)
            else:
                # Entity was mentioned in a relation but not in entities list — create stub
                entity_index[src_canon] = MergedEntity(
                    canonical_name=src_canon,
                    entity_type="unknown",
                    mentions=[chunk_meta],
                    relations=[enriched_rel],
                )
                canonical_map[src_norm] = src_canon

            # Also record on target entity (as incoming)
            if tgt_canon not in entity_index:
                entity_index[tgt_canon] = MergedEntity(
                    canonical_name=tgt_canon,
                    entity_type="unknown",
                    mentions=[chunk_meta],
                )
                canonical_map[tgt_norm] = tgt_canon

    # Phase 4: Detect contradictions
    find_contradictions(entity_index)

    return entity_index, ambiguous_merges


def find_contradictions(entity_index: dict[str, MergedEntity]) -> int:
    """
    Scan all relations for contradictions and tag them on the entities.
    Returns the total number of contradictions found.
    """
    contradiction_count = 0

    for entity_name, entity in entity_index.items():
        # Group relations by (source_entity, target_entity) pair
        pair_relations: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for rel in entity.relations:
            pair_key = (
                normalize_name(rel["source_entity"]),
                normalize_name(rel["target_entity"]),
            )
            pair_relations[pair_key].append(rel)

        for pair_key, rels in pair_relations.items():
            # Check 1: Same relation type but one negated, one not
            by_type: dict[str, list[dict]] = defaultdict(list)
            for r in rels:
                by_type[r["relation_type"]].append(r)

            for rtype, rtype_rels in by_type.items():
                negated = [r for r in rtype_rels if r.get("negated", False)]
                affirmed = [r for r in rtype_rels if not r.get("negated", False)]
                if negated and affirmed:
                    desc = (
                        f"NEGATION CONFLICT on '{rtype}' between "
                        f"{pair_key[0]} and {pair_key[1]}: "
                        f"affirmed by [{', '.join(r['source_doc_id'] for r in affirmed)}] "
                        f"vs negated by [{', '.join(r['source_doc_id'] for r in negated)}]"
                    )
                    entity.contradictions.append(desc)
                    contradiction_count += 1

            # Check 2: Contradictory relation types for the same pair
            rel_types_in_pair = {r["relation_type"] for r in rels if not r.get("negated", False)}
            for contra_set in CONTRADICTORY_PAIRS:
                overlap = rel_types_in_pair & contra_set
                if len(overlap) >= 2:
                    involved = [r for r in rels if r["relation_type"] in overlap and not r.get("negated", False)]
                    desc = (
                        f"CONTRADICTORY RELATIONS {overlap} between "
                        f"{pair_key[0]} and {pair_key[1]}: "
                        + "; ".join(
                            f"'{r['relation_type']}' from {r['source_doc_id']} "
                            f"(reliability: {r['source_reliability']})"
                            for r in involved
                        )
                    )
                    entity.contradictions.append(desc)
                    contradiction_count += 1

    return contradiction_count
