"""
Write merged entities to an Obsidian-style markdown vault.

One .md file per unique entity in /data/vault/, with:
- YAML frontmatter (entity_type, aliases)
- Relations section with typed links, negation flags, confidence tags, provenance
- Contradictions section (if any)
- Mentions section with source reliability
"""

import json
import os
import re
import logging
from graph.aggregate import MergedEntity

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.4


def sanitize_filename(name: str) -> str:
    """Convert an entity name to a safe filename for the vault."""
    # Replace characters that are problematic in filenames
    safe = re.sub(r'[<>:"/\\|?*]', '_', name)
    safe = safe.strip('. ')
    # Truncate if excessively long
    if len(safe) > 200:
        safe = safe[:200]
    return safe


def format_relation_line(rel: dict) -> str:
    """Format a single relation as an Obsidian-compatible markdown line."""
    target = rel["target_entity"]
    rtype = rel["relation_type"]
    negated = rel.get("negated", False)
    confidence = rel.get("confidence", 0.5)
    justification = rel.get("justification", "")
    doc_id = rel.get("source_doc_id", "unknown")
    chunk_id = rel.get("source_chunk_id", "")
    reliability = rel.get("source_reliability", "unknown")

    # Build the line
    neg_tag = " **[NEGATED]**" if negated else ""
    conf_tag = " `[low confidence]`" if confidence < LOW_CONFIDENCE_THRESHOLD else ""

    line = (
        f"- [[{target}]] (`{rtype}`){neg_tag}{conf_tag} "
        f"— \"{justification}\" "
        f"[source: {doc_id}, reliability: {reliability}]"
    )
    return line


def write_entity_note(entity: MergedEntity, vault_dir: str) -> str:
    """
    Write a single entity's markdown note to the vault directory.
    Returns the filepath written.
    """
    filename = sanitize_filename(entity.canonical_name) + ".md"
    filepath = os.path.join(vault_dir, filename)

    lines = []

    # --- YAML frontmatter ---
    lines.append("---")
    lines.append(f"entity_type: {entity.entity_type}")
    if entity.aliases:
        # Remove the canonical name from aliases if present
        display_aliases = sorted(entity.aliases - {entity.canonical_name})
        if display_aliases:
            aliases_str = json.dumps(display_aliases)
            lines.append(f"aliases: {aliases_str}")
    lines.append("---")
    lines.append("")

    # --- Title ---
    lines.append(f"# {entity.canonical_name}")
    lines.append("")

    # --- Relations ---
    if entity.relations:
        lines.append("## Relations")
        lines.append("")
        # Deduplicate display: group by (target, relation_type, negated) and
        # show each unique triple once, with all sources
        seen_triples = {}
        for rel in entity.relations:
            key = (rel["target_entity"], rel["relation_type"], rel.get("negated", False))
            if key not in seen_triples:
                seen_triples[key] = rel
            else:
                # Append additional source info to the justification
                existing = seen_triples[key]
                existing["justification"] += (
                    f"; also: \"{rel.get('justification', '')}\" "
                    f"[{rel.get('source_doc_id', '')}]"
                )

        for rel in seen_triples.values():
            lines.append(format_relation_line(rel))
        lines.append("")

    # --- Properties (has_property — plain text, not links) ---
    if entity.properties:
        lines.append("## Properties")
        lines.append("")
        # Deduplicate by property_value
        seen_props = {}
        for prop in entity.properties:
            pval = prop.get("property_value", "").strip()
            if not pval:
                continue
            key = (pval, prop.get("negated", False))
            if key not in seen_props:
                seen_props[key] = prop
            else:
                # Merge sources
                existing = seen_props[key]
                existing["justification"] += (
                    f"; also: \"{prop.get('justification', '')}\" "
                    f"[{prop.get('source_doc_id', '')}]"
                )

        for prop in seen_props.values():
            pval = prop["property_value"]
            negated = prop.get("negated", False)
            confidence = prop.get("confidence", 0.5)
            justification = prop.get("justification", "")
            doc_id = prop.get("source_doc_id", "unknown")
            reliability = prop.get("source_reliability", "unknown")

            neg_tag = " **[NEGATED]**" if negated else ""
            conf_tag = " `[low confidence]`" if confidence < LOW_CONFIDENCE_THRESHOLD else ""

            lines.append(
                f"- {pval}{neg_tag}{conf_tag} "
                f"— \"{justification}\" "
                f"[source: {doc_id}, reliability: {reliability}]"
            )
        lines.append("")

    # --- Contradictions ---
    if entity.contradictions:
        lines.append("## Contradictions")
        lines.append("")
        for contra in entity.contradictions:
            lines.append(f"- ⚠️ {contra}")
        lines.append("")

    # --- Mentions ---
    if entity.mentions:
        lines.append("## Mentions")
        lines.append("")
        # Deduplicate mentions
        seen_mentions = set()
        for mention in entity.mentions:
            doc_id = mention.get("doc_id", "unknown")
            chunk_id = mention.get("chunk_id", "")
            reliability = mention.get("source_reliability", "unknown")
            key = (doc_id, chunk_id)
            if key not in seen_mentions:
                seen_mentions.add(key)
                lines.append(
                    f"- {doc_id} ({chunk_id}) [reliability: {reliability}]"
                )
        lines.append("")

    # Write
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


def write_vault(entity_index: dict[str, MergedEntity], vault_dir: str) -> int:
    """
    Write all entities to the vault directory.
    Returns the number of notes written.
    """
    os.makedirs(vault_dir, exist_ok=True)
    count = 0
    for name, entity in entity_index.items():
        try:
            write_entity_note(entity, vault_dir)
            count += 1
        except Exception as e:
            logger.error("Failed to write vault note for '%s': %s", name, e)
    logger.info("Wrote %d entity notes to %s", count, vault_dir)
    return count
