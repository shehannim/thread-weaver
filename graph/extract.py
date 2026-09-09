"""
LLM-based entity/relation extraction from text chunks.

Uses DeepSeek via OpenRouter. Enforces strict JSON output schema with
up to 2 retries on malformed output before logging failure and skipping.
"""

import json
import logging
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.api_clients import call_openrouter
from graph.schema import (
    get_schema_description,
    validate_extraction,
    LLM_OUTPUT_SCHEMA,
    ExtractedEntity,
    ExtractedRelation,
)

logger = logging.getLogger(__name__)

# DeepSeek model via OpenRouter
EXTRACTION_MODEL = "deepseek/deepseek-chat"
MAX_RETRIES_MALFORMED = 2

SYSTEM_PROMPT = f"""You are a precise knowledge-graph extraction engine for the "Ashen Era" fictional universe.

Given a text chunk, extract ALL named entities and ALL relations between them.

## Output Format
Respond with valid JSON ONLY — no markdown fences, no commentary, no explanation.

## JSON Schema
```json
{json.dumps(LLM_OUTPUT_SCHEMA, indent=2)}
```

## Type Definitions
{get_schema_description()}

## Rules
1. Extract EVERY named entity in the text (proper nouns — people, places, organizations, items, events, etc.).
2. Extract EVERY relation you can identify between named entities.
3. Use the most complete canonical name for each entity. Put abbreviations in "aliases".
4. "justification": a SHORT quote (≤30 words) from the source text.
5. Negation: set "negated" to true only if the text explicitly says a relation does NOT hold.
6. Do NOT invent facts not supported by the text.
7. REFERENTIAL INTEGRITY: Every `source_entity` and `target_entity` used in your relations MUST be defined in your `entities` array above it. Do not reference entities you haven't extracted.
8. If nothing to extract, return {{"entities": [], "relations": []}}.
9. Use "related_to" only as a last resort.

## CRITICAL: target_entity vs property_value

"target_entity" must ALWAYS be a real proper-noun named entity that exists in the world.
It must NEVER be a descriptive phrase, common noun, quality, or abstract attribute.

For descriptive qualities/attributes, use relation_type "has_property":
- Set "target_entity" to "" (empty string)
- Put the quality in "property_value"

For ALL other relation types:
- Set "target_entity" to the real named entity
- Set "property_value" to "" (empty string)

### CORRECT example output:
```json
{{
  "entities": [
    {{"name": "Ashen Wastes", "entity_type": "Location", "aliases": []}},
    {{"name": "Kael Ashborn", "entity_type": "Character", "aliases": ["Kael"]}},
    {{"name": "Iron Covenant", "entity_type": "Faction", "aliases": []}}
  ],
  "relations": [
    {{"source_entity": "Ashen Wastes", "relation_type": "has_property", "target_entity": "", "property_value": "vast scale", "negated": false, "confidence": 0.8, "justification": "the Wastes stretch endlessly"}},
    {{"source_entity": "Kael Ashborn", "relation_type": "has_property", "target_entity": "", "property_value": "exceptional endurance", "negated": false, "confidence": 0.7, "justification": "known for tireless marches"}},
    {{"source_entity": "Kael Ashborn", "relation_type": "member_of", "target_entity": "Iron Covenant", "property_value": "", "negated": false, "confidence": 1.0, "justification": "Kael swore the binding oath"}}
  ]
}}
```

### WRONG relation examples (these will be REJECTED by validation):
- `target_entity: "endurance"` ❌ (Not a named entity, use has_property instead)
- `target_entity: "difficult to navigate"` ❌ (Description, use has_property instead)
- `target_entity: "vigilance"` ❌ (Not a named entity)
- `target_entity: "severe terrain"` ❌ (Not a named entity)
"""


def extract_from_chunk(chunk: dict) -> dict | None:
    """
    Extract entities and relations from a single chunk.

    Args:
        chunk: A dict with at least {chunk_text, chunk_id, doc_id,
               doc_type, source_reliability, page_number}.

    Returns:
        A dict with "entities" and "relations" lists (enriched with
        source provenance), or None if extraction failed after retries.
    """
    chunk_text = chunk.get("chunk_text", "")
    if not chunk_text.strip():
        return {"entities": [], "relations": []}

    user_message = (
        f"Document type: {chunk.get('doc_type', 'unknown')}\n"
        f"Source reliability: {chunk.get('source_reliability', 'unknown')}\n\n"
        f"Text:\n{chunk_text}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    last_error = ""
    for attempt in range(1 + MAX_RETRIES_MALFORMED):
        try:
            response = call_openrouter(
                messages=messages,
                model=EXTRACTION_MODEL,
                temperature=0.1,  # Low temperature for deterministic extraction
            )

            raw_content = response["choices"][0]["message"]["content"]

            # Strip markdown fences if the model wraps its output
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = cleaned.index("\n")
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)

            is_valid, error_msg = validate_extraction(data)
            if not is_valid:
                last_error = f"Schema validation failed: {error_msg}"
                logger.warning(
                    "Attempt %d/%d for chunk %s: %s",
                    attempt + 1, 1 + MAX_RETRIES_MALFORMED,
                    chunk.get("chunk_id", "?"), last_error,
                )
                # Add a correction prompt for the retry
                messages.append({"role": "assistant", "content": raw_content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your output had a schema error: {error_msg}. "
                        "Please fix it and respond with valid JSON only."
                    ),
                })
                continue

            # Enrich relations with source provenance
            for rel in data["relations"]:
                rel["source_chunk_id"] = chunk.get("chunk_id", "")
                rel["source_doc_id"] = chunk.get("doc_id", "")
                rel["source_reliability"] = chunk.get("source_reliability", "")

            return data

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            logger.warning(
                "Attempt %d/%d for chunk %s: %s",
                attempt + 1, 1 + MAX_RETRIES_MALFORMED,
                chunk.get("chunk_id", "?"), last_error,
            )
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": (
                    "Your response was not valid JSON. "
                    "Please respond with ONLY the JSON object, no markdown fences or commentary."
                ),
            })
            continue

        except Exception as e:
            if hasattr(e, 'last_attempt') and e.last_attempt is not None:
                real_err = e.last_attempt.exception()
                last_error = f"API/unexpected error: {real_err}"
            else:
                last_error = f"API/unexpected error: {e}"
            logger.error(
                "Attempt %d/%d for chunk %s: %s",
                attempt + 1, 1 + MAX_RETRIES_MALFORMED,
                chunk.get("chunk_id", "?"), last_error,
            )
            # Don't retry on API errors — tenacity in api_clients already handles transient ones
            break

    logger.error(
        "FAILED extraction for chunk %s after %d attempts. Last error: %s",
        chunk.get("chunk_id", "?"), 1 + MAX_RETRIES_MALFORMED, last_error,
    )
    return None
