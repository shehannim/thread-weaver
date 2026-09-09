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

Given a text chunk from this universe's documents, extract ALL named entities and ALL relations between them.

## Output Format
You MUST respond with valid JSON and nothing else — no markdown fences, no commentary.
The JSON must conform exactly to this schema:

```json
{json.dumps(LLM_OUTPUT_SCHEMA, indent=2)}
```

## Type Definitions
{get_schema_description()}

## Rules
1. Extract EVERY entity mentioned in the text, even if it only appears once.
2. Extract EVERY relation you can identify — explicit or strongly implied.
3. Each entity name should be its most complete canonical form from the text.
   List shorter forms or abbreviations in the "aliases" array.
4. For the "justification" field, provide a SHORT quote or close paraphrase
   (≤30 words) from the source text that supports the relation.
5. If the text explicitly states a relation does NOT hold, set "negated" to true.
   Example: "Kael never joined the Covenant" → member_of, negated=true.
6. Do NOT invent entities or relations not supported by the text.
7. If no entities or relations are found, return {{"entities": [], "relations": []}}.
8. Use "related_to" ONLY when no other relation type fits.
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
