"""
Entity and relation type definitions for the Ashen Era knowledge graph.

Design decisions (user-approved):
- Single Component type; use composed_of/part_of for system hierarchy.
- Relations carry a `negated` boolean for explicit negation capture.
- Confidence is 0.0–1.0; everything is kept, <0.4 tagged [low confidence].
  Filtering happens at retrieval time, not extraction time.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------
class EntityType(str, Enum):
    CHARACTER = "Character"
    FACTION = "Faction"
    LOCATION = "Location"
    ARTIFACT = "Artifact"
    EVENT = "Event"
    COMPONENT = "Component"       # Covers both parts and systems (hierarchy via relations)
    MATERIAL = "Material"
    CONCEPT = "Concept"
    CREATURE = "Creature"
    TITLE = "Title"
    DOCUMENT = "Document"
    DATE = "Date"
    CONDITION = "Condition"
    MEASUREMENT = "Measurement"


# ---------------------------------------------------------------------------
# Relation types (22)
# ---------------------------------------------------------------------------
class RelationType(str, Enum):
    # Organizational / Social
    MEMBER_OF = "member_of"
    LEADS = "leads"
    LED_BY = "led_by"
    ALLIED_WITH = "allied_with"
    ENEMY_OF = "enemy_of"
    REPORTS_TO = "reports_to"
    CREATED_BY = "created_by"

    # Spatial
    LOCATED_IN = "located_in"
    CONTAINS = "contains"
    ADJACENT_TO = "adjacent_to"
    ORIGINATES_FROM = "originates_from"

    # Temporal / Causal
    CAUSED_BY = "caused_by"
    CAUSES = "causes"
    SUCCEEDED_BY = "succeeded_by"
    PRECEDED_BY = "preceded_by"
    OCCURRED_DURING = "occurred_during"
    PARTICIPATES_IN = "participates_in"

    # Technical / Dependency
    DEPENDS_ON = "depends_on"
    REQUIRED_BY = "required_by"
    COMPOSED_OF = "composed_of"
    PART_OF = "part_of"
    POWERS = "powers"
    POWERED_BY = "powered_by"
    CONTROLS = "controls"
    CONTROLLED_BY = "controlled_by"

    # Descriptive / General
    POSSESSES = "possesses"
    POSSESSED_BY = "possessed_by"
    HAS_PROPERTY = "has_property"
    REFERS_TO = "refers_to"
    RELATED_TO = "related_to"
    CONTRADICTS = "contradicts"


# ---------------------------------------------------------------------------
# Extracted data structures
# ---------------------------------------------------------------------------
@dataclass
class ExtractedEntity:
    name: str
    entity_type: str          # Should match an EntityType value
    aliases: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class ExtractedRelation:
    source_entity: str
    relation_type: str        # Should match a RelationType value
    target_entity: str        # Must be a real named entity — empty for has_property
    negated: bool             # True if the text explicitly negates this relation
    confidence: float         # 0.0 – 1.0
    justification: str        # Short quote or paraphrase from the source text
    property_value: str = ""  # For has_property only: the descriptive quality/attribute
    source_chunk_id: str = ""
    source_doc_id: str = ""
    source_reliability: str = ""  # high / medium / low

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# JSON schema for the LLM output (embedded in the system prompt)
# ---------------------------------------------------------------------------
ENTITY_TYPE_VALUES = [e.value for e in EntityType]
RELATION_TYPE_VALUES = [r.value for r in RelationType]

LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Canonical name of the entity"},
                    "entity_type": {"type": "string", "enum": ENTITY_TYPE_VALUES},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Alternative names or abbreviations"
                    }
                },
                "required": ["name", "entity_type"]
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity": {"type": "string"},
                    "relation_type": {"type": "string", "enum": RELATION_TYPE_VALUES},
                    "target_entity": {
                        "type": "string",
                        "description": "The target entity name. Must be a real named entity matching one of the 14 entity types. For has_property, leave this as an EMPTY STRING and put the descriptive value in property_value instead."
                    },
                    "property_value": {
                        "type": "string",
                        "description": "ONLY for has_property relations: the descriptive quality, attribute, or characteristic (e.g. 'vast scale', 'extreme heat', 'difficult to navigate'). Leave empty for all other relation types."
                    },
                    "negated": {
                        "type": "boolean",
                        "description": "True if the text explicitly negates this relation (e.g. 'X is NOT a member of Y')"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "How confident you are in this extraction"
                    },
                    "justification": {
                        "type": "string",
                        "description": "A short quote or close paraphrase from the text supporting this relation"
                    }
                },
                "required": ["source_entity", "relation_type",
                             "negated", "confidence", "justification"]
            }
        }
    },
    "required": ["entities", "relations"]
}


def get_schema_description() -> str:
    """Return a human-readable description of the schema for the LLM system prompt."""
    entity_list = ", ".join(ENTITY_TYPE_VALUES)
    relation_list = ", ".join(RELATION_TYPE_VALUES)
    return (
        f"Entity types: {entity_list}\n\n"
        f"Relation types: {relation_list}\n\n"
        "Notes:\n"
        "- Component covers both individual parts AND whole systems; use "
        "composed_of/part_of to represent hierarchy.\n"
        "- Set 'negated' to true ONLY when the text explicitly states a "
        "relation does NOT hold (e.g., 'Kael is not a member of the Iron Covenant'). "
        "Default is false.\n"
        "- Use 'related_to' only as a last resort when no other relation type fits.\n"
        "- Confidence: 1.0 = explicitly stated, 0.7–0.9 = strongly implied, "
        "0.4–0.7 = somewhat implied, <0.4 = speculative.\n"
        "- IMPORTANT — has_property: the 'target_entity' field must ONLY contain "
        "real named entities (matching one of the 14 entity types). For has_property "
        "relations, put the descriptive quality/attribute in 'property_value' instead "
        "and leave 'target_entity' as an empty string. Example: "
        "source_entity='Ashen Wastes', relation_type='has_property', target_entity='', "
        "property_value='vast scale'. Do NOT create entities for descriptive phrases "
        "like 'vast scale' or 'difficult to escape'.\n"
    )


import re as _re

# Words that are clearly prompt/instruction leakage, not real entity names
_PROMPT_LEAK_PATTERNS = [
    "any faction", "any entity", "any character", "any location",
    "any named", "any other", "named power", "named entity",
    "one of the", "matching one of", "14 entity types",
]

def _is_proper_name(text: str) -> bool:
    """
    Heuristic: does this string look like a proper noun / named entity
    rather than a common-noun descriptive phrase?

    Returns True for things like "Iron Covenant", "Kael Ashborn", "Vault 7".
    Returns False for things like "endurance", "vast scale", "severe terrain",
    "canonical temperament", "any faction, house, order, or other named power".
    """
    text = text.strip()
    if not text:
        return False

    # Check for prompt leakage patterns
    text_lower = text.lower()
    for pattern in _PROMPT_LEAK_PATTERNS:
        if pattern in text_lower:
            return False

    # Reject if it's a single common lowercase word (e.g. "endurance", "vigilance")
    words = text.split()
    if len(words) == 1 and text[0].islower():
        return False

    # Reject if entirely lowercase (e.g. "severe terrain", "difficult to escape")
    if text == text.lower():
        return False

    # Reject if it reads like a sentence/clause (contains common stop-phrase patterns)
    clause_markers = [" to ", " of the ", " that ", " which ", " or other ", " is ", " are "]
    for marker in clause_markers:
        if marker in text_lower and len(words) > 3:
            return False

    # Reject if excessively long — real entity names are rarely >6 words
    if len(words) > 8:
        return False

    return True


def validate_extraction(data: dict) -> tuple[bool, str]:
    """
    Validate that the LLM output conforms to our schema.
    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Response is not a JSON object"

    if "entities" not in data:
        return False, "Missing 'entities' key"
    if "relations" not in data:
        return False, "Missing 'relations' key"

    if not isinstance(data["entities"], list):
        return False, "'entities' is not an array"
    if not isinstance(data["relations"], list):
        return False, "'relations' is not an array"

    valid_entity_names = set()
    for i, ent in enumerate(data["entities"]):
        if "name" not in ent:
            return False, f"Entity {i} missing 'name'"
        if "entity_type" not in ent:
            return False, f"Entity {i} missing 'entity_type'"
        if ent["entity_type"] not in ENTITY_TYPE_VALUES:
            return False, f"Entity {i} has invalid entity_type: {ent['entity_type']}"
        
        name_val = ent["name"].strip()
        if not _is_proper_name(name_val):
            return False, f"Entity {i} name '{name_val}' looks like a lowercase common noun or descriptive phrase. If it is a real entity, capitalize it as a proper noun."

        valid_entity_names.add(name_val.lower())
        for alias in ent.get("aliases", []):
            valid_entity_names.add(alias.lower())

    for i, rel in enumerate(data["relations"]):
        for req in ["source_entity", "relation_type",
                     "negated", "confidence", "justification"]:
            if req not in rel:
                return False, f"Relation {i} missing '{req}'"
        
        src = str(rel["source_entity"]).strip()
        if src.lower() not in valid_entity_names:
            return False, f"Relation {i} source_entity '{src}' is not defined in the entities list."

        if rel["relation_type"] not in RELATION_TYPE_VALUES:
            return False, f"Relation {i} has invalid relation_type: {rel['relation_type']}"
        if not isinstance(rel["negated"], bool):
            return False, f"Relation {i} 'negated' is not a boolean"
        if not isinstance(rel["confidence"], (int, float)):
            return False, f"Relation {i} 'confidence' is not a number"
        if not (0.0 <= rel["confidence"] <= 1.0):
            return False, f"Relation {i} 'confidence' out of range [0, 1]"

        # --- has_property: property_value required, target_entity must be empty ---
        if rel["relation_type"] == "has_property":
            if not rel.get("property_value", "").strip():
                return False, (
                    f"Relation {i} is has_property but missing or empty 'property_value'"
                )
            # Reject if target_entity is still populated with a descriptive phrase
            te = rel.get("target_entity", "").strip()
            if te:
                return False, (
                    f"Relation {i} is has_property but target_entity is '{te}' — "
                    f"move this to property_value and set target_entity to empty string"
                )
        else:
            # --- All other relation types: require proper-name target_entity ---
            te = rel.get("target_entity", "").strip()
            if not te:
                return False, (
                    f"Relation {i} ({rel['relation_type']}) has empty 'target_entity' "
                    f"— only has_property may omit target_entity"
                )
            if not _is_proper_name(te):
                return False, (
                    f"Relation {i} target_entity '{te}' looks like a descriptive phrase, "
                    f"not a proper named entity. If this is a property/quality, use "
                    f"has_property with property_value instead. If it IS a real entity, "
                    f"capitalize it as a proper noun."
                )
            if te.lower() not in valid_entity_names:
                return False, f"Relation {i} target_entity '{te}' is not defined in the entities list."

    return True, ""

