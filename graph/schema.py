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
    target_entity: str
    negated: bool             # True if the text explicitly negates this relation
    confidence: float         # 0.0 – 1.0
    justification: str        # Short quote or paraphrase from the source text
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
                    "target_entity": {"type": "string"},
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
                "required": ["source_entity", "relation_type", "target_entity",
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
    )


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

    for i, ent in enumerate(data["entities"]):
        if "name" not in ent:
            return False, f"Entity {i} missing 'name'"
        if "entity_type" not in ent:
            return False, f"Entity {i} missing 'entity_type'"
        if ent["entity_type"] not in ENTITY_TYPE_VALUES:
            return False, f"Entity {i} has invalid entity_type: {ent['entity_type']}"

    for i, rel in enumerate(data["relations"]):
        for req in ["source_entity", "relation_type", "target_entity",
                     "negated", "confidence", "justification"]:
            if req not in rel:
                return False, f"Relation {i} missing '{req}'"
        if rel["relation_type"] not in RELATION_TYPE_VALUES:
            return False, f"Relation {i} has invalid relation_type: {rel['relation_type']}"
        if not isinstance(rel["negated"], bool):
            return False, f"Relation {i} 'negated' is not a boolean"
        if not isinstance(rel["confidence"], (int, float)):
            return False, f"Relation {i} 'confidence' is not a number"
        if not (0.0 <= rel["confidence"] <= 1.0):
            return False, f"Relation {i} 'confidence' out of range [0, 1]"

    return True, ""
