import json
import os
import tempfile

def export_graph(entity_index, output_path):
    """
    Exports the given entity_index (dict of MergedEntity) to a stable JSON graph.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    entities_out = {}
    
    # Sort entities alphabetically by normalized canonical name
    sorted_entity_names = sorted(entity_index.keys(), key=lambda x: x.lower().strip())
    
    relations_out_set = set()
    relations_out_list = []
    
    for name in sorted_entity_names:
        ent = entity_index[name]
        entities_out[ent.canonical_name] = {
            "canonical_name": ent.canonical_name,
            "entity_type": ent.entity_type,
            "aliases": sorted(list(ent.aliases)),
            "mentions": ent.mentions,
            "properties": ent.properties,
            "contradictions": ent.contradictions
        }
        
        for rel in ent.relations:
            # Deduplicate by hashing a tuple of values, ensuring we only record each factual relation once
            tup = (
                rel.get("source_entity"),
                rel.get("relation_type"),
                rel.get("target_entity"),
                rel.get("negated", False),
                rel.get("confidence", 0.5),
                rel.get("justification", ""),
                rel.get("source_chunk_id", ""),
                rel.get("source_doc_id", ""),
                rel.get("source_reliability", "")
            )
            
            if tup not in relations_out_set:
                relations_out_set.add(tup)
                relations_out_list.append({
                    "source_entity": tup[0],
                    "relation_type": tup[1],
                    "target_entity": tup[2],
                    "negated": tup[3],
                    "confidence": tup[4],
                    "justification": tup[5],
                    "source_chunk_id": tup[6],
                    "source_doc_id": tup[7],
                    "source_reliability": tup[8]
                })
                
    # Export deterministic relation ordering
    relations_out_list.sort(key=lambda r: (
        r["source_entity"] or "",
        r["relation_type"] or "",
        r["target_entity"] or "",
        r["source_doc_id"] or "",
        r["source_chunk_id"] or ""
    ))
    
    graph_dict = {
        "metadata": {
            "schema_version": "1.0",
            "entity_count": len(entities_out),
            "relation_count": len(relations_out_list)
        },
        "entities": entities_out,
        "relations": relations_out_list
    }
    
    # Write atomically
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(output_path), text=True)
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)
            f.flush()
        os.replace(temp_path, output_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
        
    return {
        "output_path": output_path,
        "entity_count": len(entities_out),
        "relation_count": len(relations_out_list)
    }


def load_graph(path):
    """
    Loads the stable JSON graph from disk and validates its structure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON value must be a dictionary")
        
    for key in ["metadata", "entities", "relations"]:
        if key not in data:
            raise ValueError(f"Missing required key: '{key}'")
            
    if not isinstance(data["metadata"], dict):
        raise ValueError("'metadata' must be a dictionary")
        
    if not isinstance(data["entities"], dict):
        raise ValueError("'entities' must be a dictionary")
        
    if not isinstance(data["relations"], list):
        raise ValueError("'relations' must be a list")
        
    for rel in data["relations"]:
        if "source_entity" not in rel or "relation_type" not in rel or "target_entity" not in rel:
            raise ValueError("Relation missing source_entity, relation_type, or target_entity")
            
    meta_entity_count = data["metadata"].get("entity_count")
    if meta_entity_count != len(data["entities"]):
        raise ValueError(f"Metadata entity_count ({meta_entity_count}) does not match entities dictionary length ({len(data['entities'])})")
        
    meta_relation_count = data["metadata"].get("relation_count")
    if meta_relation_count != len(data["relations"]):
        raise ValueError(f"Metadata relation_count ({meta_relation_count}) does not match relations list length ({len(data['relations'])})")
        
    return data
