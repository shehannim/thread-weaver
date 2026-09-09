import os
import json
import tempfile
from graph.aggregate import MergedEntity
from graph.graph_store import export_graph, load_graph

def run_tests():
    # Construct an in-memory entity index without hitting corpus or env vars
    ent_a = MergedEntity(
        canonical_name="Entity A",
        entity_type="Component",
        aliases={"A_alias"},
        mentions=[{"chunk_id": "c1", "doc_id": "d1", "reliability": "high", "page_number": 1}],
        properties=[{"property_value": "shiny", "negated": False, "confidence": 0.9, "justification": "It shines", "source_chunk_id": "c1", "source_doc_id": "d1", "source_reliability": "high"}],
        relations=[{
            "source_entity": "Entity A",
            "relation_type": "depends_on",
            "target_entity": "Entity B",
            "negated": False,
            "confidence": 0.9,
            "justification": "Evidence",
            "source_chunk_id": "c1",
            "source_doc_id": "d1",
            "source_reliability": "high"
        }]
    )
    
    ent_b = MergedEntity(
        canonical_name="Entity B",
        entity_type="Component",
        aliases=set(),
        mentions=[{"chunk_id": "c2", "doc_id": "d2", "reliability": "high", "page_number": 1}],
        properties=[],
        relations=[]
    )
    
    entity_index = {"Entity A": ent_a, "Entity B": ent_b}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "graph.json")
        out_path2 = os.path.join(tmpdir, "graph2.json")
        
        # 1. Export creates a JSON file.
        summary = export_graph(entity_index, out_path)
        assert os.path.exists(out_path), "JSON file was not created"
        
        # 2. Load returns the required top-level structure.
        data = load_graph(out_path)
        assert isinstance(data, dict)
        assert "metadata" in data
        assert "entities" in data
        assert "relations" in data
        
        # 3. Entity count survives round-trip.
        assert summary["entity_count"] == 2
        assert data["metadata"]["entity_count"] == 2
        assert len(data["entities"]) == 2
        
        # 4. Relation count survives round-trip.
        assert summary["relation_count"] == 1
        assert data["metadata"]["relation_count"] == 1
        assert len(data["relations"]) == 1
        
        # 5. Alias survives round-trip.
        assert "A_alias" in data["entities"]["Entity A"]["aliases"]
        
        # 6. Mention provenance survives round-trip.
        assert data["entities"]["Entity A"]["mentions"][0]["chunk_id"] == "c1"
        
        # 7. Property survives round-trip.
        assert data["entities"]["Entity A"]["properties"][0]["property_value"] == "shiny"
        
        # 8. Relation provenance survives round-trip.
        rel = data["relations"][0]
        assert rel["source_chunk_id"] == "c1"
        assert rel["source_doc_id"] == "d1"
        
        # 9. Confidence survives round-trip.
        assert rel["confidence"] == 0.9
        
        # 10. Reliability survives round-trip.
        assert rel["source_reliability"] == "high"
        
        # 11. A second export of identical data produces semantically identical JSON.
        export_graph(entity_index, out_path2)
        with open(out_path, 'r', encoding='utf-8') as f1, open(out_path2, 'r', encoding='utf-8') as f2:
            assert f1.read() == f2.read(), "Second export produced different JSON text"
            
        # 12. export_graph does not mutate the original relation dictionary.
        assert isinstance(entity_index["Entity A"].aliases, set), "Original alias set was mutated"
        assert len(entity_index["Entity A"].relations) == 1
        
        # 13. Missing graph file raises FileNotFoundError.
        try:
            load_graph(os.path.join(tmpdir, "does_not_exist.json"))
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
            
        # 14. Invalid top-level JSON raises ValueError.
        bad_json = os.path.join(tmpdir, "bad1.json")
        with open(bad_json, 'w') as f:
            f.write("[]")
        try:
            load_graph(bad_json)
            assert False, "Should have raised ValueError for non-dict JSON"
        except ValueError:
            pass
            
        # 15. Incorrect metadata counts raise ValueError.
        bad_json2 = os.path.join(tmpdir, "bad2.json")
        bad_data = {
            "metadata": {"entity_count": 99, "relation_count": 1},
            "entities": {"A": {}},
            "relations": [{"source_entity": "A", "relation_type": "x", "target_entity": "B"}]
        }
        with open(bad_json2, 'w') as f:
            json.dump(bad_data, f)
        try:
            load_graph(bad_json2)
            assert False, "Should have raised ValueError for incorrect counts"
        except ValueError:
            pass
            
        print("PASS: all test cases for graph_store.py")

if __name__ == "__main__":
    run_tests()
