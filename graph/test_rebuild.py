import os
import json
import tempfile

from graph.rebuild import rebuild_graph
from graph.graph_store import load_graph

def test_rebuild():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "extractions.jsonl")
        graph_out = os.path.join(tmpdir, "graph.json")
        vault_dir = os.path.join(tmpdir, "vault")
        stats_out = os.path.join(tmpdir, "stats.json")
        
        # 1. Create temporary extraction JSONL file
        # 2. Include two valid extraction records
        # 4. Include at least two entities, one depends_on relation, provenance, chunk_meta
        records = [
            {
                "entities": [
                    {"name": "Entity A", "type": "Component"},
                    {"name": "Entity B", "type": "Component"}
                ],
                "relations": [
                    {
                        "source_entity": "Entity A",
                        "relation_type": "depends_on",
                        "target_entity": "Entity B",
                        "source_chunk_id": "c1",
                        "source_doc_id": "d1",
                        "source_reliability": "high"
                    }
                ],
                "chunk_meta": {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "source_reliability": "high",
                    "page_number": 1
                }
            },
            {
                "entities": [
                    {"name": "Entity C", "type": "Component"}
                ],
                "relations": [],
                "chunk_meta": {
                    "chunk_id": "c2",
                    "doc_id": "d1",
                    "source_reliability": "high",
                    "page_number": 2
                }
            }
        ]
        
        with open(input_path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            # 3. Include one malformed JSON line
            f.write("this is a malformed line\n")
            f.write(" \n") # empty line should be ignored
            
        # Capture timestamp before to verify file remains unchanged
        mtime_before = os.path.getmtime(input_path)
        
        # 5. Call rebuild_graph directly
        summary = rebuild_graph(
            input_path=input_path,
            graph_output_path=graph_out,
            vault_output_dir=vault_dir,
            stats_output_path=stats_out
        )
        
        # 6. Assertions
        assert summary["valid_records"] == 2
        assert summary["malformed_records"] == 1
        assert os.path.exists(graph_out)
        
        graph_data = load_graph(graph_out)
        assert graph_data["metadata"]["entity_count"] == 3
        assert graph_data["metadata"]["relation_count"] == 1
        
        rel = graph_data["relations"][0]
        assert rel["source_chunk_id"] == "c1"
        assert rel["source_doc_id"] == "d1"
        assert rel["source_reliability"] == "high"
        
        # Vault notes
        assert summary["notes_written"] == 3
        assert os.path.exists(os.path.join(vault_dir, "Entity A.md"))
        assert os.path.exists(os.path.join(vault_dir, "Entity B.md"))
        
        # Statistics file exists
        assert os.path.exists(stats_out)
        
        # Input file remains unchanged
        mtime_after = os.path.getmtime(input_path)
        assert mtime_before == mtime_after, "Input file was modified"
        
        # 7. Test missing input raises FileNotFoundError
        try:
            rebuild_graph(os.path.join(tmpdir, "missing.jsonl"), graph_out, vault_dir, stats_out)
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass
            
        # 8. Test file with no valid records raises ValueError
        empty_path = os.path.join(tmpdir, "empty.jsonl")
        with open(empty_path, 'w', encoding='utf-8') as f:
            f.write("invalid line 1\n")
        try:
            rebuild_graph(empty_path, graph_out, vault_dir, stats_out)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
            
        print("PASS: all test cases for rebuild.py")

if __name__ == "__main__":
    test_rebuild()
