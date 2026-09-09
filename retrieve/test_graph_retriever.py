import os
import json
import tempfile
import collections

from retrieve.graph_retriever import GraphRetriever

def test_retriever():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "graph.json")
        chunks_path = os.path.join(tmpdir, "chunks.jsonl")
        
        # 1. Create fake chunks.jsonl
        # Include evidence deduplication and missing chunk tests
        with open(chunks_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"chunk_id": "c1", "chunk_text": "text1", "doc_id": "docA"}) + "\n")
            f.write(json.dumps({"chunk_id": "c2", "chunk_text": "text2", "doc_id": "docB"}) + "\n")
            f.write(json.dumps({"chunk_id": "c3", "chunk_text": "text3", "doc_id": "docC"}) + "\n")
            # c4 is deliberately missing for missing chunk warning
            
        # 2. Create fake graph.json
        graph_data = {
            "metadata": {"schema_version": "1.0", "entity_count": 5, "relation_count": 6},
            "entities": {
                "Entity Alpha": {
                    "canonical_name": "Entity Alpha",
                    "entity_type": "Component",
                    "aliases": ["A", "The Alpha", "is"]
                },
                "Entity Beta": {
                    "canonical_name": "Entity Beta",
                    "entity_type": "Component",
                    "aliases": []
                },
                "Entity Gamma": {
                    "canonical_name": "Entity Gamma",
                    "entity_type": "Component",
                    "aliases": []
                },
                "Entity Delta": {
                    "canonical_name": "Entity Delta",
                    "entity_type": "Component",
                    "aliases": []
                },
                "Entity Epsilon": {
                    "canonical_name": "Entity Epsilon",
                    "entity_type": "Component",
                    "aliases": ["EPS"]
                }
            },
            "relations": [
                # Leads (Alpha -> Beta)
                {
                    "source_entity": "Entity Alpha",
                    "relation_type": "leads",
                    "target_entity": "Entity Beta",
                    "source_chunk_id": "c1",
                    "source_reliability": "medium"
                },
                # Incoming traversal (Gamma -> Beta)
                {
                    "source_entity": "Entity Gamma",
                    "relation_type": "follows",
                    "target_entity": "Entity Beta",
                    "source_chunk_id": "c2",
                    "source_reliability": "high"
                },
                # Cycle (Beta -> Alpha)
                {
                    "source_entity": "Entity Beta",
                    "relation_type": "cycle_test",
                    "target_entity": "Entity Alpha",
                    "source_chunk_id": "c3",
                    "source_reliability": "low"
                },
                # Duplicate edge (same as above but different chunk)
                {
                    "source_entity": "Entity Beta",
                    "relation_type": "cycle_test",
                    "target_entity": "Entity Alpha",
                    "source_chunk_id": "c3",
                    "source_reliability": "high"
                },
                # Negated edge
                {
                    "source_entity": "Entity Alpha",
                    "relation_type": "dislikes",
                    "target_entity": "Entity Delta",
                    "source_chunk_id": "c2",
                    "source_reliability": "high",
                    "negated": True
                },
                # Edge with missing chunk
                {
                    "source_entity": "Entity Delta",
                    "relation_type": "knows",
                    "target_entity": "Entity Epsilon",
                    "source_chunk_id": "c4", # Missing
                    "source_reliability": "low"
                }
            ]
        }
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f)
            
        # 1. Initialization
        retriever = GraphRetriever(graph_path=graph_path, chunks_path=chunks_path)
        
        # 2. Exact Canonical Matching
        # 4. Case-insensitive matching
        res = retriever.retrieve("where is entity alpha located?")
        assert res["seed_entities"][0]["canonical_name"] == "Entity Alpha"
        assert res["seed_entities"][0]["method"] == "exact"
        
        # 3. Alias matching
        res = retriever.retrieve("What about The Alpha?")
        assert res["seed_entities"][0]["canonical_name"] == "Entity Alpha"
        assert res["seed_entities"][0]["method"] == "alias"
        
        # 1-char alias and stopword rejection
        res = retriever.retrieve("A is something.")
        assert all(s["method"] != "alias" for s in res["seed_entities"]), "Should not match 1-char or stopword aliases"
        
        # 5. Substring matching
        res = retriever.retrieve("Tell me about the Entity Alphas")
        assert res["seed_entities"][0]["canonical_name"] == "Entity Alpha"
        assert res["seed_entities"][0]["method"] == "substring"
        
        # 6. Fuzzy matching above threshold
        res = retriever.retrieve("Enetty Alphha", fuzzy_threshold=80)
        assert res["seed_entities"][0]["canonical_name"] == "Entity Alpha"
        assert res["seed_entities"][0]["method"] == "fuzzy"
        
        # 7. Fuzzy rejection below threshold
        res = retriever.retrieve("Enetty Alphha", fuzzy_threshold=99)
        assert len(res["seed_entities"]) == 0
        
        # 8. Exact match prevents unrelated fuzzy padding
        res = retriever.retrieve("Entity Alpha")
        methods = [s["method"] for s in res["seed_entities"]]
        assert "fuzzy" not in methods
        
        # 9. One-hop traversal
        res = retriever.retrieve("Tell me about The Alpha", max_hops=1)
        visited = {p["nodes"][-1] for p in res["paths"]}
        assert "Entity Beta" in visited
        assert "Entity Delta" in visited
        assert "Entity Gamma" not in visited # Gamma is 2 hops away (Alpha -> Beta <- Gamma)
        
        # 10. Two-hop traversal
        res = retriever.retrieve("Tell me about The Alpha", max_hops=2)
        visited2 = {p["nodes"][-1] for p in res["paths"]}
        assert "Entity Gamma" in visited2
        
        # 11. Incoming traversal preserves factual direction
        # From Alpha (seed) -> Beta (hop 1) <- Gamma (hop 2)
        # Check that Gamma -> Beta edge has Gamma as source
        gamma_rel = next(r for r in res["relations"] if r["source_entity"] == "Entity Gamma")
        assert gamma_rel["target_entity"] == "Entity Beta"
        
        # 12. Cycle prevention
        # The path Alpha -> Beta -> Alpha shouldn't loop infinitely
        assert len(res["paths"]) < 20 # Bounded, didn't explode
        
        # 13. Edge deduplication
        # Beta -> Alpha via c3 has two entries in JSON (with different reliability), should both be parsed, but they are exact duplicates except reliability? No, if edge_hash is same, they are deduplicated.
        beta_alpha = [r for r in res["relations"] if r["source_entity"] == "Entity Beta" and r["target_entity"] == "Entity Alpha"]
        assert len(beta_alpha) == 1
        
        # 14. Negated relations appear only in conflicts
        assert any(r["source_entity"] == "Entity Alpha" and r["target_entity"] == "Entity Delta" for r in res["conflicts"])
        assert not any(r["source_entity"] == "Entity Alpha" and r["target_entity"] == "Entity Delta" for r in res["relations"])
        
        # 15. Evidence deduplication
        # 16. Multiple supporting relations per evidence chunk
        # c2 is used by Gamma->Beta (relations) and Alpha->Delta (conflicts)
        c2_ev = [e for e in res["evidence"] if e["chunk_id"] == "c2"]
        assert len(c2_ev) == 1
        
        # 17. Reliability prioritization
        # Gamma->Beta is 'high', Alpha->Beta is 'medium'. High should appear first in relations
        assert res["relations"][0]["source_reliability"] == "high"
        
        # 18. max_hops
        res_0 = retriever.retrieve("Tell me about The Alpha", max_hops=0)
        assert len(res_0["relations"]) == 0
        
        # 19. max_edges
        res_edges = retriever.retrieve("Tell me about The Alpha", max_hops=2, max_edges=1)
        assert (len(res_edges["relations"]) + len(res_edges["conflicts"])) == 1
        
        # 20. max_evidence
        res_ev = retriever.retrieve("Tell me about The Alpha", max_hops=2, max_evidence=1)
        assert len(res_ev["evidence"]) == 1
        
        # 21. Missing chunk warning
        res_missing = retriever.retrieve("Entity Delta", max_hops=1)
        # Delta -> Epsilon uses missing chunk c4
        assert any("Missing chunk ID: c4" in w for w in res_missing["warnings"])
        
        # 22. Valid no-match result
        res_none = retriever.retrieve("zzzxxyy nonexistent", fuzzy_threshold=95)
        assert res_none["seed_entities"] == []
        assert res_none["paths"] == []
        assert res_none["evidence"] == []
        assert "No graph entity matched the question." in res_none["warnings"]
        
        # 23. Missing graph error
        try:
            GraphRetriever(graph_path="missing_graph.json", chunks_path=chunks_path)
            assert False, "Should raise FileNotFoundError for missing graph"
        except FileNotFoundError:
            pass
            
        # 24. Missing chunks error
        try:
            GraphRetriever(graph_path=graph_path, chunks_path="missing_chunks.jsonl")
            assert False, "Should raise FileNotFoundError for missing chunks"
        except FileNotFoundError:
            pass
            
        # 25. no API or network dependency - tested externally by offline assertion
        
        print("PASS: all tests")

if __name__ == "__main__":
    test_retriever()
