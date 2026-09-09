import os
import json
import re
import logging
from thefuzz import fuzz

from graph.graph_store import load_graph

logger = logging.getLogger(__name__)

class GraphRetriever:
    def __init__(
        self,
        graph_path="data/processed/graph.json",
        chunks_path="data/processed/chunks.jsonl"
    ):
        # 1. Path Resolution
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        def resolve_path(p):
            if os.path.isabs(p):
                return p
            return os.path.join(project_root, p)
            
        self.graph_path = resolve_path(graph_path)
        self.chunks_path = resolve_path(chunks_path)
        
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Graph file missing: {self.graph_path}")
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks file missing: {self.chunks_path}")
            
        # 2. Load Graph
        self.graph_data = load_graph(self.graph_path)
        if not self.graph_data.get("entities") or not self.graph_data.get("relations"):
            raise ValueError("Graph is empty or structurally invalid.")
            
        # 3. Load Chunks
        self.chunk_lookup = {}
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        c = json.loads(line)
                        cid = c.get("chunk_id")
                        if cid:
                            self.chunk_lookup[cid] = c
                    except json.JSONDecodeError:
                        pass
                        
        if not self.chunk_lookup:
            raise ValueError("Chunks file is empty or structurally invalid.")
            
        # 4. Initialization Indexes
        self.canon_lookup = {}
        self.norm_canon_lookup = {}
        self.norm_alias_lookup = {}
        self.outgoing = {}
        self.incoming = {}
        
        for cname, ent in self.graph_data["entities"].items():
            self.canon_lookup[cname] = ent
            
            norm_cname = self._normalize(cname)
            if norm_cname:
                self.norm_canon_lookup[norm_cname] = cname
                
            for alias in ent.get("aliases", []):
                norm_al = self._normalize(alias)
                if norm_al:
                    self.norm_alias_lookup[norm_al] = cname
                    
            self.outgoing[cname] = []
            self.incoming[cname] = []
            
        for rel in self.graph_data["relations"]:
            src = rel.get("source_entity")
            tgt = rel.get("target_entity")
            if src in self.outgoing:
                self.outgoing[src].append(rel)
            if tgt in self.incoming:
                self.incoming[tgt].append(rel)
                
    def _normalize(self, text):
        """
        Conservative normalization: casefold, strip whitespace, 
        collapse spaces, replace simple punctuation.
        """
        if not text:
            return ""
        text = text.casefold()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def _get_seeds(self, question, limit=5, fuzzy_threshold=85):
        q_norm = self._normalize(question)
        if not q_norm:
            return []
            
        q_tokens = set(q_norm.split())
        seeds = []
        seen = set()
        
        def add_seed(canonical, matched, method, score):
            if canonical not in seen and len(seeds) < limit:
                seeds.append({
                    "canonical_name": canonical,
                    "matched_text": matched,
                    "method": method,
                    "score": score
                })
                seen.add(canonical)
                
        # Do not select common stop words as aliases
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for", "with", "on", "at", "by", "from", "and", "or"}
        
        q_padded = f" {q_norm} "
        
        # Prefer longer and more specific names
        sorted_canons = sorted(self.norm_canon_lookup.items(), key=lambda x: len(x[0]), reverse=True)
        sorted_aliases = sorted(self.norm_alias_lookup.items(), key=lambda x: len(x[0]), reverse=True)
        
        # 1. Exact canonical-name occurrence in the question
        for norm_can, can in sorted_canons:
            if f" {norm_can} " in q_padded:
                add_seed(can, norm_can, "exact", 100)
                
        # 2. Exact alias occurrence
        for norm_al, can in sorted_aliases:
            # Do not select a one-character alias, nor stop words
            if len(norm_al) > 1 and norm_al not in stop_words:
                if f" {norm_al} " in q_padded:
                    add_seed(can, norm_al, "alias", 100)
                    
        # 3. Case-insensitive canonical-name substring occurrence
        for norm_can, can in sorted_canons:
            if len(norm_can) > 3 and norm_can in q_norm:
                add_seed(can, norm_can, "substring", 90)
                
        # 4. Case-insensitive alias substring occurrence
        for norm_al, can in sorted_aliases:
            if len(norm_al) > 3 and norm_al not in stop_words and norm_al in q_norm:
                add_seed(can, norm_al, "substring", 90)
                
        # 5. Token overlap
        for norm_can, can in sorted_canons:
            can_tokens = set(norm_can.split())
            overlap = can_tokens.intersection(q_tokens) - stop_words
            # Do not let token overlap select unrelated entities based on one generic token (must be long if 1 token, or 2+ tokens)
            if len(overlap) >= 2 or (len(overlap) == 1 and len(list(overlap)[0]) >= 5):
                add_seed(can, norm_can, "token", 80)
                
        # 6. Fuzzy matching only as a final fallback
        # Do not add weak fuzzy matches merely to fill the five-seed limit if strong matches exist.
        if not seeds:
            fuzzy_candidates = []
            for norm_can, can in sorted_canons:
                if len(norm_can) > 2:
                    score = fuzz.partial_ratio(norm_can, q_norm)
                    if score >= fuzzy_threshold:
                        fuzzy_candidates.append((can, norm_can, score))
            
            fuzzy_candidates.sort(key=lambda x: x[2], reverse=True)
            for can, norm_can, score in fuzzy_candidates:
                add_seed(can, norm_can, "fuzzy", score)
                
        return seeds

    def retrieve(self, question, max_hops=3, max_edges=40, max_evidence=12, fuzzy_threshold=85):
        trace = []
        warnings = []
        conflicts = []
        relations = []
        paths = []
        
        trace.append(f"Normalizing question: {question}")
        seeds = self._get_seeds(question, limit=5, fuzzy_threshold=fuzzy_threshold)
        
        if not seeds:
            warnings.append("No graph entity matched the question.")
            
        visited_nodes = set([s["canonical_name"] for s in seeds])
        
        # queue stores (current_node, distance, path_of_nodes)
        queue = [(s["canonical_name"], 0, [s["canonical_name"]]) for s in seeds]
        queue.sort(key=lambda x: x[0])
        
        for s in seeds:
            paths.append({"nodes": [s["canonical_name"]], "hop_count": 0})
            
        collected_edge_hashes = set()
        
        def edge_hash(rel):
            return (rel.get("source_entity"), rel.get("relation_type"), rel.get("target_entity"), rel.get("source_chunk_id"))
            
        def rel_score(r):
            rel_map = {"high": 3, "medium": 2, "low": 1}
            return rel_map.get(r.get("source_reliability", "low"), 0)
            
        while queue and len(relations) + len(conflicts) < max_edges:
            current_node, dist, path_nodes = queue.pop(0)
            
            if dist >= max_hops:
                continue
                
            # Traverse both outgoing and incoming relations, preserve original direction
            rels = self.outgoing.get(current_node, []) + self.incoming.get(current_node, [])
            
            # Reliability prioritization: sort by reliability, then deterministic fields
            rels.sort(key=lambda r: (-rel_score(r), r.get("source_entity",""), r.get("relation_type",""), r.get("target_entity",""), r.get("source_chunk_id", "")))
            
            for rel in rels:
                if len(relations) + len(conflicts) >= max_edges:
                    break
                    
                eh = edge_hash(rel)
                if eh not in collected_edge_hashes:
                    collected_edge_hashes.add(eh)
                    
                    if rel.get("negated", False):
                        conflicts.append(rel)
                    else:
                        relations.append(rel)
                        
                    trace.append(f"Traversed {rel.get('source_entity')} -> {rel.get('target_entity')}")
                    
                    neighbor = rel.get("target_entity") if rel.get("source_entity") == current_node else rel.get("source_entity")
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        new_path = path_nodes + [neighbor]
                        paths.append({"nodes": new_path, "hop_count": dist + 1})
                        queue.append((neighbor, dist + 1, new_path))
                        
        # Final stable sort
        relations.sort(key=lambda r: (-rel_score(r), r.get("source_entity",""), r.get("relation_type",""), r.get("target_entity",""), r.get("source_chunk_id", "")))
                        
        # Extract evidence chunks up to max_evidence limit
        evidence = []
        seen_chunks = set()
        
        for rel in relations + conflicts:
            cid = rel.get("source_chunk_id")
            if cid and cid not in seen_chunks:
                if cid in self.chunk_lookup:
                    evidence.append(self.chunk_lookup[cid])
                    seen_chunks.add(cid)
                else:
                    warnings.append(f"Missing chunk ID: {cid}")
            if len(evidence) >= max_evidence:
                break
                
        metrics = {
            "distinct_source_document_count": len(set(e.get("doc_id") for e in evidence if e.get("doc_id")))
        }
                
        return {
            "question": question,
            "seed_entities": seeds,
            "paths": paths,
            "relations": relations,
            "evidence": evidence,
            "conflicts": conflicts,
            "warnings": warnings,
            "trace": trace,
            "metrics": metrics
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline Multi-Hop Graph Retriever")
    parser.add_argument("question", help="Question to query")
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--max-edges", type=int, default=40)
    parser.add_argument("--max-evidence", type=int, default=12)
    parser.add_argument("--fuzzy-threshold", type=int, default=85)
    
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.CRITICAL)
    
    try:
        retriever = GraphRetriever()
        result = retriever.retrieve(
            args.question,
            max_hops=args.max_hops,
            max_edges=args.max_edges,
            max_evidence=args.max_evidence,
            fuzzy_threshold=args.fuzzy_threshold
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        exit(1)
