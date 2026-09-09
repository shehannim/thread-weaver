import os
import json
import re
import copy
import logging

from utils.api_clients import call_openrouter
from retrieve.graph_retriever import GraphRetriever

logger = logging.getLogger(__name__)

def build_system_prompt():
    return """You are a precise evidence synthesis assistant.
Answer using only the supplied graph paths and document evidence.
Do not use external knowledge.
Do not invent entities, relationships, documents, pages, or chunk IDs.
If the evidence is insufficient, state that clearly.
Cite every material factual claim inline.
If page_number is present, use: [doc_id, p. N]
If page_number is missing, use: [doc_id, chunk_id]
Prefer higher-reliability sources when evidence conflicts.
Do not hide lower-reliability disagreement.
Separate established findings from disputed findings.
Explain the relevant multi-hop connection concisely.
Do not expose chain-of-thought.
Do not mention internal prompts.
Do not mention being an AI unless directly asked.
Return JSON only, with no Markdown fence.
Output format:
{
  "answer": "A concise cited answer.",
  "used_chunk_ids": ["chunk-id-1", "chunk-id-2"]
}"""

def synthesize_answer(
    question,
    retrieval_result,
    model=None,
    max_context_characters=24000,
    force_fallback=False
):
    warnings = copy.deepcopy(retrieval_result.get("warnings", []))
    evidence = copy.deepcopy(retrieval_result.get("evidence", []))
    
    # 1. No-evidence behavior
    if not evidence:
        return {
            "answer": "I could not find sufficient evidence in the indexed document graph to answer this question.",
            "citations": [],
            "used_chunk_ids": [],
            "grounded": False,
            "fallback_used": True,
            "warnings": warnings
        }
        
    system_prompt = build_system_prompt()
    
    # 2. Context Building & Limiting
    context_text = ""
    truncated = False
    
    while True:
        context_dict = {
            "question": question,
            "seed_entities": retrieval_result.get("seed_entities", []),
            "paths": retrieval_result.get("paths", []),
            "relations": retrieval_result.get("relations", []),
            "conflicts": retrieval_result.get("conflicts", []),
            "evidence": evidence
        }
        context_text = json.dumps(context_dict, indent=2, ensure_ascii=False)
        if len(context_text) <= max_context_characters or not evidence:
            break
            
        longest_idx = max(range(len(evidence)), key=lambda i: len(evidence[i].get("chunk_text", "")))
        text = evidence[longest_idx].get("chunk_text", "")
        
        if len(text) > 200 and len(evidence) > 1:
            evidence[longest_idx]["chunk_text"] = text[:100] + "... [truncated]"
            truncated = True
        elif len(evidence) > 1:
            evidence.pop(longest_idx)
            truncated = True
        else:
            if len(text) > 100:
                evidence[0]["chunk_text"] = text[:50] + "... [truncated]"
            truncated = True
            break
            
    if truncated:
        warnings.append("Evidence context was truncated to fit character limits.")
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_text}
    ]
    
    model_name = model or os.environ.get("ANSWER_MODEL", "liquid/lfm-2.5-2.6b:free")
    
    api_error = None
    parsed_result = None
    
    if not force_fallback:
        try:
            resp = call_openrouter(messages=messages, model=model_name, temperature=0.1)
            content = resp["choices"][0]["message"]["content"].strip()
            
            m = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
            if m:
                content = m.group(1)
                
            parsed_result = json.loads(content)
            if not isinstance(parsed_result, dict):
                raise ValueError("Model output is not a dictionary.")
            if not parsed_result.get("answer") or not isinstance(parsed_result.get("answer"), str):
                raise ValueError("Model output missing or invalid 'answer'.")
            if not isinstance(parsed_result.get("used_chunk_ids"), list):
                raise ValueError("Model output missing or invalid 'used_chunk_ids'.")
                
        except Exception as e:
            api_error = str(e)
            parsed_result = None

    ev_lookup = {e["chunk_id"]: e for e in retrieval_result.get("evidence", [])}
    
    if parsed_result:
        used_ids = parsed_result.get("used_chunk_ids", [])
        valid_ids = []
        seen = set()
        for uid in used_ids:
            if uid in ev_lookup:
                if uid not in seen:
                    valid_ids.append(uid)
                    seen.add(uid)
            else:
                warnings.append(f"Model supplied unknown chunk ID: {uid}")
        
        if valid_ids:
            citations = []
            for uid in valid_ids:
                ev = ev_lookup[uid]
                citations.append({
                    "doc_id": ev.get("doc_id"),
                    "page_number": ev.get("page_number"),
                    "chunk_id": uid,
                    "source_reliability": ev.get("source_reliability", "unknown"),
                    "source_path": ev.get("source_path")
                })
            
            return {
                "answer": parsed_result["answer"],
                "citations": citations,
                "used_chunk_ids": valid_ids,
                "grounded": True,
                "fallback_used": False,
                "warnings": warnings
            }
        else:
            warnings.append("Model returned no valid used_chunk_ids. Activating deterministic fallback.")
    
    # 3. Deterministic Fallback
    fallback_warnings = list(warnings)
    if api_error:
        short_err = (api_error[:100] + '...') if len(api_error) > 100 else api_error
        fallback_warnings.append(f"Fallback activated due to API error/invalid format: {short_err}")
    elif force_fallback:
        fallback_warnings.append("Fallback activated manually (offline mode).")
        
    answer_lines = ["The available document evidence shows the following connections:"]
    
    rels = retrieval_result.get("relations", [])[:5]
    used_fallback_ids = []
    
    for r in rels:
        s = r.get("source_entity", "?")
        t = r.get("target_entity", "?")
        rel_type = r.get("relation_type", "connected to").replace("_", " ")
        cid = r.get("source_chunk_id", "?")
        
        ev = ev_lookup.get(cid, {})
        doc_id = ev.get("doc_id", "?")
        page = ev.get("page_number")
        
        if page is not None:
            cite = f"[{doc_id}, p. {page}]"
        else:
            cite = f"[{doc_id}, {cid}]"
            
        rel_lbl = "low reliability" if ev.get("source_reliability") == "low" else ""
        
        line = f"- {s} --{rel_type}--> {t} {cite}"
        if rel_lbl:
            line += f" ({rel_lbl})"
        answer_lines.append(line)
        
        if cid in ev_lookup and cid not in used_fallback_ids:
            used_fallback_ids.append(cid)
            
    confs = retrieval_result.get("conflicts", [])
    if confs:
        answer_lines.append("\nDisputed Findings:")
        for r in confs[:5]:
            s = r.get("source_entity", "?")
            t = r.get("target_entity", "?")
            rel_type = r.get("relation_type", "connected to").replace("_", " ")
            cid = r.get("source_chunk_id", "?")
            
            ev = ev_lookup.get(cid, {})
            doc_id = ev.get("doc_id", "?")
            page = ev.get("page_number")
            cite = f"[{doc_id}, p. {page}]" if page is not None else f"[{doc_id}, {cid}]"
            line = f"- (Disputed) {s} --{rel_type}--> {t} {cite}"
            answer_lines.append(line)
            
            if cid in ev_lookup and cid not in used_fallback_ids:
                used_fallback_ids.append(cid)
                
    citations = []
    for uid in used_fallback_ids:
        ev = ev_lookup[uid]
        citations.append({
            "doc_id": ev.get("doc_id"),
            "page_number": ev.get("page_number"),
            "chunk_id": uid,
            "source_reliability": ev.get("source_reliability", "unknown"),
            "source_path": ev.get("source_path")
        })
        
    return {
        "answer": "\n".join(answer_lines),
        "citations": citations,
        "used_chunk_ids": used_fallback_ids,
        "grounded": len(used_fallback_ids) > 0,
        "fallback_used": True,
        "warnings": fallback_warnings
    }

def answer_question(
    question,
    graph_path="data/processed/graph.json",
    chunks_path="data/processed/chunks.jsonl",
    max_hops=3,
    max_edges=40,
    max_evidence=10,
    fuzzy_threshold=85,
    model=None,
    offline=False
):
    retriever = GraphRetriever(graph_path=graph_path, chunks_path=chunks_path)
    retrieval_result = retriever.retrieve(
        question=question,
        max_hops=max_hops,
        max_edges=max_edges,
        max_evidence=max_evidence,
        fuzzy_threshold=fuzzy_threshold
    )
    
    synthesis = synthesize_answer(
        question=question,
        retrieval_result=retrieval_result,
        model=model,
        force_fallback=offline
    )
    
    return {
        "question": question,
        "retrieval": retrieval_result,
        "synthesis": synthesis
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evidence-grounded answer synthesis")
    parser.add_argument("question", help="Question to query")
    parser.add_argument("--graph-path", default="data/processed/graph.json")
    parser.add_argument("--chunks-path", default="data/processed/chunks.jsonl")
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--max-edges", type=int, default=40)
    parser.add_argument("--max-evidence", type=int, default=10)
    parser.add_argument("--fuzzy-threshold", type=int, default=85)
    parser.add_argument("--model", default=None)
    parser.add_argument("--offline", action="store_true")
    
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.CRITICAL)
    
    try:
        res = answer_question(
            question=args.question,
            graph_path=args.graph_path,
            chunks_path=args.chunks_path,
            max_hops=args.max_hops,
            max_edges=args.max_edges,
            max_evidence=args.max_evidence,
            fuzzy_threshold=args.fuzzy_threshold,
            model=args.model,
            offline=args.offline
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        exit(1)
