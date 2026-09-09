import os
import time
import json
import streamlit as st

# Helper functions for UI
def format_relation_label(label: str) -> str:
    if not label:
        return ""
    return label.replace("_", " ")

def format_citation(citation: dict) -> str:
    doc_id = citation.get("doc_id", "Unknown")
    page = citation.get("page_number")
    chunk_id = citation.get("chunk_id", "Unknown")
    
    if page is not None:
        base = f"[{doc_id}, p. {page}]"
    else:
        base = f"[{doc_id}, {chunk_id}]"
        
    return base

def format_path(path_dict: dict, relations: list) -> str:
    nodes = path_dict.get("nodes", [])
    if not nodes:
        return ""
        
    formatted = [nodes[0]]
    
    for i in range(len(nodes) - 1):
        a = nodes[i]
        b = nodes[i + 1]
        
        # find relation
        rel_type = "connected to"
        direction = "->"
        for r in relations:
            src = r.get("source_entity")
            tgt = r.get("target_entity")
            
            if src == a and tgt == b:
                rel_type = r.get("relation_type", "connected to")
                direction = "->"
                break
            elif src == b and tgt == a:
                rel_type = r.get("relation_type", "connected to")
                direction = "<-"
                break
                
        lbl = format_relation_label(rel_type)
        formatted.append(f"{direction} {lbl}")
        formatted.append(b)
        
    return "\n".join(formatted)

def truncate_excerpt(text: str, max_characters: int = 1500) -> str:
    if not text:
        return ""
    if len(text) <= max_characters:
        return text
    return text[:max_characters] + "... [excerpt truncated]"

# Streamlit App
if __name__ == "__main__":
    from retrieve.graph_retriever import GraphRetriever
    from synth.answer import synthesize_answer

    st.set_page_config(
        page_title="Thread Weaver",
        page_icon="🧵",
        layout="wide"
    )
    
    st.title("Thread Weaver")
    st.subheader("Multi-Hop Knowledge Graph Assistant")
    st.markdown("Thread Weaver connects evidence across documents using entity-guided graph traversal and returns source-grounded answers with citations.")
    
    # Path Resolution
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    graph_path = os.path.join(project_root, "data", "processed", "graph.json")
    chunks_path = os.path.join(project_root, "data", "processed", "chunks.jsonl")
    
    @st.cache_resource
    def load_retriever():
        if not os.path.exists(graph_path):
            return None, "Knowledge graph not found. Run: python -m graph.rebuild --input data/processed/sample_extractions.jsonl"
        if not os.path.exists(chunks_path):
            return None, "Document chunks not found. Run the ingestion pipeline first."
        try:
            return GraphRetriever(graph_path=graph_path, chunks_path=chunks_path), None
        except Exception as e:
            return None, f"Failed to initialize retriever: {str(e)}"
            
    retriever, init_error = load_retriever()
    
    if init_error:
        st.warning(init_error)
        st.stop()
        
    # Sidebar
    st.sidebar.header("Retrieval Settings")
    max_hops = st.sidebar.slider("Maximum hops", min_value=1, max_value=4, value=3)
    max_edges = st.sidebar.slider("Maximum graph edges", min_value=10, max_value=60, value=40, step=5)
    max_evidence = st.sidebar.slider("Maximum evidence chunks", min_value=3, max_value=15, value=10)
    fuzzy_threshold = st.sidebar.slider("Fuzzy match threshold", min_value=70, max_value=100, value=85)
    answer_mode = st.sidebar.radio("Answer mode", ["Offline deterministic", "Online synthesis"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Graph Statistics")
    st.sidebar.text(f"Entity count: {retriever.graph_data['metadata'].get('entity_count', 0)}")
    st.sidebar.text(f"Relation count: {retriever.graph_data['metadata'].get('relation_count', 0)}")
    st.sidebar.text(f"Indexed chunks: {len(retriever.chunk_lookup)}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Source Reliability Legend")
    st.sidebar.markdown("- **High**: preferred authoritative source category\n- **Medium**: supporting narrative or wiki source\n- **Low**: potentially disputed ephemera\n\n*(These are heuristic priorities, not guaranteed truth)*")
    
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
        
    st.markdown("### Sample Questions")
    col1, col2, col3 = st.columns(3)
    if col1.button("The Purge of Blackport"):
        st.session_state["question_input"] = "How is The Purge of Blackport connected to other entities?"
    if col2.button("The Silent Choir"):
        st.session_state["question_input"] = "What is connected to The Silent Choir?"
    if col3.button("Cindermere Hold"):
        st.session_state["question_input"] = "How is Cindermere Hold connected to factions, characters, or locations?"
        
    with st.form("query_form"):
        question = st.text_area("Ask a question that may require connecting facts across documents", value=st.session_state.get("question_input", ""))
        submitted = st.form_submit_button("Weave the Evidence")
        
    if submitted:
        if not question.strip():
            st.warning("Please enter a valid question.")
        else:
            st.session_state["question_input"] = question
            start_time = time.perf_counter()
            with st.spinner("Following evidence threads across the archive..."):
                try:
                    retrieval_result = retriever.retrieve(
                        question=question,
                        max_hops=max_hops,
                        max_edges=max_edges,
                        max_evidence=max_evidence,
                        fuzzy_threshold=fuzzy_threshold
                    )
                    
                    if not retrieval_result.get("seed_entities"):
                        st.session_state["result"] = {"no_match": True, "retrieval": retrieval_result}
                    else:
                        force_fallback = (answer_mode == "Offline deterministic")
                        synth_result = synthesize_answer(
                            question=question,
                            retrieval_result=retrieval_result,
                            force_fallback=force_fallback
                        )
                        elapsed = time.perf_counter() - start_time
                        
                        st.session_state["result"] = {
                            "no_match": False,
                            "retrieval": retrieval_result,
                            "synthesis": synth_result,
                            "time": elapsed
                        }
                except Exception as e:
                    st.error(f"An unexpected error occurred during processing: {str(e)[:200]}")
                    
    if "result" in st.session_state:
        res = st.session_state["result"]
        ret = res.get("retrieval", {})
        syn = res.get("synthesis", {})
        
        if res.get("no_match"):
            st.error("No indexed entity matched this question.")
            for w in ret.get("warnings", []):
                st.warning(w)
            st.info("Try using an exact character, place, artifact, faction, or event name, or try one of the sample questions.")
        else:
            st.header("Answer")
            with st.container(border=True):
                st.write(syn.get("answer", ""))
                
            if syn.get("grounded"):
                st.success("Grounded in retrieved evidence")
            else:
                st.error("Insufficient indexed evidence")
                
            if syn.get("fallback_used"):
                st.caption("Deterministic evidence summary used.")
                
            st.header("Key Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Seed Entities", len(ret.get("seed_entities", [])))
            m_col2.metric("Graph Relations Traversed", len(ret.get("relations", [])))
            m_col3.metric("Paths Found", len(ret.get("paths", [])))
            m_col4.metric("Evidence Chunks", len(ret.get("evidence", [])))
            
            m2_col1, m2_col2, m2_col3, m2_col4 = st.columns(4)
            m2_col1.metric("Source Documents", syn.get("metrics", {}).get("distinct_source_document_count", 0))
            max_h = max([p.get("hop_count", 0) for p in ret.get("paths", [])], default=0)
            m2_col2.metric("Maximum Hop Reached", max_h)
            m2_col3.metric("Response Time (s)", f"{res.get('time', 0):.2f}")
            
            if ret.get("seed_entities"):
                st.header("Matched Entities")
                for seed in ret.get("seed_entities", []):
                    st.markdown(f"**{seed.get('canonical_name')}**\n\n{seed.get('method').capitalize()} match, score {seed.get('score')}")
                    
            st.header("Evidence Paths")
            paths = ret.get("paths", [])[:10]
            for p in paths:
                with st.container(border=True):
                    st.text(format_path(p, ret.get("relations", []) + ret.get("conflicts", [])))
                    st.caption(f"Hop count: {p.get('hop_count')}")
                    
            st.header("Citations")
            for c in syn.get("citations", []):
                st.markdown(f"- {format_citation(c)} (Reliability: {c.get('source_reliability', 'unknown')})")
                
            st.header("Retrieved Source Excerpts")
            for ev in ret.get("evidence", []):
                doc_id = ev.get("doc_id", "Unknown")
                page = ev.get("page_number", "N/A")
                rel = ev.get("source_reliability", "unknown")
                with st.expander(f"{doc_id} | Page {page} | {rel.capitalize()}"):
                    st.write(f"**Chunk ID:** {ev.get('chunk_id')}")
                    if ev.get("source_path"):
                        st.write(f"**Source Path:** {ev.get('source_path')}")
                    if ev.get("extraction_method"):
                        st.write(f"**Extraction Method:** {ev.get('extraction_method')}")
                    st.markdown("---")
                    st.text(truncate_excerpt(ev.get("chunk_text", "")))
                    
            conflicts = ret.get("conflicts", [])
            if conflicts:
                st.header("Conflicting or Disputed Evidence")
                for c in conflicts:
                    st.warning(f"**Factual Relation:** {c.get('source_entity')} --{format_relation_label(c.get('relation_type', ''))}--> {c.get('target_entity')}")
                    st.write(f"- Document: {c.get('source_doc_id')}")
                    st.write(f"- Reliability: {c.get('source_reliability')}")
                    if c.get('justification'):
                        st.write(f"- Justification: {c.get('justification')}")
                        
            with st.expander("Operational Retrieval Trace", expanded=False):
                st.markdown("**Auditable retrieval operations**")
                for t in syn.get("trace", []):
                    st.text(t)
                    
            all_warnings = ret.get("warnings", []) + syn.get("warnings", [])
            if all_warnings:
                with st.expander("Warnings"):
                    for w in all_warnings:
                        st.warning(w)
