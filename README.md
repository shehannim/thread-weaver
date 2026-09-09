# Thread Weaver - Knowledge Graph Multi-Hop QA System

A knowledge-graph-based multi-hop QA system designed for the "Connecting Facts Across Thousands of Pages" hackathon challenge. The system weaves threads of evidence across a mixed-format document corpus (novels, wiki articles, codex data books, and ephemera) into an Obsidian-style knowledge graph to connect facts that plain vector RAG cannot easily retrieve.

## Project Structure

- `/ingest` - Document parsing, OCR fallback, and semantic chunking.
- `/graph` - Entity/relation extraction and graph building (to be implemented).
- `/retrieve` - Graph traversal and retrieval logic (to be implemented).
- `/synth` - Answer synthesis logic (to be implemented).
- `/ui` - Streamlit application (to be implemented).
- `/utils` - External API wrappers with exponential backoff (OpenRouter and Voyage AI).
- `/data/raw` - The original "Ashen Era Archive" corpus (read-only).
- `/data/vault` - Output markdown notes with typed links.
- `/data/processed` - Intermediate artifacts (e.g., `chunks.jsonl`).
- `/docs` - Architecture and decision documents.

## Setup

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **System Dependencies**:
   - Install **Tesseract OCR** for Windows (required for scanning zero-text PDFs and images).
   - Ensure the installation path matches `TESSERACT_CMD` in your `.env` (default is `C:\Program Files\Tesseract-OCR\tesseract.exe`).

3. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in the required keys:
   - Set `OPENROUTER_API_KEY` for LLM calls.
   - Set `VOYAGE_API_KEY` for `voyage-4-large` embeddings.
   - Set `CORPUS_PATH` to the absolute path of your "Ashen Era Archive" folder.

## Running Ingestion

The ingestion module parses PDF, DOCX, MD, TXT, and scanned-image files. It routes PDFs based on their document type (using `pdfplumber` for structured codex books and `PyMuPDF` for faster text extraction on novels/wikis). It also features fallback OCR using `pytesseract` for zero-text PDF pages. Semantic chunking is handled by LangChain's `RecursiveCharacterTextSplitter`.

To run the document parsing and chunking:

```bash
cd /path/to/ashen_era_qa
python ingest/ingest.py
```

This will read all files from your `CORPUS_PATH`, process them, and output normalized JSON records to `data/processed/chunks.jsonl`.
