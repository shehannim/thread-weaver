import os
import json
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def get_doc_metadata(filepath):
    # Heuristics based on folder name or file name
    # The corpus contains novels, wiki articles, codex data books, ephemera docs.
    # Assuming directory structure or filenames like novels/..., codex/...
    path_lower = filepath.lower()
    if "codex" in path_lower:
        return "codex", "high"
    elif "wiki" in path_lower:
        return "wiki", "medium"
    elif "novel" in path_lower:
        return "novel", "medium"
    elif "ephemera" in path_lower:
        return "ephemera", "low"
    else:
        return "unknown", "low"

def chunk_text(text, doc_id, doc_type, reliability, start_page):
    # Semantic chunking using RecursiveCharacterTextSplitter
    # It prefers paragraph \n\n, then sentence \n, then spaces.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_reliability": reliability,
            "page_number": start_page,
            "chunk_text": chunk.strip(),
            "chunk_id": f"{doc_id}_{start_page}_{i}"
        })
    return records

def process_pdf(filepath, doc_type, doc_id, reliability):
    records = []
    # Route based on doc_type (User Choice A1)
    if doc_type == "codex":
        # Use pdfplumber for data books where table structure matters
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                
                # Check for near-zero text (image page)
                if len(text.strip()) < 50:
                    # Need to OCR this page
                    im = page.to_image(resolution=300)
                    text = pytesseract.image_to_string(im.original)
                
                if text.strip():
                    records.extend(chunk_text(text, doc_id, doc_type, reliability, i + 1))
    else:
        # Use PyMuPDF for text-heavy docs
        doc = fitz.open(filepath)
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text()
            
            if len(text.strip()) < 50:
                # Near-zero text -> treat as scan and OCR
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            
            if text.strip():
                records.extend(chunk_text(text, doc_id, doc_type, reliability, i + 1))
        doc.close()
    return records

def process_docx(filepath, doc_id, doc_type, reliability):
    doc = docx.Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])
    if text.strip():
        return chunk_text(text, doc_id, doc_type, reliability, 1)
    return []

def process_text(filepath, doc_id, doc_type, reliability):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    if text.strip():
        return chunk_text(text, doc_id, doc_type, reliability, 1)
    return []

def process_image(filepath, doc_id, doc_type, reliability):
    img = Image.open(filepath)
    text = pytesseract.image_to_string(img)
    if text.strip():
        return chunk_text(text, doc_id, doc_type, reliability, 1)
    return []

def run_ingestion(raw_dir, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for root, _, files in os.walk(raw_dir):
            for file in files:
                filepath = os.path.join(root, file)
                doc_id = os.path.splitext(file)[0]
                doc_type, reliability = get_doc_metadata(filepath)
                
                ext = os.path.splitext(file)[1].lower()
                records = []
                try:
                    if ext == '.pdf':
                        records = process_pdf(filepath, doc_type, doc_id, reliability)
                    elif ext == '.docx':
                        records = process_docx(filepath, doc_id, doc_type, reliability)
                    elif ext in ['.md', '.txt']:
                        records = process_text(filepath, doc_id, doc_type, reliability)
                    elif ext in ['.jpg', '.jpeg', '.png']:
                        records = process_image(filepath, doc_id, doc_type, reliability)
                    else:
                        print(f"Skipping unsupported file: {filepath}")
                        continue
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                
                for record in records:
                    f_out.write(json.dumps(record) + '\n')
                
                print(f"Processed {file} ({len(records)} chunks)")

if __name__ == "__main__":
    # Expect corpus to be at CORPUS_PATH env var, or fallback to data/raw
    # In .env, we assume the user sets CORPUS_PATH to their local directory
    # containing the 'Ashen Era Archive'.
    CORPUS_PATH = os.getenv("CORPUS_PATH", os.path.join(os.path.dirname(__file__), "../data/raw"))
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/processed/chunks.jsonl")
    
    print(f"Starting ingestion from {CORPUS_PATH}...")
    run_ingestion(CORPUS_PATH, OUTPUT_FILE)
    print(f"Ingestion complete. Output saved to {OUTPUT_FILE}")
