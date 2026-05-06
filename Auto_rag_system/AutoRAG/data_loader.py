# =======================
# FILE: data_loader.py
# =======================
import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    # Prefer maintained package if present
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json
from pathlib import Path
import argparse
def _load_arxiv_json(json_path: str, max_records: int | None = None) -> list[Document]:
    docs = []
    p = Path(json_path)
    if not p.exists():
        return docs
    try:
        count = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = (obj.get("title") or "").strip()
                abstract = (obj.get("abstract") or "").strip()
                if not title and not abstract:
                    continue
                content = f"Title: {title}\nAbstract: {abstract}".strip()
                metadata = {
                    "source": str(p),
                    "id": obj.get("id"),
                    "categories": obj.get("categories"),
                    "submitter": obj.get("submitter"),
                    "doi": obj.get("doi"),
                }
                docs.append(Document(page_content=content, metadata=metadata))
                count += 1
                if max_records is not None and count >= max_records:
                    break
    except Exception as e:
        print(f"[WARNING] Failed reading JSON lines from {p}: {e}")
    return docs

def load_and_index_data(
    data_path: str = "data/pdfs/",
    save_path: str = "embeddings/faiss_index",
    include_arxiv_json: bool = True,
    arxiv_json_path: str = "archive (3)/arxiv-metadata-oai-snapshot.json",
    arxiv_max_records: int | None = None,
):
    """
    Load PDF documents, split them into chunks, generate embeddings, and create a FAISS index.
    Args:
        data_path (str): Path to directory containing PDF files
        save_path (str): Path to save the FAISS index
    Returns:
        str: Path where the index was saved
    """
    # Create directories if they don't exist
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[INFO] Loading documents from {data_path}")
    try:
        # Load PDFs
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=UnstructuredPDFLoader)
        documents = loader.load()
        pdf_doc_count = len(documents)
        print(f"[INFO] Loaded {pdf_doc_count} PDF documents")
        json_docs = []
        if include_arxiv_json:
            json_docs = _load_arxiv_json(arxiv_json_path, max_records=arxiv_max_records)
            print(f"[INFO] Loaded {len(json_docs)} JSON records from '{arxiv_json_path}'")
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        texts = splitter.split_documents(documents)
        if json_docs:
            texts.extend(json_docs)
        print(f"[INFO] Prepared {len(texts)} chunks/records for embedding")
        if len(texts) == 0:
            print("[WARNING] No documents or JSON records found. Skipping index creation.")
            return None
        # Create embeddings
        print("[INFO] Generating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        # Store embeddings in FAISS
        print("[INFO] Creating FAISS index...")
        vectorstore = FAISS.from_documents(texts, embeddings)
        # Save the index
        vectorstore.save_local(save_path)
        print(f"[SUCCESS] Indexed {len(texts)} chunks. Saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"[ERROR] Failed to process documents: {str(e)}")
        return None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs and optional arXiv JSONL")
    parser.add_argument("--data-path", default="data/pdfs/", help="Path to PDFs directory")
    parser.add_argument("--save-path", default="embeddings/faiss_index", help="Path to save FAISS index")
    parser.add_argument("--include-arxiv", action="store_true", help="Include arXiv JSONL ingestion")
    parser.add_argument("--arxiv-json", default="archive (3)/arxiv-metadata-oai-snapshot.json", help="Path to arXiv JSONL file")
    parser.add_argument("--arxiv-max", type=int, default=None, help="Max number of JSON records to load (None for all)")
    args = parser.parse_args()
    load_and_index_data(
        data_path=args.data_path,
        save_path=args.save_path,
        include_arxiv_json=args.include_arxiv,
        arxiv_json_path=args.arxiv_json,
        arxiv_max_records=args.arxiv_max,
    )
