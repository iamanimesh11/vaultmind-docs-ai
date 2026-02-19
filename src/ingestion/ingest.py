"""
ingest.py
---------
Handles loading PDF documents, splitting them into chunks,
generating embeddings, and storing them in a FAISS vector database.
"""

import argparse
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
VECTOR_STORE_PATH = Path(__file__).parent.parent.parent / "vector_store"


def load_and_split(pdf_path: str) -> list:
    """Load a PDF and split it into overlapping text chunks."""
    logger.info(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"✅ Split into {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks: list) -> None:
    """Generate embeddings and save the FAISS vector store to disk."""
    logger.info("🔢 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info("💾 Building and saving FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    db.save_local(str(VECTOR_STORE_PATH))
    logger.info(f"✅ Vector store saved to: {VECTOR_STORE_PATH}")


def process_document(pdf_path: str) -> None:
    """End-to-end pipeline: load PDF → chunk → embed → store."""
    chunks = load_and_split(pdf_path)
    build_vector_store(chunks)
    logger.info("🎉 Document ingestion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into the FAISS vector store.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to ingest.")
    args = parser.parse_args()
    process_document(args.pdf_path)
