# utils/config.py
import os

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"  # Small local model, no API needed

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIR = "vector_db"