# src/embed_store.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.config import EMBEDDING_MODEL, PERSIST_DIR

def create_vector_db(chunks):
    """
    Create ChromaDB vector store with embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return db