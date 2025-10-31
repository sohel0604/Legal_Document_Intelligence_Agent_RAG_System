# src/data_loader.py
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """
    Load and extract text from PDF using LangChain PyPDFLoader.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents