# src/rag_chain.py
from transformers import pipeline
from utils.config import LLM_MODEL

def build_rag_chain(db):
    """
    Create RAG components with local LLM and Chroma retriever.
    No API needed - runs completely locally.
    """
    # Load local text generation model
    llm = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2
    )
    
    # Get retriever from database
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    return (llm, retriever)

def ask_question(components, query):
    """
    Run RAG pipeline to get answer and context.
    Completely local - no API calls.
    """
    llm, retriever = components
    
    # Get relevant documents using invoke (updated method)
    sources = retriever.invoke(query)
    
    # Format context from sources
    context = "\n\n".join([doc.page_content for doc in sources])
    
    # Create prompt with context and question
    prompt = f"""Answer the question based only on the following context:

Context: {context}

Question: {query}

Answer:"""
    
    # Get answer from local LLM
    result = llm(prompt, max_length=512, truncation=True)
    answer = result[0]['generated_text']
    
    return answer, sources