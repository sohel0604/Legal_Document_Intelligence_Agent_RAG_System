# streamlit_app.py
import streamlit as st
import os
from src.data_loader import load_pdf
from src.text_splitter import split_docs
from src.embed_store import create_vector_db
from src.rag_chain import build_rag_chain, ask_question

st.set_page_config(page_title="⚖️ Legal Document Intelligence Agent", layout="wide")

st.title("⚖️ Legal Document Intelligence Agent (RAG System)")
st.markdown("Upload a **Legal PDF** (e.g., contract, privacy policy) and ask your legal questions.")
st.info("🔒 **100% Private** - runs completely on my machine!")

# Initialize session state
if 'qa' not in st.session_state:
    st.session_state.qa = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader("📄 Upload a legal document (PDF)", type="pdf")

if uploaded_file and not st.session_state.processed:
    # Save uploaded file
    file_path = "uploaded_doc.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded successfully!")

    try:
        with st.spinner("🔍 Processing document... (This may take a moment on first run)"):
            # Load and process document
            docs = load_pdf(file_path)
            chunks = split_docs(docs)
            db = create_vector_db(chunks)
            st.session_state.qa = build_rag_chain(db)
            st.session_state.processed = True
        st.success("✅ Document ready for Q&A!")
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        st.session_state.processed = False

# Q&A section - only show if document is processed
if st.session_state.processed and st.session_state.qa:
    question = st.text_input("💬 Ask a question about the document:")

    if question:
        try:
            with st.spinner("🧠 Analyzing..."):
                answer, sources = ask_question(st.session_state.qa, question)

            st.markdown("### 🧠 Answer:")
            st.write(answer)

            st.markdown("### 📜 Relevant Text Snippets:")
            for i, src in enumerate(sources, 1):
                with st.expander(f"📄 Snippet {i}"):
                    st.markdown(src.page_content[:400] + "...")
        except Exception as e:
            st.error(f"❌ Error getting answer: {str(e)}")

# Add a reset button
if st.session_state.processed:
    if st.button("🔄 Upload New Document"):
        st.session_state.qa = None
        st.session_state.processed = False
        if os.path.exists("uploaded_doc.pdf"):
            os.remove("uploaded_doc.pdf")
        st.rerun()