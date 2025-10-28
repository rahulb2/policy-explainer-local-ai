import streamlit as st
from PyPDF2 import PdfReader
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# ----------------------
# Streamlit Page Setup
# ----------------------
st.set_page_config(page_title="Policy Explainer", page_icon="📜")
st.title("📜 Policy Explainer (Local LLM Edition)")

st.write("Upload legal or policy documents and get **plain language explanations** of complex text — all processed locally using Ollama and HuggingFace.")

# Upload Document
uploaded_file = st.file_uploader("Upload a legal or policy document (PDF)", type=["pdf"])

# ----------------------
# Function: Extract text from PDF
# ----------------------
def extract_text_from_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    reader = PdfReader(tmp_file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ----------------------
# Main Logic
# ----------------------
if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)

    if raw_text.strip() == "":
        st.warning("⚠️ Could not extract text from this PDF.")
    else:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)

        # Create local embeddings
        st.write("🔹 Generating embeddings locally with HuggingFace...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Use Ollama as the local LLM
        st.write("🤖 Using Ollama as local LLM backend...")
        llm = Ollama(model="llama3")

        # Build RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        st.subheader("💬 Ask questions about the policy/legal text")
        query = st.text_input("E.g. 'Explain section 5 in simple terms' or 'What does this policy require me to do?' ")

        if query:
            with st.spinner("Analyzing and simplifying..."):
                response = qa_chain.run(query)
            st.success("✅ Explanation:")
            st.write(response)

        # Optional: show raw text
        with st.expander("📄 Show Extracted Text"):
            st.text_area("Document Text", raw_text, height=300)
else:
    st.info("👆 Please upload a PDF to begin.")

st.markdown("---")
st.caption("⚡ Runs locally with HuggingFace embeddings + Ollama LLM. No API key required.")
