
# *****************************************************************************
# But: Mise en place d'un RAG EMSI-ChatBot
# Prof. OUHMIDA Asmae
# T.A.F:
# 1/ Démarrer le programme
# 2/ Analyser le code
# 3/ Améliorer le programme
#*******************************************************************************

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain_classic.chains import RetrievalQA



# Set Streamlit page configuration
# st.set_page_config(page_title="PDF-RAG", page_icon="📄")
st.set_page_config(page_title="PDF-RAG", page_icon="👀")

# Sidebar for PDF upload
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Main area for Q&A
st.title("Local PDF-RAG with LangChain, Ollama, and Chroma")

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Handle PDF upload and processing
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split the PDF
        loader = PyPDFLoader(uploaded_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)

        # Create embeddings and store in Chroma
        embeddings = HuggingFaceEmbeddings()
        st.session_state.db = Chroma.from_documents(texts, embeddings)

        # Initialize the QA chain with Ollama
        llm = Ollama(model="llama3.2")  # Or your preferred Ollama model
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.db.as_retriever(),
            return_source_documents=True,
        )

        st.success("PDF processed and ready for questions!")

# User input for query
query = st.chat_input("Ask a question about the PDF:")

# Generate and display response
if query and st.session_state.qa_chain:
    with st.spinner("Generating response..."):
        result = st.session_state.qa_chain(query)
        st.write("**Answer:**", result["result"])
        st.write("**Sources:**")
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata['source']} (page {doc.metadata['page']})")