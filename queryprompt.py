import fitz  # PyMuPDF
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import get_embedding_function
import streamlit as st
import asyncio
import time
import os
import shutil

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Function to extract text from an uploaded PDF using a temporary file
def extract_text_from_uploaded_pdf(uploaded_file, progress_bar):
    # Save the uploaded file to a temporary file
    temp_file_path = f"/tmp/{uploaded_file.name}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Use PyPDFLoader to load documents from the temporary file
    document_loader = PyPDFLoader(temp_file_path)
    documents = document_loader.load()

    # Clean up the temporary file
    os.remove(temp_file_path)

    # Update progress bar
    progress_bar.progress(1.0)

    return documents

# Initialize embeddings, model, and vector store
@st.cache_resource  # Singleton, prevent multiple initializations
def init_chain(top_k: int):
    model_kwargs = {'trust_remote_code': True}
    embedding = HuggingFaceEmbeddings(model_name='nomic-ai/nomic-embed-text-v1.5', model_kwargs=model_kwargs)
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.2)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    # Create chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(k=top_k),
        return_source_documents=True
    )

    return chain

# Function to clear data from the database
def clear_database():
    # Initialize the vector store
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    
    try:
        # Delete all documents from the vector store
        vectordb.clear()
        st.sidebar.success("Database cleared")
    except Exception as e:
        st.sidebar.error(f"Error clearing database: {e}")

# Initialize chat history and processing flags
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'file_processing' not in st.session_state:
    st.session_state.file_processing = False
if 'query_processing' not in st.session_state:
    st.session_state.query_processing = False
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

def main():
    st.set_page_config(page_title="Chat with your Documents", layout="wide")
    st.title("Chat with your Documents")

    # Create placeholders for the spinner and chat input
    spinner_placeholder = st.empty()
    chat_placeholder = st.empty()

    # Sidebar for user inputs and settings
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of Documents to Retrieve", 1, 10, 5)

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload a PDF or document file", type=["pdf"])
    if uploaded_file is not None:
        if not st.session_state.file_uploaded:
            # Set file processing flag
            st.session_state.file_processing = True
            st.session_state.file_uploaded = True

            # Show spinner while processing the uploaded file
            with spinner_placeholder:
                with st.spinner("Processing the uploaded file..."):
                    progress_bar = st.progress(0)

                    # Extract text from the uploaded file
                    documents = extract_text_from_uploaded_pdf(uploaded_file, progress_bar)

                    # Add extracted text to the vector store
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
                    texts = text_splitter.split_documents(documents)
                    text_contents = [doc.page_content for doc in texts]  # Extract text content from Document objects
                    model_kwargs = {'trust_remote_code': True}
                    embedding_function = HuggingFaceEmbeddings(model_name='nomic-ai/nomic-embed-text-v1.5', model_kwargs=model_kwargs)
                    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
                    vectordb.add_texts(text_contents)  # Pass text content instead of Document objects
                    st.sidebar.success("Text added to vector store")

                    # Reset file processing flag
                    st.session_state.file_processing = False
                    spinner_placeholder.empty()

     # Button to clear the database
    if st.sidebar.button("Clear Database"):
        clear_database()

    query_text = st.chat_input("Ask anything")
    if query_text:
        # Set query processing flag
        st.session_state.query_processing = True

        # Show spinner while processing the query
        with spinner_placeholder:
            with st.spinner("Processing your query..."):
                response_text = asyncio.run(query_rag(query_text, top_k))
                st.session_state.chat_history.append({"user": query_text, "assistant": response_text})

        # Reset query processing flag
        st.session_state.query_processing = False
        spinner_placeholder.empty()

        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.markdown(chat["assistant"])

async def query_rag(query_text: str, top_k: int):
    start_time = time.time()

    # Initialize the chain
    chain = init_chain(top_k)
    
    # Perform the retrieval-based QA
    response = chain(query_text)
    
    response_text = response['result']
    sources = response['source_documents']

    # Prepare context
    context_start = time.time()
    context_text = "\n\n---\n\n".join([doc.page_content for doc in sources])
    context_end = time.time()
    st.write(f"Context Preparation Time: {context_end - context_start:.2f} seconds")

    total_time = time.time() - start_time
    st.write(f"Total Time: {total_time:.2f} seconds")
    
    return response_text

if __name__ == "__main__":
    main()
