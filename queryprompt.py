from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import streamlit as st
import asyncio
import time

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize embeddings, model, and vector store
@st.cache_resource  # Singleton, prevent multiple initializations
def init_chain():
    model_kwargs = {'trust_remote_code': True}
    embedding = HuggingFaceEmbeddings(model_name='nomic-ai/nomic-embed-text-v1.5', model_kwargs=model_kwargs)
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.2)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    # Create chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(k=5),
        return_source_documents=True
    )
    return chain

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.set_page_config(page_title="Chat with your Documents", layout="wide")
    st.title("Chat with your Documents")

    # Sidebar for user inputs and settings
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of Documents to Retrieve", 1, 10, 5)

    query_text = st.chat_input("Ask anything")
    if query_text:
        with st.spinner("Processing..."):
            response_text = asyncio.run(query_rag(query_text, top_k))
            st.session_state.chat_history.append({"user": query_text, "assistant": response_text})

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["assistant"])

async def query_rag(query_text: str, top_k: int):
    start_time = time.time()

    # Initialize the chain
    chain = init_chain()
    
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
