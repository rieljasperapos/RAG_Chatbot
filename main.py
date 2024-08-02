import streamlit as st
import queryprompt

def main():
  st.title("Chat with your Documents")
  st.text("TESTING")
  query_text = st.text_input("Ask anything")
  # queryprompt.query_rag(query_text)