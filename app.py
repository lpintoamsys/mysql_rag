import streamlit as st
from rag_mysql import RAGMySQL

st.title("RAG System for MySQL")
rag_agent = RAGMySQL()

question = st.text_input("Ask a question about the database:")

if st.button("Submit"):
    if question:
        response = rag_agent.query_rag(question)
        st.write("### Answer:")
        st.write(response)
    else:
        st.warning("Please enter a question.")