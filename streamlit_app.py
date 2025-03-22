import streamlit as st
from main import process_documents, retrieve_chunks, generate_answer
import os

# Streamlit UI
st.title("PDF RAG App")
st.write("Upload PDFs and ask questions based on their content.")

# Configure OpenAI API key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif "OPENAI_API_KEY" in os.environ:
    pass
else:
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Session state to track processed files
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Process uploaded files
if uploaded_files and not st.session_state.processed:
    with st.spinner("Processing PDFs..."):
        num_chunks = process_documents(uploaded_files)
        st.session_state.processed = True
        st.success(f"Processed {len(uploaded_files)} PDFs into {num_chunks} chunks.")

# Query input and processing
if st.session_state.get('processed', False):
    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                retrieved_chunks = retrieve_chunks(query)
                answer, sources = generate_answer(query, retrieved_chunks)
                
                # Display answer
                st.subheader("Answer")
                st.write(answer)
                
                # Display sources
                st.subheader("Sources")
                for source, page in set(sources):  # Remove duplicates
                    st.write(f"- {source} (Page {page})")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload at least one PDF to start.")