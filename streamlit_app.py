import streamlit as st
from main import process_documents, retrieve_chunks, generate_answer
import os
import traceback
import io

# Streamlit UI
st.title("PDF & Excel RAG App")
st.write("Upload PDFs and Excel files and ask questions based on their content.")

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

# File uploader - simplified approach
st.write("### Upload Files")
st.write("Supported formats: PDF (.pdf), Excel (.xlsx, .xls)")

# Create separate uploaders for each file type to avoid MIME type issues
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
excel_files = st.file_uploader("Upload Excel files", type=["xlsx", "xls"], accept_multiple_files=True)

# Combine the files
uploaded_files = pdf_files + excel_files if excel_files else pdf_files

# Debug information
if uploaded_files:
    st.write("### Uploaded Files")
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1].lower()
        st.write(f"- {file.name} (Type: {file_extension})")

# Session state to track processed files
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'error' not in st.session_state:
    st.session_state.error = None

# Process uploaded files
if uploaded_files and not st.session_state.processed:
    with st.spinner("Processing files..."):
        try:
            num_chunks = process_documents(uploaded_files)
            if num_chunks > 0:
                st.session_state.processed = True
                st.success(f"Processed {len(uploaded_files)} files into {num_chunks} chunks.")
            else:
                st.error("No text could be extracted from the uploaded files.")
                st.session_state.error = "Failed to extract text from files."
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error processing files: {error_msg}")
            st.session_state.error = error_msg
            st.expander("See detailed error").write(traceback.format_exc())

# If there was an error, provide a reset button
if st.session_state.error:
    if st.button("Reset"):
        st.session_state.processed = False
        st.session_state.error = None
        st.experimental_rerun()

# Query input and processing
if st.session_state.get('processed', False):
    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                try:
                    retrieved_chunks = retrieve_chunks(query)
                    answer, sources = generate_answer(query, retrieved_chunks)
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(answer)
                    
                    # Display sources
                    st.subheader("Sources")
                    for source, page in set(sources):  # Remove duplicates
                        st.write(f"- {source} (Page {page})")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.expander("See detailed error").write(traceback.format_exc())
        else:
            st.warning("Please enter a question.")
elif not st.session_state.error:
    st.info("Please upload at least one PDF to start.")