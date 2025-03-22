import streamlit as st
import sys
import os
import traceback
import io
import re

# Make sure the current directory is in the path for importing main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main
from main import process_documents, retrieve_chunks, generate_answer, process_google_sheet

# Streamlit UI
st.title("PDF & Excel RAG App")
st.write("Upload PDFs and Excel files or link Google Sheets and ask questions based on their content.")

# Add a comprehensive reset button at the top
if st.button("🔄 Reset Everything"):
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Rerun the app to clear UI elements
    st.rerun()  # Changed from experimental_rerun() to rerun()

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

# File upload section
st.write("### Upload Files or Enter Google Sheet URL")
st.write("Supported formats: PDF (.pdf), Excel (.xlsx, .xls), Google Sheets (URL)")

# Initialize session state for Google Sheet
if 'gsheet_url' not in st.session_state:
    st.session_state.gsheet_url = ""
if 'gsheet_processed' not in st.session_state:
    st.session_state.gsheet_processed = False

# Google Sheets URL input
gsheet_url = st.text_input(
    "Enter Google Sheets URL (must be publicly accessible or shared with view access)",
    value=st.session_state.gsheet_url
)

# Validate Google Sheets URL if provided
valid_gsheet = False
if gsheet_url:
    # Simple validation for Google Sheets URL format
    gsheet_pattern = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(/.*)?'
    if re.match(gsheet_pattern, gsheet_url):
        valid_gsheet = True
        st.session_state.gsheet_url = gsheet_url
    else:
        st.error("Invalid Google Sheets URL. Please enter a valid URL.")

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

# Process uploaded files or Google Sheet
if (uploaded_files and not st.session_state.processed) or (valid_gsheet and not st.session_state.gsheet_processed):
    with st.spinner("Processing files and data sources..."):
        try:
            num_chunks = 0
            
            # Process uploaded files if any
            if uploaded_files:
                num_chunks += process_documents(uploaded_files)
                
            # Process Google Sheet if valid URL provided
            if valid_gsheet:
                try:
                    sheet_chunks = process_google_sheet(gsheet_url)
                    num_chunks += sheet_chunks
                    st.session_state.gsheet_processed = True
                except Exception as e:
                    st.error(f"Error processing Google Sheet: {str(e)}")
                    st.expander("See detailed error").write(traceback.format_exc())
            
            if num_chunks > 0:
                st.session_state.processed = True
                st.success(f"Processed {len(uploaded_files) if uploaded_files else 0} files and " +
                          f"{1 if valid_gsheet and st.session_state.gsheet_processed else 0} Google Sheets " +
                          f"into {num_chunks} chunks.")
                
                # Display information about the chunks
                st.subheader("Generated Document Chunks")
                st.write(f"Total chunks created: {len(main.chunks_with_metadata)}")
                
                # Group chunks by source
                sources = {}
                for chunk in main.chunks_with_metadata:
                    source = chunk["source"]
                    if source not in sources:
                        sources[source] = []
                    sources[source].append(chunk)
                
                # Display chunks grouped by source
                for source, chunks in sources.items():
                    with st.expander(f"{source} - {len(chunks)} chunks"):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {i}** - Page: {chunk['page']}")
                            # Display a preview of the text (first 100 characters)
                            preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                            st.text(preview)
                            
                            # Show full text in a nested expander
                            with st.expander("Show full content"):
                                st.markdown(chunk['text'])
            else:
                st.error("No text could be extracted from the provided sources.")
                st.session_state.error = "Failed to extract text from sources."
                st.session_state.processed = False  # Add this line to ensure processed state is correctly set
                st.session_state.gsheet_processed = False  # Reset this too if there's an error
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error processing sources: {error_msg}")
            st.session_state.error = error_msg
            st.session_state.processed = False  # Ensure processed state is reset on error
            st.session_state.gsheet_processed = False  # Reset this too if there's an error

# If there was an error, provide a reset button
if st.session_state.error:
    if st.button("Reset Processing"):
        st.session_state.processed = False
        st.session_state.gsheet_processed = False
        st.session_state.error = None
        st.rerun()  # Changed from experimental_rerun() to rerun()

# Query input and processing - add safety checks
if st.session_state.get('processed', False) or st.session_state.get('gsheet_processed', False):
    st.write("### Ask Questions About Your Documents")
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                try:
                    # Add a verification check before retrieving chunks
                    if not (st.session_state.get('processed', False) or st.session_state.get('gsheet_processed', False)):
                        st.error("Please process documents first before asking questions.")
                    else:
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
                            
                            # Display chunks used for the answer
                            st.subheader("Chunks Used for Answer")
                            with st.expander("Show parsed chunks"):
                                for i, chunk in enumerate(retrieved_chunks, 1):
                                    st.markdown(f"#### Chunk {i} - {chunk['source']} (Page {chunk['page']})")
                                    
                                    # Display the chunk content in a formatted way
                                    st.markdown("```")
                                    st.text(chunk['text'])
                                    st.markdown("```")
                                    
                                    # Add metadata as a table if it has interesting structure
                                    if chunk['page'] and 'Sheet' in str(chunk['page']):
                                        if 'Summary' in str(chunk['page']):
                                            st.info("This is a summary chunk containing column information and statistics.")
                                        elif 'Rows' in str(chunk['page']):
                                            rows_info = str(chunk['page']).split('Rows ')[1].strip(')')
                                            st.info(f"This chunk contains data rows {rows_info}.")
                                    
                                    st.markdown("---")
                        except ValueError as ve:
                            st.error(str(ve))
                        except Exception as e:
                            st.error(f"Error retrieving or processing data: {str(e)}")
                            st.session_state.error = str(e)
                            st.expander("See detailed error").write(traceback.format_exc())
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.expander("See detailed error").write(traceback.format_exc())
        else:
            st.warning("Please enter a question.")
elif not st.session_state.error:
    st.info("Please upload at least one file or provide a Google Sheet URL to start.")
else:
    st.error("There was an error processing your documents. Please reset and try again.")
    if st.button("Reset and Try Again"):
        st.session_state.processed = False
        st.session_state.gsheet_processed = False
        st.session_state.error = None
        st.rerun()