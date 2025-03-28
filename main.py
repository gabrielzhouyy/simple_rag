import pdfplumber
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer
import re
import openai
import os
import io
import traceback
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pandas as pd
import datetime
import mimetypes
import requests
import json
import sys

# Ensure NLTK data is downloaded properly
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        print("Downloading NLTK punkt data...")
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")

# Initialize global variables
vector_db = None
chunks_with_metadata = []

# Disable torch warning messages that are causing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize tokenizer with simplified error handling
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
    try:
        # Fallback to a simpler tokenizer
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    except:
        print("Could not load any tokenizer, using simple text splitting")
        # Define a simple tokenizer function as fallback
        def simple_tokenize(text):
            return text.split()
        tokenizer = type('', (), {})()
        tokenizer.tokenize = simple_tokenize

# Initialize sentence transformer with simpler approach to avoid torch errors
try:
    # Create sentence transformer with specific device and no_cache to avoid errors
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    # Test the model with a simple input to verify it works
    _ = model.encode(["Test sentence"], convert_to_numpy=True)
    print("Successfully loaded SentenceTransformer model")
except Exception as e:
    print(f"Error loading sentence transformer model: {str(e)}")
    # Define a very simple fallback embedding function using numpy
    print("Using fallback embedding method")
    def simple_encode(sentences, convert_to_numpy=True):
        # Simple hashing-based embedding (not good for semantic search but works as fallback)
        embeddings = []
        for sentence in sentences:
            # Create a simple embedding based on character hashing
            chars = list(sentence.lower())
            hash_values = [hash(char) % 256 for char in chars]
            # Pad or truncate to fixed dimension
            embedding_size = 384  # Standard size for mini models
            if len(hash_values) >= embedding_size:
                embedding = np.array(hash_values[:embedding_size], dtype=np.float32)
            else:
                embedding = np.zeros(embedding_size, dtype=np.float32)
                embedding[:len(hash_values)] = hash_values
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding)
        return np.array(embeddings)

    # Replace the model's encode method with our simple implementation
    model = type('', (), {})()
    model.encode = simple_encode

def process_documents(files):
    """
    Process uploaded files (PDFs and Excel): parse, chunk, embed, and store in FAISS.
    """
    global vector_db, chunks_with_metadata

    # Reset previous data
    chunks_with_metadata = []
    all_chunks = []

    # Process each file based on type
    for file in files:
        try:
            # Check file type by extension (more reliable than MIME type)
            file_extension = os.path.splitext(file.name)[1].lower()
            
            print(f"Processing file: {file.name} with extension {file_extension}")
            
            if file_extension == '.pdf':
                process_pdf_file(file, all_chunks, chunks_with_metadata)
            elif file_extension in ['.xlsx', '.xls']:
                process_excel_file(file, all_chunks, chunks_with_metadata)
            else:
                print(f"Unsupported file format: {file_extension}")
        except Exception as e:
            print(f"Error processing file {file.name}: {str(e)}")
            traceback.print_exc()
    
    # If no chunks were extracted, return 0
    if not all_chunks:
        return 0
        
    try:
        # Embed chunks using all-MiniLM-L6-v2
        embeddings = model.encode(all_chunks, convert_to_numpy=True)

        # Initialize FAISS index with cosine similarity
        dimension = embeddings.shape[1]
        vector_db = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        vector_db.add(embeddings)
    except Exception as e:
        print(f"Error creating vector database: {str(e)}")
        raise

    return len(all_chunks)

def process_pdf_file(pdf_file, all_chunks, chunks_with_metadata):
    """Process a single PDF file and extract chunks"""
    try:
        # Create a copy of the file in memory to avoid issues with file pointers
        pdf_content = io.BytesIO(pdf_file.read())
        pdf_file.seek(0)  # Reset pointer for potential future use
        
        with pdfplumber.open(pdf_content) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if text:
                        try:
                            # Fallback to simple splitting if NLTK fails
                            try:
                                sentences = nltk.sent_tokenize(text)
                            except LookupError:
                                # Simple fallback in case NLTK data is not available
                                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
                                
                            current_chunk = ""
                            for sentence in sentences:
                                if len(tokenizer.tokenize(current_chunk + " " + sentence)) < 500:
                                    current_chunk += " " + sentence
                                else:
                                    if current_chunk:
                                        all_chunks.append(current_chunk.strip())
                                        chunks_with_metadata.append({
                                            "text": current_chunk.strip(),
                                            "source": pdf_file.name,
                                            "page": page_num
                                        })
                                    current_chunk = sentence
                            if current_chunk:
                                all_chunks.append(current_chunk.strip())
                                chunks_with_metadata.append({
                                    "text": current_chunk.strip(),
                                    "source": pdf_file.name,
                                    "page": page_num
                                })
                        except Exception as e:
                            print(f"Error processing text on page {page_num}: {str(e)}")
                            # Use fixed-size chunking as fallback
                            chunks = fixed_size_chunking(text)
                            for chunk in chunks:
                                all_chunks.append(chunk)
                                chunks_with_metadata.append({
                                    "text": chunk,
                                    "source": pdf_file.name,
                                    "page": page_num
                                })
                except Exception as e:
                    print(f"Error extracting text from page {page_num}: {str(e)}")
    except Exception as e:
        print(f"Error processing PDF file {pdf_file.name}: {str(e)}")

def process_excel_file(excel_file, all_chunks, chunks_with_metadata):
    """Process an Excel file and extract chunks with robust error handling"""
    try:
        # Copy the file content to memory buffer
        excel_content = io.BytesIO(excel_file.read())
        excel_file.seek(0)  # Reset pointer
        
        # First try with pandas directly (simplest approach)
        try:
            # Try without specifying engine first
            df = pd.read_excel(excel_content)
            process_generic_excel_sheet(df, "Sheet1", all_chunks, chunks_with_metadata, excel_file.name)
            return
        except Exception as e:
            print(f"Standard Excel reading failed: {str(e)}")
            excel_content.seek(0)  # Reset pointer
        
        # Try each engine explicitly, with multiple approaches
        engines = ['openpyxl', 'xlrd']
        sheet_names = []
        
        for engine in engines:
            try:
                print(f"Trying Excel engine: {engine}")
                # Try to get the first sheet only
                df = pd.read_excel(excel_content, engine=engine)
                if not df.empty:
                    process_generic_excel_sheet(df, "Sheet1", all_chunks, chunks_with_metadata, excel_file.name)
                    return
            except Exception as e:
                print(f"Failed with engine {engine}: {str(e)}")
                excel_content.seek(0)  # Reset pointer
        
        # If we're here, all methods failed
        raise Exception(f"Could not read Excel file {excel_file.name} with any available method")
            
    except Exception as e:
        print(f"Error processing Excel file {excel_file.name}: {str(e)}")
        # Add this information as a text chunk so it's searchable
        error_text = f"Error processing file {excel_file.name}: {str(e)}"
        all_chunks.append(error_text)
        chunks_with_metadata.append({
            "text": error_text,
            "source": excel_file.name,
            "page": "Error"
        })

def process_travel_plan(excel_content, sheet_names, all_chunks, chunks_with_metadata, filename):
    """Extract information from a travel plan Excel file"""
    extracted_info = []
    
    # Travel-related fields to look for
    flight_fields = ['airline', 'flight', 'departure', 'arrival', 'origin', 'destination']
    hotel_fields = ['hotel', 'accommodation', 'check-in', 'check-out', 'stay']
    activity_fields = ['activity', 'event', 'tour', 'visit', 'sightseeing']
    
    for sheet_idx, sheet_name in enumerate(sheet_names, 1):
        try:
            df = pd.read_excel(excel_content, sheet_name=sheet_name)
            
            # Drop completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                continue
                
            # Convert headers to string and lowercase for matching
            headers = [str(col).lower() for col in df.columns]
            
            # Extract flight details
            flight_info = []
            flight_cols = [i for i, header in enumerate(headers) 
                           if any(field in header for field in flight_fields)]
            
            if flight_cols:
                for _, row in df.iterrows():
                    flight_data = {}
                    for col_idx in flight_cols:
                        col_name = df.columns[col_idx]
                        value = row[col_name]
                        if pd.notna(value):
                            flight_data[str(col_name)] = str(value)
                    
                    if flight_data:
                        flight_info.append(flight_data)
            
            # Extract hotel details similarly
            hotel_info = []
            hotel_cols = [i for i, header in enumerate(headers) 
                          if any(field in header for field in hotel_fields)]
            
            if hotel_cols:
                for _, row in df.iterrows():
                    hotel_data = {}
                    for col_idx in hotel_cols:
                        col_name = df.columns[col_idx]
                        value = row[col_name]
                        if pd.notna(value):
                            hotel_data[str(col_name)] = str(value)
                    
                    if hotel_data:
                        hotel_info.append(hotel_data)
            
            # Extract activity details similarly
            activity_info = []
            activity_cols = [i for i, header in enumerate(headers) 
                             if any(field in header for field in activity_fields)]
            
            if activity_cols:
                for _, row in df.iterrows():
                    activity_data = {}
                    for col_idx in activity_cols:
                        col_name = df.columns[col_idx]
                        value = row[col_name]
                        if pd.notna(value):
                            activity_data[str(col_name)] = str(value)
                    
                    if activity_data:
                        activity_info.append(activity_data)
            
            # Create a structured summary of the travel information
            summary = f"Travel Plan Information (Sheet: {sheet_name}):\n\n"
            
            if flight_info:
                summary += "Flight Details:\n"
                for i, flight in enumerate(flight_info, 1):
                    summary += f"  Flight {i}: {', '.join([f'{k}: {v}' for k, v in flight.items()])}\n"
                summary += "\n"
            
            if hotel_info:
                summary += "Hotel Details:\n"
                for i, hotel in enumerate(hotel_info, 1):
                    summary += f"  Hotel {i}: {', '.join([f'{k}: {v}' for k, v in hotel.items()])}\n"
                summary += "\n"
            
            if activity_info:
                summary += "Activity Details:\n"
                for i, activity in enumerate(activity_info, 1):
                    summary += f"  Activity {i}: {', '.join([f'{k}: {v}' for k, v in activity.items()])}\n"
            
            # Add the summary as a chunk
            if len(summary.strip()) > 10:  # Only add if there's meaningful content
                all_chunks.append(summary)
                chunks_with_metadata.append({
                    "text": summary,
                    "source": filename,
                    "page": f"Sheet {sheet_name}"
                })
            
            # If we couldn't extract structured travel info, fallback to general processing
            if not (flight_info or hotel_info or activity_info):
                process_generic_excel_sheet(df, sheet_name, all_chunks, chunks_with_metadata, filename)
                
        except Exception as e:
            print(f"Error processing sheet {sheet_name} as travel plan: {str(e)}")
            # Fallback to generic processing
            try:
                df = pd.read_excel(excel_content, sheet_name=sheet_name)
                process_generic_excel_sheet(df, sheet_name, all_chunks, chunks_with_metadata, filename)
            except Exception as inner_e:
                print(f"Fallback processing failed for sheet {sheet_name}: {str(inner_e)}")

def process_structured_database(excel_content, sheet_names, all_chunks, chunks_with_metadata, filename):
    """Extract information from a structured database Excel file"""
    for sheet_idx, sheet_name in enumerate(sheet_names, 1):
        try:
            df = pd.read_excel(excel_content, sheet_name=sheet_name)
            process_generic_excel_sheet(df, sheet_name, all_chunks, chunks_with_metadata, filename)
        except Exception as e:
            print(f"Error processing sheet {sheet_name} as database: {str(e)}")

def process_generic_excel_sheet(df, sheet_name, all_chunks, chunks_with_metadata, filename):
    """Process any Excel sheet in a generic way to extract text"""
    try:
        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            return
            
        # Get column descriptions
        columns_desc = f"Columns in sheet {sheet_name}: {', '.join(str(col) for col in df.columns)}"
        
        # Create chunks from the dataframe
        # First, a summary chunk with column info and basic stats
        summary = f"Excel Sheet: {sheet_name}\n{columns_desc}\n"
        summary += f"Contains {len(df)} rows and {len(df.columns)} columns.\n"
        
        # Add data type information
        summary += "Data types:\n"
        for col in df.columns:
            summary += f"  {col}: {df[col].dtype}\n"
            
        # Add some basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary += "\nNumeric column statistics:\n"
            for col in numeric_cols:
                summary += f"  {col}: Min={df[col].min()}, Max={df[col].max()}, Avg={df[col].mean():.2f}\n"
        
        # Add the summary as a chunk
        all_chunks.append(summary)
        chunks_with_metadata.append({
            "text": summary,
            "source": filename,
            "page": f"Sheet {sheet_name} (Summary)"
        })
        
        # Convert dataframe to text chunks
        # Process by groups of rows to create manageable chunks
        rows_per_chunk = 10
        for i in range(0, len(df), rows_per_chunk):
            chunk_df = df.iloc[i:i+rows_per_chunk]
            chunk_text = f"Excel Sheet: {sheet_name} (Rows {i+1}-{min(i+rows_per_chunk, len(df))})\n\n"
            
            # Convert chunk to text
            for _, row in chunk_df.iterrows():
                for col in chunk_df.columns:
                    if pd.notna(row[col]):
                        # Format dates and times specially
                        if isinstance(row[col], (datetime.datetime, datetime.date)):
                            chunk_text += f"{col}: {row[col].strftime('%Y-%m-%d %H:%M:%S' if isinstance(row[col], datetime.datetime) else '%Y-%m-%d')}, "
                        else:
                            chunk_text += f"{col}: {row[col]}, "
                chunk_text += "\n"
            
            if len(chunk_text.strip()) > 10:  # Only add if there's meaningful content
                all_chunks.append(chunk_text)
                chunks_with_metadata.append({
                    "text": chunk_text,
                    "source": filename,
                    "page": f"Sheet {sheet_name} (Rows {i+1}-{min(i+rows_per_chunk, len(df))})"
                })
                
    except Exception as e:
        print(f"Error in generic Excel processing for sheet {sheet_name}: {str(e)}")

def retrieve_chunks(query, top_k=3):
    """
    Retrieve top-k relevant chunks from FAISS based on query.
    Top-k=3 chosen as a balance between context richness and response brevity.
    """
    global vector_db, chunks_with_metadata
    
    # Check if vector_db is initialized
    if vector_db is None or len(chunks_with_metadata) == 0:
        print("Vector database is not initialized or no chunks are available")
        raise ValueError("No documents have been processed yet. Please upload and process files before asking questions.")
    
    try:
        # Handle the case where top_k is greater than the number of chunks
        top_k = min(top_k, len(chunks_with_metadata))
        
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        distances, indices = vector_db.search(query_embedding, top_k)
        
        # Check if indices are valid
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(chunks_with_metadata)]
        retrieved_chunks = [chunks_with_metadata[idx] for idx in valid_indices]
        
        # If no chunks were retrieved, return a message
        if not retrieved_chunks:
            raise ValueError("Could not find relevant information in the processed documents.")
            
        return retrieved_chunks
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        if "Faiss index needs to be trained" in str(e):
            raise ValueError("The search index is not ready. Please upload and process files before asking questions.")
        else:
            raise ValueError(f"Error retrieving relevant information: {str(e)}")

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def call_openai_api(prompt):
    """Call OpenAI API with retry logic for robustness"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # You can change to "gpt-4o-mini" or other models
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error calling OpenAI API: {str(e)}")

def generate_answer(query, retrieved_chunks):
    """
    Generate an answer using retrieved chunks and OpenAI API.
    """
    # Construct prompt with context
    context = "\n\n".join([f"Source: {chunk['source']} (Page {chunk['page']}):\n{chunk['text']}" 
                           for chunk in retrieved_chunks])
    
    prompt = f"""Based on the following context, answer the query: '{query}'
    If the answer is not in the context, say "I don't have enough information to answer this question."
    
    Context:
    {context}
    
    Query: {query}
    Answer:"""
    
    try:
        response = call_openai_api(prompt)
        sources = [(chunk['source'], chunk['page']) for chunk in retrieved_chunks]
        return response, sources
    except Exception as e:
        return f"Error generating answer: {str(e)}", []

def fixed_size_chunking(text, chunk_size=500, overlap=100):
    """
    Fallback method: Fixed-size chunking with overlap.
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

def process_google_sheet(sheet_url):
    """
    Process a Google Sheet URL: extract data, chunk, and store in FAISS.
    """
    global vector_db, chunks_with_metadata
    
    print(f"Processing Google Sheet: {sheet_url}")
    all_new_chunks = []
    
    try:
        # Extract the sheet ID from the URL
        sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not sheet_id_match:
            raise ValueError("Could not extract sheet ID from URL")
        
        sheet_id = sheet_id_match.group(1)
        
        # Use the Google Sheets API to export as CSV (public sheets only)
        sheet_data = fetch_google_sheet_data(sheet_id)
        
        if not sheet_data:
            raise ValueError("No data could be retrieved from the Google Sheet")
            
        # Process each sheet in the Google Sheet
        sheet_names = list(sheet_data.keys())
        
        # Check if it might be a travel plan (similar to Excel processing)
        is_travel_plan = False
        travel_keywords = ['flight', 'hotel', 'accommodation', 'itinerary', 'reservation', 
                          'departure', 'arrival', 'check-in', 'check-out', 'travel']
        
        for sheet_name, df in sheet_data.items():
            try:
                # Convert column names to lowercase for case-insensitive matching
                headers = [str(col).lower() for col in df.columns]
                
                # Check for travel keywords in headers
                if any(keyword in ' '.join(headers) for keyword in travel_keywords):
                    is_travel_plan = True
                    break
                
                # Check first few rows for travel-related content
                sample_content = ' '.join(df.head(5).values.flatten().astype(str).tolist()).lower()
                if any(keyword in sample_content for keyword in travel_keywords):
                    is_travel_plan = True
                    break
            except Exception as e:
                print(f"Error checking sheet {sheet_name} for travel keywords: {str(e)}")
        
        # Process each sheet
        for sheet_name, df in sheet_data.items():
            try:
                if is_travel_plan:
                    process_gsheet_travel_data(df, sheet_name, all_new_chunks, sheet_url)
                else:
                    process_generic_excel_sheet(df, sheet_name, all_new_chunks, chunks_with_metadata, sheet_url)
            except Exception as e:
                print(f"Error processing Google Sheet '{sheet_name}': {str(e)}")
        
        # If new chunks were extracted, add to the vector database
        if all_new_chunks:
            try:
                # Extract just the text for embedding
                new_chunk_texts = [chunk["text"] for chunk in all_new_chunks]
                
                # Add to the global chunks list
                for chunk in all_new_chunks:
                    chunks_with_metadata.append(chunk)
                
                # Get embeddings for the new chunks
                new_embeddings = model.encode(new_chunk_texts, convert_to_numpy=True)
                
                # If the vector_db doesn't exist yet, create it
                if vector_db is None:
                    dimension = new_embeddings.shape[1]
                    vector_db = faiss.IndexFlatIP(dimension)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(new_embeddings)
                
                # Add to the existing index
                vector_db.add(new_embeddings)
            except Exception as e:
                print(f"Error adding Google Sheet data to vector database: {str(e)}")
                raise
    
    except Exception as e:
        print(f"Error processing Google Sheet: {str(e)}")
        traceback.print_exc()
        raise
    
    return len(all_new_chunks)

def fetch_google_sheet_data(sheet_id):
    """
    Fetch data from a Google Sheet using the export feature.
    This works for publicly accessible sheets or those shared with "Anyone with the link" view access.
    """
    sheet_data = {}
    
    try:
        # First, try to get the list of sheet names
        sheet_info_url = f"https://spreadsheets.google.com/feeds/worksheets/{sheet_id}/public/basic?alt=json"
        response = requests.get(sheet_info_url)
        
        if response.status_code != 200:
            # Try the older export format which may work in some cases
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            response = requests.get(csv_url)
            
            if response.status_code == 200:
                # Successfully got the CSV data
                df = pd.read_csv(io.StringIO(response.text))
                sheet_data["Sheet1"] = df
                return sheet_data
            else:
                raise ValueError(f"Google Sheet is not publicly accessible or cannot be accessed. Make sure it is shared with 'Anyone with the link' view access.")
        
        # Parse the JSON to get sheet names
        sheet_info = response.json()
        sheets = sheet_info.get('feed', {}).get('entry', [])
        
        if not sheets:
            raise ValueError("No sheets found in the Google Sheet")
        
        # Process each sheet
        for sheet in sheets:
            sheet_name = sheet.get('title', {}).get('$t', 'Unknown')
            sheet_id_part = sheet.get('id', {}).get('$t', '').split('/')[-1]
            
            # Get the sheet data as CSV
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_id_part}"
            sheet_response = requests.get(csv_url)
            
            if sheet_response.status_code == 200:
                # Convert CSV to DataFrame
                df = pd.read_csv(io.StringIO(sheet_response.text))
                sheet_data[sheet_name] = df
    
    except Exception as e:
        print(f"Error fetching Google Sheet data: {str(e)}")
        raise
    
    return sheet_data

def process_gsheet_travel_data(df, sheet_name, all_chunks, sheet_url):
    """Process Google Sheet data that appears to be a travel plan"""
    # Same logic as process_travel_plan but adapted for direct dataframe input
    # Travel-related fields to look for
    flight_fields = ['airline', 'flight', 'departure', 'arrival', 'origin', 'destination']
    hotel_fields = ['hotel', 'accommodation', 'check-in', 'check-out', 'stay']
    activity_fields = ['activity', 'event', 'tour', 'visit', 'sightseeing']
    
    try:
        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            return
            
        # Convert headers to string and lowercase for matching
        headers = [str(col).lower() for col in df.columns]
        
        # Extract flight details
        flight_info = []
        flight_cols = [i for i, header in enumerate(headers) 
                       if any(field in header for field in flight_fields)]
        
        if flight_cols:
            for _, row in df.iterrows():
                flight_data = {}
                for col_idx in flight_cols:
                    col_name = df.columns[col_idx]
                    value = row[col_name]
                    if pd.notna(value):
                        flight_data[str(col_name)] = str(value)
                
                if flight_data:
                    flight_info.append(flight_data)
        
        # Extract hotel details similarly
        hotel_info = []
        hotel_cols = [i for i, header in enumerate(headers) 
                      if any(field in header for field in hotel_fields)]
        
        if hotel_cols:
            # ...similar code as for flight_info...
            for _, row in df.iterrows():
                hotel_data = {}
                for col_idx in hotel_cols:
                    col_name = df.columns[col_idx]
                    value = row[col_name]
                    if pd.notna(value):
                        hotel_data[str(col_name)] = str(value)
                
                if hotel_data:
                    hotel_info.append(hotel_data)
        
        # Extract activity details similarly
        activity_info = []
        activity_cols = [i for i, header in enumerate(headers) 
                         if any(field in header for field in activity_fields)]
        
        if activity_cols:
            # ...similar code as for flight_info...
            for _, row in df.iterrows():
                activity_data = {}
                for col_idx in activity_cols:
                    col_name = df.columns[col_idx]
                    value = row[col_name]
                    if pd.notna(value):
                        activity_data[str(col_name)] = str(value)
                
                if activity_data:
                    activity_info.append(activity_data)
        
        # Create a structured summary of the travel information
        summary = f"Travel Plan Information from Google Sheet (Sheet: {sheet_name}):\n\n"
        
        if flight_info:
            summary += "Flight Details:\n"
            for i, flight in enumerate(flight_info, 1):
                summary += f"  Flight {i}: {', '.join([f'{k}: {v}' for k, v in flight.items()])}\n"
            summary += "\n"
        
        if hotel_info:
            summary += "Hotel Details:\n"
            for i, hotel in enumerate(hotel_info, 1):
                summary += f"  Hotel {i}: {', '.join([f'{k}: {v}' for k, v in hotel.items()])}\n"
            summary += "\n"
        
        if activity_info:
            summary += "Activity Details:\n"
            for i, activity in enumerate(activity_info, 1):
                summary += f"  Activity {i}: {', '.join([f'{k}: {v}' for k, v in activity.items()])}\n"
        
        # Add the summary as a chunk
        if len(summary.strip()) > 10:  # Only add if there's meaningful content
            chunks_with_metadata.append({
                "text": summary,
                "source": f"Google Sheet: {sheet_url}",
                "page": f"Sheet: {sheet_name}"
            })
            all_chunks.append({
                "text": summary,
                "source": f"Google Sheet: {sheet_url}",
                "page": f"Sheet: {sheet_name}"
            })
        
        # If we couldn't extract structured travel info, fallback to general processing
        if not (flight_info or hotel_info or activity_info):
            process_generic_excel_sheet(df, sheet_name, all_chunks, chunks_with_metadata, f"Google Sheet: {sheet_url}")
            
    except Exception as e:
        print(f"Error processing Google Sheet as travel plan: {str(e)}")
        # Fallback to generic processing
        try:
            process_generic_excel_sheet(df, sheet_name, all_chunks, chunks_with_metadata, f"Google Sheet: {sheet_url}")
        except Exception as inner_e:
            print(f"Fallback processing failed for Google Sheet: {str(inner_e)}")

def process_generic_excel_sheet(df, sheet_name, all_chunks, chunks_with_metadata, filename):
    """Process any Excel sheet or Google Sheet in a generic way to extract text"""
    # ...existing code...
    # Need to check if we're adding to both all_chunks and chunks_with_metadata
    # or just one of them depending on context
    try:
        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            return
            
        # Get column descriptions
        columns_desc = f"Columns in sheet {sheet_name}: {', '.join(str(col) for col in df.columns)}"
        
        # Create chunks from the dataframe
        # First, a summary chunk with column info and basic stats
        summary = f"Sheet: {sheet_name}\n{columns_desc}\n"
        summary += f"Contains {len(df)} rows and {len(df.columns)} columns.\n"
        
        # Add data type information
        summary += "Data types:\n"
        for col in df.columns:
            summary += f"  {col}: {df[col].dtype}\n"
            
        # Add some basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary += "\nNumeric column statistics:\n"
            for col in numeric_cols:
                summary += f"  {col}: Min={df[col].min()}, Max={df[col].max()}, Avg={df[col].mean():.2f}\n"
        
        # Add the summary as a chunk
        chunk_data = {
            "text": summary,
            "source": filename,
            "page": f"Sheet {sheet_name} (Summary)"
        }
        
        # Check if all_chunks is an actual list (not our global one)
        if isinstance(all_chunks, list):
            all_chunks.append(chunk_data)
        
        # Always update the global chunks_with_metadata
        chunks_with_metadata.append(chunk_data)
        
        # Convert dataframe to text chunks
        # Process by groups of rows to create manageable chunks
        rows_per_chunk = 10
        for i in range(0, len(df), rows_per_chunk):
            chunk_df = df.iloc[i:i+rows_per_chunk]
            chunk_text = f"Sheet: {sheet_name} (Rows {i+1}-{min(i+rows_per_chunk, len(df))})\n\n"
            
            # Convert chunk to text
            for _, row in chunk_df.iterrows():
                for col in chunk_df.columns:
                    if pd.notna(row[col]):
                        # Format dates and times specially
                        if isinstance(row[col], (datetime.datetime, datetime.date)):
                            chunk_text += f"{col}: {row[col].strftime('%Y-%m-%d %H:%M:%S' if isinstance(row[col], datetime.datetime) else '%Y-%m-%d')}, "
                        else:
                            chunk_text += f"{col}: {row[col]}, "
                chunk_text += "\n"
            
            if len(chunk_text.strip()) > 10:  # Only add if there's meaningful content
                chunk_data = {
                    "text": chunk_text,
                    "source": filename,
                    "page": f"Sheet {sheet_name} (Rows {i+1}-{min(i+rows_per_chunk, len(df))})"
                }
                
                # Check if all_chunks is an actual list (not our global one)
                if isinstance(all_chunks, list):
                    all_chunks.append(chunk_data)
                
                # Always update the global chunks_with_metadata
                chunks_with_metadata.append(chunk_data)
                
    except Exception as e:
        print(f"Error in generic sheet processing for sheet {sheet_name}: {str(e)}")

# ...other existing functions...