import pdfplumber
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer
import re
import openai
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Download required NLTK data
nltk.download('punkt', quiet=True)

# Initialize global variables
vector_db = None
chunks_with_metadata = []
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer('all-MiniLM-L6-v2')

def process_documents(pdf_files):
    """
    Process uploaded PDF files: parse, chunk, embed, and store in FAISS.
    """
    global vector_db, chunks_with_metadata

    # Reset previous data
    chunks_with_metadata = []
    all_chunks = []

    # Parse PDFs using pdfplumber (chosen for its robustness and free availability)
    # pdfplumber is preferred over Unstructured because it provides reliable text extraction
    # and metadata like page numbers with a simple API, and it’s entirely free.
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    # Semantic chunking using NLTK sentence tokenization
                    # NLTK is free, widely used, and robust for sentence boundary detection.
                    sentences = nltk.sent_tokenize(text)
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

    # Fallback: If semantic chunking fails or is too complex, use fixed-size chunking
    # with 500 tokens and 100-token overlap (commented out for simplicity).
    # all_chunks = fixed_size_chunking(text, chunk_size=500, overlap=100)

    # Embed chunks using all-MiniLM-L6-v2 (free, fast, and high-performing)
    # Chosen over all-mpnet-base-v2 for lower memory footprint while maintaining good quality.
    embeddings = model.encode(all_chunks, convert_to_numpy=True)

    # Initialize FAISS index with cosine similarity
    dimension = embeddings.shape[1]
    vector_db = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    vector_db.add(embeddings)

    return len(all_chunks)

def retrieve_chunks(query, top_k=3):
    """
    Retrieve top-k relevant chunks from FAISS based on query.
    Top-k=3 chosen as a balance between context richness and response brevity.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = vector_db.search(query_embedding, top_k)
    retrieved_chunks = [chunks_with_metadata[idx] for idx in indices[0]]
    return retrieved_chunks

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