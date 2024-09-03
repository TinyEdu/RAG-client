import os
import fitz  # PyMuPDF to read PDFs
import chromadb
from tqdm import tqdm
import requests

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

import chromadb
from chromadb.config import Settings
from requests.auth import AuthBase
from chromadb import Client
# Function to get text embeddings from the locally hosted LLM (Ollama)
def get_embedding(text, model="llama3.1", request_timeout=30.0, base_url="http://localhost:7101"):
    url = f"{base_url}/v1/embeddings"
    payload = {"model": model, "text": text}
    response = requests.post(url, json=payload, timeout=request_timeout)
    response.raise_for_status()
    return response.json()["embedding"]

# ----------------------------------

# Configure the connection to Chroma DB
db = chromadb.HttpClient(host='localhost', port=7100) 

# ----------------------------------

# Directory containing PDF files
pdf_directory = "/home/nikodem-ub1/github/RAG-client/data"

from llama_index.llms.ollama import Ollama  

llm = Ollama(model="llama3.1", request_timeout=30.0, base_url="http://localhost:11111")
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Function to chunk text semantically
def semantic_chunking(text, chunk_size=1000):
    # https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5
    # `percentile` (default) — In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.

    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])

    return semantic_chunks
    


# Function to process a single PDF file
def process_pdf(pdf_path, collection):
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text")
        chunks = semantic_chunking(text)
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            collection.add(
                ids=[f"{os.path.basename(pdf_path)}_page_{page_number}_chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"pdf_name": os.path.basename(pdf_path), "page_number": page_number, "chunk_index": i}]
            )

# Function to process all PDFs in the specified directory
def process_all_pdfs(pdf_directory):
    collection = db.get_or_create_collection("pdf_chunks")
    
    for pdf_file in tqdm(os.listdir(pdf_directory)):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            process_pdf(pdf_path, collection)

# Start processing all PDFs
process_all_pdfs(pdf_directory)

print("Semantic chunking and embedding process completed.")
