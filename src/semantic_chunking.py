import os
import fitz  # PyMuPDF to read PDFs
import chromadb
from tqdm import tqdm
import requests

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from chromadb.config import Settings
from requests.auth import AuthBase
from chromadb import Client

from llama_index.llms.ollama import Ollama

# Connect to ChromaDB server
db = chromadb.HttpClient(host="localhost", port=7100)

# Directory where PDF files are stored
pdf_directory = "/home/nikodem-ub1/github/RAG-client/data"

# LLM and embedding model initialization
llm = Ollama(model="llama3.1", request_timeout=30.0, base_url="http://localhost:7101")
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# Function to chunk text semantically based on the steps from the article
def semantic_chunking(text):
    # Initialize the SemanticChunker with the embedding model and use the percentile method for splitting
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    
    # Split the text into chunks using the chunker and return the semantic chunks
    semantic_chunks = semantic_chunker.create_documents([text])
    
    return semantic_chunks

# Function to process a single PDF file and add to ChromaDB
def process_pdf(pdf_path, collection):
    # Open the PDF document
    doc = fitz.open(pdf_path)
    
    # Loop through each page in the PDF
    for page_number in tqdm(range(len(doc)), desc=f"Processing {os.path.basename(pdf_path)}"):
        page = doc.load_page(page_number)
        
        # Extract the text from the page
        text = page.get_text("text")
        
        # Use semantic chunking to split the text into meaningful chunks
        chunks = semantic_chunking(text)
        
        # Loop through each chunk and generate embeddings
        for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc=f"Page {page_number}", leave=False):
            # Extract the actual text content from the Document object
            chunk_text = chunk.page_content  # Assuming `page_content` holds the actual text of the chunk

            # Embed the text content of the chunk (instead of the Document object)
            embedding = embed_model.embed_documents([chunk_text])  # Ensure it's passed as a list
            
            # Add the chunk information and embedding to the ChromaDB collection
            collection.add(
                ids=[f"{os.path.basename(pdf_path)}_page_{page_number}_chunk_{i}"],
                embeddings=embedding,  # Embedding should be a list of vectors
                documents=[chunk_text],  # Store the chunk text, not the Document object
                metadatas=[{
                    "pdf_name": os.path.basename(pdf_path), 
                    "page_number": page_number, 
                    "chunk_index": i
                }]
            )

# Function to process all PDFs in the specified directory
def process_all_pdfs(pdf_directory):
    # Create or get the collection in ChromaDB
    collection = db.get_or_create_collection("compiler_writing_books")
    
    # Loop through all PDF files in the specified directory
    for pdf_file in tqdm(os.listdir(pdf_directory), desc="Processing PDFs"):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            # Process each PDF file and add chunks to the collection
            process_pdf(pdf_path, collection)

# Start processing all PDFs
process_all_pdfs(pdf_directory)

print("Semantic chunking and embedding process completed.")
