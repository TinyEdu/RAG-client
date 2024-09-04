import os
import fitz  # PyMuPDF to read PDFs
import chromadb
from tqdm import tqdm
import requests

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from chromadb.config import Settings
from requests.auth import AuthBase
from chromadb import Client

from llama_index.llms.ollama import Ollama  


db = chromadb.HttpClient(host="localhost", port=7100)
print(db.list_collections())

pdf_directory = "/home/nikodem-ub1/github/RAG-client/data"


llm = Ollama(model="llama3.1", request_timeout=30.0, base_url="http://localhost:7101")
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# Function to chunk text semantically
def semantic_chunking(text, chunk_size=1000):
    # Initialize the SemanticChunker with the embedding model
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    
    # Split the text into chunks using the chunker
    semantic_chunks = semantic_chunker.create_documents([text])
    
    return semantic_chunks

# Function to process a single PDF file
def process_pdf(pdf_path, collection):
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text")
        chunks = semantic_chunking(text)
        for i, chunk in enumerate(chunks):
            # Extract the actual text content from the Document object
            chunk_text = chunk.page_content  # Assuming `page_content` holds the actual text of the chunk

            print(dir(embed_model))  
            # Get the embedding for the text content, not the Document object
            embedding = embed_model.embed_documents(chunk_text)
            
            collection.add(
                ids=[f"{os.path.basename(pdf_path)}_page_{page_number}_chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk_text],  # Store the chunk text, not the Document object
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
