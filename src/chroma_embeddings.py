import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize the ChromaDB client with the URL to your local instance
client = chromadb.Client(chromadb.Settings(chroma_api_impl="rest", chroma_server_host="localhost", chroma_server_http_port="7100"))

# Set up a collection for storing embeddings
collection_name = "pdf_embeddings"
if collection_name in client.list_collections():
    collection = client.get_collection(collection_name)
else:
    collection = client.create_collection(collection_name)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def embed_and_store_pdf(pdf_path):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Generate embeddings
    embedding = model.encode(text)
    
    # Store the embedding in ChromaDB
    pdf_name = os.path.basename(pdf_path)
    collection.add(
        documents=[text],
        metadatas=[{"pdf_name": pdf_name}],
        ids=[pdf_name]
    )

# Path to the directory containing the PDF files
pdf_directory = "/home/nikodem-ub1/github/RAG-client/data"

# Iterate over PDF files and store embeddings
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        embed_and_store_pdf(pdf_path)

print(f"Embeddings for PDFs in {pdf_directory} have been stored in ChromaDB.")
