# semantic_file_chunking.py
import os
import fitz
import git

from tqdm.auto import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from utils import fetch_website_content


def semantic_chunking(text, embed_model):
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    semantic_chunks = semantic_chunker.create_documents([text])

    return semantic_chunks

# Function to process different file types
def process_file(filepath, collection, embed_model):
    _, file_extension = os.path.splitext(filepath)

    # Handle PDF files
    if file_extension.lower() == '.pdf':
        doc = fitz.open(filepath)
        for page_number in tqdm(range(len(doc)), desc=f"Processing {os.path.basename(filepath)}"):
            page = doc.load_page(page_number)
            text = page.get_text("text")
            chunks = semantic_chunking(text, embed_model)

            for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc=f"Page {page_number}", leave=False):
                chunk_text = chunk.page_content
                embedding = embed_model.embed_documents([chunk_text])

                collection.add(
                    ids=[f"{os.path.basename(filepath)}_page_{page_number}_chunk_{i}"],
                    embeddings=embedding,
                    documents=[chunk_text],
                    metadatas=[{
                        "filename": os.path.basename(filepath),
                        "page_number": page_number,
                        "chunk_index": i
                    }]
                )
    # Handle text-based files (.cpp, .c, .h, .hpp, .sh, .txt, .md)
    elif file_extension.lower() in ['.cpp', '.c', '.h', '.hpp', '.sh', '.txt', '.md']:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            chunks = semantic_chunking(text, embed_model)

            for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc=f"Processing {os.path.basename(filepath)}", leave=False):
                chunk_text = chunk.page_content
                embedding = embed_model.embed_documents([chunk_text])

                collection.add(
                    ids=[f"{os.path.basename(filepath)}_chunk_{i}"],
                    embeddings=embedding,
                    documents=[chunk_text],
                    metadatas=[{
                        "filename": os.path.basename(filepath),
                        "chunk_index": i
                    }]
                )
    # Handle HTML files
    elif file_extension.lower() == '.html':
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            chunks = semantic_chunking(text, embed_model)

            for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc=f"Processing {os.path.basename(filepath)}", leave=False):
                chunk_text = chunk.page_content
                embedding = embed_model.embed_documents([chunk_text])

                collection.add(
                    ids=[f"{os.path.basename(filepath)}_chunk_{i}"],
                    embeddings=embedding,
                    documents=[chunk_text],
                    metadatas=[{
                        "filename": os.path.basename(filepath),
                        "chunk_index": i
                    }]
                )
    else:
        print(f"Unsupported file type: {file_extension}")

def process_website(url, collection, embed_model):
    content = fetch_website_content(url)
    chunks = semantic_chunking(content, embed_model)
    
    # Loop through each chunk and generate embeddings
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="url", leave=False):
        chunk_text = chunk.page_content  

        # Embed the text content of the chunk
        embedding = embed_model.embed_documents([chunk_text])  # Ensure it's passed as a list
        
        collection.add(
            ids=[f"www_{url}_chunk_{i}"],
            embeddings=embedding,  
            documents=[chunk_text],
            metadatas=[{
                "website_url": url, 
                "chunk_index": i
            }]
        )


def process_repository(repo_url, clone_dir, extensions):
    try:
        git.Repo.clone_from(repo_url, clone_dir)
        print(f"Repository cloned successfully to {clone_dir}")
    except git.GitCommandError as e:
        print(f"Error cloning repository: {e}")
        return None

    # Dictionary to store the file paths and their contents
    file_contents = {}

    # Walk through the cloned directory and find files with the given extensions
    for root, dirs, files in os.walk(clone_dir):
        for file in files:
            # Check if the file has one of the specified extensions
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                
                # Read and store the file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_contents[file_path] = f.read()
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    return file_contents
