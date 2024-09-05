import json

import chromadb
from tqdm import tqdm

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from llama_index.llms.ollama import Ollama

from utils import *
from semantic_file_chunking import * 

with open('/home/nikodem-ub1/github/RAG-client/config.json', 'r') as file:
    config = json.load(file)


# Connect to ChromaDB server
db = chromadb.HttpClient(host=config["chromadb"]["host"], port=config["chromadb"]["port"])

# LLM and embedding model initialization
llm = Ollama(model=config["models"]["llm"], request_timeout=30.0, base_url="http://" + str(config["ollama"]["host"]) + ":" + str(config["ollama"]["port"]))
embed_model = FastEmbedEmbeddings(model_name=config["models"]["embeddings"])


# Function to process all files in the specified directory
def add_files_to_collection(files, collection_name, embed_model):
    # Create or get the given collection in ChromaDB
    collection = db.get_or_create_collection(collection_name)
    
    # Loop through all files in the specified directory
    for filepath in tqdm(files, desc="Processing files"):
        process_file(filepath, collection, embed_model)


# Function to process all website links
def add_websites_to_collection(urls, collection_name, embed_model):
    # Create or get the given collection in ChromaDB
    collection = db.get_or_create_collection(collection_name)
    
    # Loop through all given urls 
    for url in tqdm(urls, desc="Processing urls"):
        process_website(url, collection, embed_model)


def process_collection(collection_name, embed_model, directory_path):
    extensions = ['.pdf', '.cpp', '.c', '.h', '.hpp', '.sh', 'html', '.txt', '.md']

    files = get_all_files_with_types(extensions, directory_path)
    collection = db.get_or_create_collection(collection_name)

    for file in files:
        process_file(file, collection, embed_model)


if __name__ == "__main__":
    # Iterate over the "collections" section
    for collection_name, description in config['collections'].items():
        print(f"Collection Name: {collection_name}")
        print(f"Description: {description}")
        print()

        if config["data_path"][collection_name] == "":
            process_collection(collection_name, embed_model, config["data_path"][collection_name])

        process_repository("https://github.com/TinyEdu/VtC-compiler", "codebase", ['.cpp', '.c', '.h', '.hpp', '.sh'])

    print("Semantic chunking and embedding process completed.")


