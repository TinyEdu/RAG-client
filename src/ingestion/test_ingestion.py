# search.py

import json
import chromadb
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from llama_index.llms.ollama import Ollama

from utils import *


# Load configuration
with open('/home/nikodem-ub1/github/RAG-client/config.json', 'r') as file:
    config = json.load(file)


# Connect to ChromaDB server
db = chromadb.HttpClient(host=config["chromadb"]["host"], port=config["chromadb"]["port"])

# LLM and embedding model initialization
llm = Ollama(model=config["models"]["llm"], request_timeout=30.0, base_url="http://" + str(config["ollama"]["host"]) + ":" + str(config["ollama"]["port"]))
embed_model = FastEmbedEmbeddings(model_name=config["models"]["embeddings"])


def search_collection(collection_name, query, embed_model, top_k=5):
    """
    Search for the top_k most relevant documents in the collection for the given query.
    """
    # Get the collection from ChromaDB
    collection = db.get_collection(collection_name)
    
    # Embed the query using the embedding model (using `embed_query` instead of `embed_text`)
    query_embedding = embed_model.embed_query(query)
    
    # Perform a search in ChromaDB using the query embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    # Return the results
    return results



if __name__ == "__main__":
    # Specify which collection you want to query and the search query
    collection_name = "compiler_writing_books"  # Replace with the name of the collection you want to search
    query = "write me an example visitor pattern class in c++"  # Replace with your search query
    
    # Perform the search and retrieve the top results
    results = search_collection(collection_name, query, embed_model)
    
    # Inspect the structure of the results
    print("Raw Search Results:")
    print(results)  # This will help us understand the structure of the results
    
    # Assuming results['documents'] contains a list of lists, iterate over the results
    print("\nSearch Results:")
    for i, result in enumerate(results['documents']):
        print(f"Result {i + 1}:")
        print(f"Document: {result}")
        print()
