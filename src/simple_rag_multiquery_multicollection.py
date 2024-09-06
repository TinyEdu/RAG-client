import json
import os

import logging
import numpy as np

from chromadb import HttpClient
from chromadb.config import Settings  # Import Settings class
from chromadb.utils import embedding_functions
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from chat import chat

from llama_index.llms.ollama import Ollama  
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

COLLECTION_THRESHOLD = 0.5

config_path = os.path.join(os.path.dirname(__file__), '../config.json')

# Load the JSON data from the file
with open(config_path, 'r') as config_file:
    config_data = json.load(config_file)

# Access the collections map
global collections 
collections = config_data.get('collections', {})


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize ChromaDB client
logging.info("Initializing ChromaDB client")
db = HttpClient(host="localhost", port=7100)

# Set up the embedder
logging.info("Setting up FastEmbedEmbeddings")
embedder = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

from langchain_ollama import OllamaLLM

# Set up the LLM
logging.info("Setting up the Ollama LLM")
llm = OllamaLLM(model="llama3.1", request_timeout=30.0, base_url="http://localhost:7101")

# Use the Settings class to configure client settings for Chroma
logging.info("Configuring Chroma client settings")
client_settings = Settings(
    chroma_server_host="localhost", 
    chroma_server_http_port="7100"
)

# Assuming collections is a list of collection objects, and each has a 'description' or 'metadata'
def select_collection_semantically(query):
    # Embed the query
    query_embedding = embedder.embed_documents([query])[0]
    
    best_collection = None
    highest_similarity = 0

    for collection_name, collection_description in collections.items(): 
        collection_embedding = embedder.embed_documents([collection_description])[0]
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, collection_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(collection_embedding))
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_collection = collection_name

    
    # check if the similarity is high enough to be considered
    if similarity > COLLECTION_THRESHOLD:
        return  None
    
    return best_collection


# Define function to retrieve documents
def retrieve_docs(query, collection_name="pdf_chunks", k=5):
    # Initialize Chroma vector store with the proper client settings
    logging.info("Initializing Chroma vector store")
    vectorstore = Chroma(
        collection_name=collection_name, 
        embedding_function=embedder, 
        client_settings=client_settings 
    )

    # Initialize the retriever from the vector store
    logging.info("Setting up document retriever from vector store")
    db_retriever = vectorstore.as_retriever()
    
    logging.info(f"Retrieving documents for query: {query}")
    retriever = MultiQueryRetriever.from_llm(
        retriever=db_retriever, llm=llm
    )

    docs = retriever.get_relevant_documents(query=query)

    query_embedding = embedder.embed_documents([query])[0]
    collection = db.get_collection(collection_name)

    # Query the vector database to get the top-k most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    # ChromaDB returns a list of results. Let's extract the documents properly.
    retrieved_docs = results["documents"]
    # flatten the list 
    retrieved_docs = [doc for sublist in retrieved_docs for doc in sublist]

    logging.info(f"Retrieved {len(retrieved_docs)} documents")
    return retrieved_docs

# Function to generate an answer using the retrieved context and Ollama LLM
def generate_answer_with_context(query, context):
    logging.info(f"Generating answer with retrieved context for query: {query}")
    # Combine the retrieved context into a single string
    context_str = "\n\n".join(context)
    
    # Define a prompt template for the LLM
    rag_template = """Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.
                      User's Query: {query}
                      Context: {context}"""
    
    # Create the final prompt
    prompt = rag_template.format(query=query, context=context_str)

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    return chat(messages) # @TODO: change it --> ollamaLLM

# RAG pipeline function: retrieve context and generate an answer
def rag_pipeline(query, k=5):
    logging.info(f"Starting RAG pipeline for query: {query}")

    # Step 0: Select the most semantically relevant collection
    collection_name = select_collection_semantically(query)

    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs = retrieve_docs(query, collection_name, k)
    
    # Step 2: Generate an answer using the retrieved context and the LLM
    answer = generate_answer_with_context(query, retrieved_docs)
    
    logging.info(f"Generated answer for query: {query}")
    return answer

# _________________________________________
# EXAMPLE
if __name__ == "__main__":
    logging.info("Running example query")
    print("\n\nResponse:", rag_pipeline("Explain how does a parser work"))