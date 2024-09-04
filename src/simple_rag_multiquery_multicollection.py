import json
import requests

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Initialize ChromaDB client
db = chromadb.HttpClient(host="localhost", port=7100)

# Embedding model setup (BAAI/bge-base-en-v1.5)
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Function to embed the query using the same embedding model
def embed_query(query):
    return embed_model.embed_documents([query])[0]  # Return the single embedding

# Function to retrieve relevant documents from multiple collections in ChromaDB using the embedding
def retrieve_docs_multi_collection(query, collection_names=["pdf_chunks", "other_collection"], k=5):
    # Embed the query using the embedding model
    query_embedding = embed_query(query)

    all_retrieved_docs = []

    # Iterate over each collection
    for collection_name in collection_names:
        try:
            # Get the collection where the embeddings are stored
            collection = db.get_collection(collection_name)

            # Query the vector database to get the top-k most similar chunks
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            # Print the structure of the results for debugging
            print(f"Results structure for collection '{collection_name}':", results)

            # ChromaDB returns a list of results. Let's extract the documents properly.
            retrieved_docs = results["documents"]

            # Since it's a list of lists, flatten the list to get all retrieved documents
            retrieved_docs = [doc for sublist in retrieved_docs for doc in sublist]

            # Append the retrieved documents to the overall list
            all_retrieved_docs.extend(retrieved_docs)

        except Exception as e:
            print(f"Error querying collection '{collection_name}': {e}")
            continue

    return all_retrieved_docs

def chat(messages):
    r = requests.post(
        "http://0.0.0.0:7101/api/chat",
        json={"model": "llama3.1", "messages": messages, "stream": True},
        stream=True
    )
    r.raise_for_status()
    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
            print(content, end="", flush=True)

        if body.get("done", False):
            message["content"] = output
            return message

# Function to generate an answer using the retrieved context and Ollama LLM
def generate_answer_with_context(query, context):
    # Combine the retrieved context into a single string
    context_str = "\n\n".join(context)
    
    # Define a prompt template for the LLM
    rag_template = """\
Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.

User's Query:
{query}

Context:
{context}
"""
    
    # Create the final prompt
    prompt = rag_template.format(query=query, context=context_str)

    messages = [
        {"role": "user", "content": prompt}
    ]
    # Assuming the correct method is `generate` or `ask`
    response = chat(messages)
    return response

# RAG pipeline function: retrieve context from multiple collections and generate an answer
def rag_pipeline(query, collection_names=["pdf_chunks", "other_collection"], k=5):
    # Step 1: Retrieve relevant documents from multiple collections in ChromaDB
    retrieved_docs = retrieve_docs_multi_collection(query, collection_names, k)
    
    # Step 2: Generate an answer using the retrieved context and the LLM
    answer = generate_answer_with_context(query, retrieved_docs)
    
    return answer

# Example query
query = "Explain how does a parser work"

# Run the RAG pipeline, querying multiple collections
collection_names = ["pdf_chunks", "other_collection"]  # Add more collection names as needed
response = rag_pipeline(query, collection_names)
print("Response:", response)
