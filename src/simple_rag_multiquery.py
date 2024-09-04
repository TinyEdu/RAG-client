import chromadb
from chromadb.utils import embedding_functions
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from chat import chat

db = chromadb.HttpClient(host="localhost", port=7100)
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Function to embed the query using the same embedding model
def embed_query(query):
    return embed_model.embed_documents([query])[0] 

# Function to retrieve relevant documents from ChromaDB using the embedding
def retrieve_docs(query, collection_name="pdf_chunks", k=5):
    query_embedding = embed_query(query)
    collection = db.get_collection(collection_name)

    # Query the vector database to get the top-k most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    # ChromaDB returns a list of results. Let's extract the documents properly.
    retrieved_docs = results["documents"]

    # Since it's a list of lists, flatten the list to get all retrieved documents
    retrieved_docs = [doc for sublist in retrieved_docs for doc in sublist]

    return retrieved_docs

# Function to generate an answer using the retrieved context and Ollama LLM
def generate_answer_with_context(query, context):
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
    
    return chat(messages) 

# RAG pipeline function: retrieve context and generate an answer
def rag_pipeline(query, collection_name="pdf_chunks", k=5):
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs = retrieve_docs(query, collection_name, k)
    
    # Step 2: Generate an answer using the retrieved context and the LLM
    answer = generate_answer_with_context(query, retrieved_docs)
    
    return answer

# _________________________________________
# EXAMPLE
print("\n\nResponse:", rag_pipeline("Explain how does a parser work"))