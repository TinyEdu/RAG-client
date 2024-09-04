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

# Initialize ChromaDB client
db = HttpClient(host="localhost", port=7100)

# Set up the embedder
embedder = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

from langchain_ollama import OllamaLLM

# Set up the LLM
llm = OllamaLLM(model="llama3.1", request_timeout=30.0, base_url="http://localhost:7101")

# Use the Settings class to configure client settings for Chroma
client_settings = Settings(
    chroma_server_host="localhost", 
    chroma_server_http_port="7100"
)

# Initialize Chroma vector store with the proper client settings
vectorstore = Chroma(
    collection_name="pdf_chunks", 
    embedding_function=embedder, 
    client_settings=client_settings  # Use Settings instance instead of a dictionary
)

# Initialize the retriever from the vector store
db_retriever = vectorstore.as_retriever()

# Define function to retrieve documents
def retrieve_docs(query, collection_name="pdf_chunks", k=5):
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
