import json
import os
import logging
import numpy as np

from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_ollama import OllamaLLM
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma

COLLECTION_THRESHOLD = 0


class RAGPipeline:
    def __init__(self, config_path, llm_model="llama3.1", llm_base_url="http://localhost:7101", chroma_host="localhost", chroma_port=7100):
        # Load the JSON data from the configuration file
        with open(config_path, 'r') as config_file:
            self.config_data = json.load(config_file)

        self.collections = self.config_data.get('collections', {})

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize ChromaDB client
        logging.info("Initializing ChromaDB client")
        self.db = HttpClient(host=chroma_host, port=chroma_port)

        logging.info("Fetching all collections:")
        # Fetch all collections
        collections = self.db.list_collections()
        for collection in collections:
            print(f"Collection: {collection.name}, Document count: {collection.count()}")

        # Output the collections
        for collection in collections:
            print(collection.name)

        # Set up the embedder
        logging.info("Setting up FastEmbedEmbeddings")
        self.embedder = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        # Set up the LLM
        logging.info("Setting up the Ollama LLM")
        self.llm = OllamaLLM(model=llm_model, request_timeout=30.0, base_url=llm_base_url)

        # Configure client settings for Chroma
        logging.info("Configuring Chroma client settings")
        self.client_settings = Settings(
            chroma_server_host=chroma_host,
            chroma_server_http_port=str(chroma_port)
        )


    def select_collections_semantically(self, query):
        # Embed the query
        query_embedding = self.embedder.embed_documents([query])[0]

        # List to store collections and their similarities
        collection_similarities = []
        COLLECTION_THRESHOLD = 0.5  

        # Iterate through collections and calculate their similarity to the query
        for collection_name, collection_description in self.collections.items():
            collection_embedding = self.embedder.embed_documents([collection_description])[0]

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, collection_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(collection_embedding))

            # If similarity is above the threshold, add it to the list
            if similarity >= COLLECTION_THRESHOLD:
                print(f"Collection: {collection_name}, Similarity: {similarity}")
                collection_similarities.append((collection_name, similarity))

        # Sort collections by similarity in descending order
        collection_similarities = sorted(collection_similarities, key=lambda x: x[1], reverse=True)

        # Return the top N collections based on similarity
        N=2
        return [collection[0] for collection in collection_similarities[:N]] if collection_similarities else None

    def retrieve_docs(self, query, collection_name, k=5):
        # Initialize Chroma vector store with the proper client settings
        logging.info("Initializing Chroma vector store")
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedder,
            client_settings=self.client_settings
        )

        # Initialize the retriever from the vector store
        logging.info("Setting up document retriever from vector store")
        db_retriever = vectorstore.as_retriever()

        logging.info(f"Retrieving documents for query: {query}")
        retriever = MultiQueryRetriever.from_llm(
            retriever=db_retriever, llm=self.llm
        )

        # Retrieve the documents using MultiQueryRetriever
        docs = retriever.get_relevant_documents(query=query)

        print("______________________")
        print("TEST:      ->", retriever.get_relevant_documents(query="abstract syntax tree"))
        logging.info(f"Retrieved {len(docs)} documents")
        return docs

    def generate_answer_with_context(self, query, contexts):
        logging.info(f"Generating answer with retrieved context for query: {query}")
        # Combine the retrieved context into a single string
        context_str = "\n"
        for context in contexts:
            context_str.join([doc.page_content for doc in context])


        print("____________\n\n")
        print(context_str)
        print("____________\n\n")

        # Define a prompt template for the LLM
        rag_template = """Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.
                          User's Query: {query}
                          Context: {context}"""

        # Create the final prompt
        prompt = rag_template.format(query=query, context=context_str)

        # Wrap the prompt in a list (since the LLM expects a list of prompts)
        return self.llm.generate([prompt])  # Pass prompt as a list of strings


    def run(self, query, k=5):
       logging.info(f"Starting RAG pipeline for query: {query}")

       # Step 0: Select the most semantically relevant collection
       collections = self.select_collections_semantically(query)

       retrieved_docs = []  # Initialize as a list of documents
       if collections is None:
           logging.warning("No semantically relevant collection found")
       else:
           # Step 1: Retrieve relevant documents from ChromaDB
           for collection in collections:
               docs = self.retrieve_docs(query, collection, k)
               print("+++++++++++++++++++")
               print(docs)
               print("+++++++++++++++++++")
               retrieved_docs.extend(docs)  # Collect documents instead of concatenating strings

       # Step 2: Generate an answer using the retrieved context and the LLM
       answer = self.generate_answer_with_context(query, retrieved_docs)  # Pass the document objects list

       logging.info(f"Generated answer for query: {query}")
       generations = answer.generations
       text_answer = generations[0][0].text
       return text_answer

# _________________________________________
# EXAMPLE USAGE
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "/home/nikodem-ub1/github/RAG-client/config.json")
    pipeline = RAGPipeline(config_path)
    print("--- --- --- Response --- --- ---\n", pipeline.run("write me an example visitor pattern class in c++"))

