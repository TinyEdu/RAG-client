# RAG-client

RAG-client is a tool designed to facilitate the creation of a Retrieval-Augmented Generation (RAG) application within the TinyEdu platform. This client integrates various components to extract, process, and query documents and code files using advanced language models.

## Features

- **Document Parsing**: Supports PDF and DOCX file parsing to extract text content.
- **Custom Document Loader**: Utilizes a custom loader to handle specific file types and integrate with the LlamaIndex for vector-based document indexing.
- **Code Analysis**: Includes a code reader tool for analyzing and returning the contents of code files.
- **Query Engine**: Implements a vector index for efficient text search and retrieval, integrated with a query engine powered by language models.
- **LLM Integration**: Interacts with Ollama and CodeLlama models for generating and processing queries.

## Requirements

- Python 3.x
- Required Python packages listed in `requirements.txt`.

## Usage

1. **Setup**: Clone the repository and install the required dependencies using `pip install -r requirements.txt`.
2. **Run the Client**: Execute the main script to start the RAG-client and interact with it via prompts.
   ```bash
   python python/main.py
   ```
3. **Processing Documents**: The client will parse and index documents from the specified directory and allow querying through the integrated language models.
4. **Code Reading**: Utilize the code reader tool to fetch and analyze code files.

## Resources

https://youtu.be/JLmI0GJuGlY?si=NsrlUHlY6GE0MVoZ  
https://www.llamaindex.ai/  
https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/  
https://medium.com/@m.adel.abdelhady/build-a-simple-rag-application-with-llama-index-ff2366afc7fb  
https://medium.com/@lars.chr.wiik/a-straightforward-guide-to-retrieval-augmented-generation-rag-0031bccece7f  
