
## Requirements
Python - 3.10.12

1. **Setup**
```
sudo apt install python3.10
python3 -m venv ai
source ai/bin/activate
pip3 install -r config/requirements.txt
```
2. **Run the Client**: Execute the main script to start the RAG-client and interact with it via prompts.
   ```
   streamlit run src/main.py
   ```
3. **Processing Documents**: The client will parse and index documents from the specified directory and allow querying through the integrated language models.
4. **Code Reading**: Utilize the code reader tool to fetch and analyze code files.

## Resources

https://youtu.be/JLmI0GJuGlY?si=NsrlUHlY6GE0MVoZ  
https://www.llamaindex.ai/  
https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/  
https://medium.com/@m.adel.abdelhady/build-a-simple-rag-application-with-llama-index-ff2366afc7fb  
https://medium.com/@lars.chr.wiik/a-straightforward-guide-to-retrieval-augmented-generation-rag-0031bccece7f  
