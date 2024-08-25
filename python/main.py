import os
import time
import logging
import fitz  # PyMuPDF for PDF parsing
from docx import Document as DocxDocument  # python-docx for DOCX parsing
from pydantic import BaseModel
import ast

from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.output_parsers import PydanticOutputParser

from code_reader import code_reader
from prompts import context, code_parser_template

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to parse PDF files using PyMuPDF
def parse_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to parse DOCX files using python-docx
def parse_docx(file_path):
    doc = DocxDocument(file_path)
    text = "\n".join(para.text for para in doc.paragraphs)
    return text

# Custom document loader to handle both PDF and DOCX files
class CustomSimpleDirectoryReader(SimpleDirectoryReader):
    def __init__(self, input_dir, file_extractor=None):
        super().__init__(input_dir)
        self.file_extractor = file_extractor or {}

    def load_data(self):
        documents = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith('.pdf'):
                        content = parse_pdf(file_path)
                    elif file.endswith('.docx'):
                        content = parse_docx(file_path)
                    else:
                        logging.warning(f"Unsupported file type: {file}")
                        continue

                    # Create Document object
                    document = Document(text=content, metadata={"source": file_path})
                    documents.append(document)
                except Exception as e:
                    logging.error(f"Failed to process file {file_path}: {e}")
                    continue
        return documents

# Setup LLM model
model = Ollama(model="llama3.1", request_timeout=30.0)
documents = CustomSimpleDirectoryReader("/home/nikodem-ub1/github/RAG-client/data").load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=model)

# Setup tools and agents
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This provides documentation about a project called DEGA. Use this for reading documentation."
        ),
    ),
    code_reader
]

code_llm = Ollama(model="codellama")
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

# Pydantic output formatter
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, model])

def main():
    while (prompt := input("Enter a prompt (q to quit) -> ")) != "q":
        start_time = time.time()

        retries = 0
        while retries < 3:
            try:
                result = agent.query(prompt)
                next_result = output_pipeline.run(response=result)

                # Safely parse JSON output
                clean_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
                break
            except (ValueError, SyntaxError) as e:
                retries += 1
                logging.error(f"Parsing error: {e}. Retrying... ({retries}/3)")
            except Exception as e:
                retries += 1
                logging.error(f"Unexpected error: {e}. Retrying... ({retries}/3)")

        if retries == 3:
            logging.error("Failed to get a response after 3 retries. Please try again.")
            continue

        print("Code Output:")
        print(clean_json["code"])
        print("\n\nDescription:", clean_json["description"])

        filename = clean_json["filename"]

        try:
            with open(filename, "w") as f:
                f.write(clean_json["code"])
            logging.info(f"Code saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving code to file: {e}")

        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
