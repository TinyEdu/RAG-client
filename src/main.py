import os  
import time  
import logging 
import fitz 
from docx import Document as DocxDocument 
from pydantic import BaseModel 
import ast  
import json

from llama_index.llms.ollama import Ollama  
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, PromptTemplate 
from llama_index.core.embeddings import resolve_embed_model  
from llama_index.core.tools import QueryEngineTool, ToolMetadata 
from llama_index.core.agent import ReActAgent
from llama_index.core.query_pipeline import QueryPipeline 
from llama_index.core.output_parsers import PydanticOutputParser 

from prompts import context, code_parser_template

# Configure logging to capture important events and errors
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

                    document = Document(text=content, metadata={"source": file_path})
                    documents.append(document)
                except Exception as e:
                    logging.error(f"Failed to process file {file_path}: {e}")
                    continue
        return documents

# Adding an additional resource from the Komputronik Biznes website
def add_web_resource():
    web_content = "Komputronik Biznes is an IT solutions integrator offering comprehensive services in consulting, hardware, software, and IT infrastructure modernization."
    documents.append(Document(text=web_content, metadata={"source": "https://www.komputronikbiznes.pl/"}))

# Setup LLM model with a specific configuration
model = Ollama(model="llama3.1", request_timeout=30.0)

# Load documents from a directory, using the custom document loader
documents = CustomSimpleDirectoryReader("/home/nikodem-ub1/github/RAG-client/data").load_data()

# Add web resource content
add_web_resource()

# Resolve the embedding model using the specified model name
embed_model = resolve_embed_model("local:BAAI/bge-m3")

# Create a vector index from the loaded documents
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Convert the vector index into a query engine that can interface with the LLM
query_engine = vector_index.as_query_engine(llm=model)

# Setup tools
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This provides documentation about a project called DEGA. Use this for reading documentation."
        ),
    )
]

# Initialize the ReActAgent using the tools
agent = ReActAgent.from_tools(tools, llm=model, verbose=True, context=context)

# Pydantic output formatter
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

# Initialize the Pydantic output parser with the defined output structure
parser = PydanticOutputParser(CodeOutput)

# Format the code parser template with the parser
json_prompt_str = parser.format(code_parser_template)

# Create a PromptTemplate object using the formatted string
json_prompt_tmpl = PromptTemplate(json_prompt_str)

# Setup a query pipeline with a chain of templates and models
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, model])

# Main loop to interact with the user
def main():
    while (prompt := input("Enter a prompt (q to quit) -> ")) != "q":
        start_time = time.time()

        retries = 0
        clean_json = None  # Initialize outside the loop

        # Check if the prompt is likely asking for code generation
        is_code_related = any(keyword in prompt.lower() for keyword in ["code", "script", "function", "class", "method"])

        for i in range(3):  # Iterate 3 times to simulate a thought process
            try:
                logging.info(f"Iteration {i+1}/3: Querying the agent.")
                
                if is_code_related:
                    result = agent.query(prompt)
                    next_result = output_pipeline.run(response=result)
                else:
                    # Directly use the agent for non-code related queries
                    result = agent.query(prompt)
                    next_result = result

                # Print raw output for inspection
                print(f"Raw output (Iteration {i+1}/3): {next_result}")

                # Clean and parse the JSON output if code-related
                if is_code_related:
                    json_str = str(next_result).replace("assistant:", "").strip()
                    clean_json = json.loads(json_str)
                else:
                    clean_json = {
                        "code": "",
                        "description": str(next_result),
                        "filename": "response.txt"
                    }

                logging.info(f"Iteration {i+1}/3 complete.")
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                retries += 1
                logging.error(f"Parsing error: {e}. Retrying... ({retries}/3)")
            except Exception as e:
                retries += 1
                logging.error(f"Unexpected error: {e}. Retrying... ({retries}/3)")

            if clean_json:
                break  # Exit loop if parsing was successful

        if retries == 3:
            logging.error("Failed to get a response after 3 retries. Please try again.")
            continue

        print("Final Output:")
        print(clean_json["code"] if clean_json["code"] else clean_json["description"])
        print("\n\nDescription:", clean_json["description"])

        filename = clean_json["filename"]

        try:
            with open(filename, "w") as f:
                f.write(clean_json["code"] if clean_json["code"] else clean_json["description"])
            logging.info(f"Output saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving output to file: {e}")

        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time} seconds")

# Entry point of the script
if __name__ == "__main__":
    main()
