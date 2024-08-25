from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.query_pipeline import QueryPipeline
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for PDF parsing
from docx import Document as DocxDocument  # python-docx for DOCX parsing

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
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Custom document loader to handle both PDF and DOCX files
class CustomSimpleDirectoryReader(SimpleDirectoryReader):
    def __init__(self, input_dir, file_extractor=None):
        super().__init__(input_dir)
        self.file_extractor = file_extractor if file_extractor else {}

    def load_data(self):
        documents = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.pdf'):
                    content = parse_pdf(file_path)
                elif file.endswith('.docx'):
                    content = parse_docx(file_path)
                else:
                    continue
                # Create Document object
                document = Document(text=content, metadata={"file_path": file_path})
                documents.append(document)
        return documents

model = Ollama(model="llama3.1", request_timeout=30.0)

result = model.complete("The quick brown fox jumps over the lazy dog")

documents = CustomSimpleDirectoryReader("data").load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=model)

result = query_engine.query("opis gry konfiguracja bohaterów w projekcie dega")

print(result)

ools = [
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="api_documentation",
        description="this gives documentation about a project called DEGA, Use this for reading documentation"
        ),
    )
]

code_llm = Ollama(model="codellama")
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context="")