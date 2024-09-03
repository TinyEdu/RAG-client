import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from typing import List

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()