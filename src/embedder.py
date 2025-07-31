from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)
