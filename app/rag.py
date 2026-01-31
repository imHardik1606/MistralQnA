import numpy as np
import faiss
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


def retrieve(question_embedding, index, chunks):
    distances, indices = index.search(
        np.array([question_embedding]).astype("float32"),
        TOP_K
    )
    return [chunks[i] for i in indices[0]]
