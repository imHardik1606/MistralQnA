# diagnostic.py
import sys
sys.path.append('.')
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.rag import chunk_text

# Test with your actual function
text = "a" * 5000
chunks = chunk_text(text)

print(f"Config: CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")
print(f"Text length: {len(text)}")
print(f"Number of chunks: {len(chunks)}")
print(f"Chunk lengths: {[len(c) for c in chunks]}")

# Show the actual chunking logic
print("\nActual chunk boundaries:")
for i in range(min(3, len(chunks))):  # Show first 3 chunks
    start = i * (CHUNK_SIZE - CHUNK_OVERLAP)
    end = start + CHUNK_SIZE
    print(f"Chunk {i}: chars[{start}:{end}] = length {len(chunks[i])}")