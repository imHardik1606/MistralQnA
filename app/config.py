import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

CHAT_MODEL = "mistral-small-latest"
EMBED_MODEL = "mistral-embed"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
