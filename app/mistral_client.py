from mistralai import Mistral
from app.config import CHAT_MODEL, EMBED_MODEL

class MistralClient:
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)

    def embed(self, texts):
        response = self.client.embeddings.create(
            model=EMBED_MODEL,
            inputs=texts
        )
        return [e.embedding for e in response.data]

    def chat(self, prompt):
        response = self.client.chat.complete(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content
