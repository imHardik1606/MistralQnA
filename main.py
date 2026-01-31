from app.ui import render, read_pdf, show_sources
from app.config import MISTRAL_API_KEY
from app.mistral_client import MistralClient
from app.rag import chunk_text, build_index, retrieve

import streamlit as st

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set")

client = MistralClient(MISTRAL_API_KEY)

uploaded, question = render()

if uploaded:
    text = read_pdf(uploaded)
    chunks = chunk_text(text)

    embeddings = client.embed(chunks)
    index = build_index(embeddings)

    if question:
        q_embedding = client.embed([question])[0]
        context = retrieve(q_embedding, index, chunks)

        prompt = f"""
Answer the question using only the context below.
If the answer is not present, say you don't know.

Context:
{chr(10).join(context)}

Question:
{question}
"""

        context = retrieve(q_embedding, index, chunks)
        answer = client.chat(prompt)
        st.subheader("Answer")
        st.write(answer)

        show_sources(context)
