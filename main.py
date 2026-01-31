import os
import streamlit as st
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
from mistralai import Mistral
import faiss

# -----------------------
# Setup
# -----------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("Set MISTRAL_API_KEY in .env")

client = Mistral(api_key=MISTRAL_API_KEY)

CHAT_MODEL = "mistral-small-latest"
EMBED_MODEL = "mistral-embed"

# -----------------------
# Helper functions
# -----------------------

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_texts(texts):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        inputs=texts
    )
    return np.array([e.embedding for e in response.data]).astype("float32")


def build_vector_store(chunks):
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve_chunks(question, chunks, index, k=4):
    q_embedding = embed_texts([question])
    distances, indices = index.search(q_embedding, k)
    return [chunks[i] for i in indices[0]]


def ask_mistral(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Mistral Document Assistant", layout="wide")

st.title("📄 Mistral AI Document Assistant")
st.write("Upload a document and chat with it using Mistral AI.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading document..."):
        text = read_pdf(uploaded_file)
        chunks = chunk_text(text)

    with st.spinner("Creating embeddings..."):
        index, _ = build_vector_store(chunks)

    st.success("Document processed! You can now ask questions.")

    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Thinking..."):
            relevant_chunks = retrieve_chunks(question, chunks, index)
            answer = ask_mistral(question, relevant_chunks)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved context"):
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {i+1}**")
                st.write(chunk)
