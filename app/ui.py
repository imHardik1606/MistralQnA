import streamlit as st
from pypdf import PdfReader

def read_pdf(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)


def render():
    st.title("📄 Mistral Document Assistant")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    question = st.text_input("Ask a question")

    return uploaded, question

def show_sources(chunks):
    with st.expander("📚 Retrieved context (sources)"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i + 1}**")
            st.write(chunk)
            st.divider()
