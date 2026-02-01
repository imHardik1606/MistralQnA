import streamlit as st
from pypdf import PdfReader

def read_pdf(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)


def render():
    st.title("📄 Mistral Document Assistant")
    st.caption("Upload a document and ask grounded questions using Mistral AI")

    uploaded = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF to chat with its content"
    )

    return uploaded   

def show_sources(chunks):
    with st.expander("📚 Retrieved context (sources)"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i + 1}**")
            st.write(chunk)
            st.divider()

def show_status(message):
    return st.status(message, expanded=True)
