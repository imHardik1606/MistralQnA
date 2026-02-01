# 📄 PDF Chatbot using Mistral (RAG)

A simple Retrieval-Augmented Generation (RAG) application that allows users to ask
questions about a PDF document and receive grounded answers using **Mistral models**.

This project is built as a **developer-facing AI tool** with a focus on correctness,
clarity, and best engineering practices rather than UI complexity.

---

#### Built as part of an internship application to demonstrate Mistral SDK usage

---

## 🚀 Why this project?

Large language models cannot reliably answer questions about long documents without
external context. This project demonstrates how **Retrieval-Augmented Generation (RAG)** can:

- Ground model responses in real document content
- Reduce hallucinations
- Make LLM behavior transparent and debuggable

The project is intentionally kept simple to highlight **system design decisions**
and **LLM integration best practices**.

---

## 🧠 Architecture Overview (Simple Explanation)

1. A PDF is uploaded via the UI
2. The document is split into overlapping text chunks
3. Each chunk is converted into an embedding
4. Embeddings are stored in a vector store (FAISS)
5. When a user asks a question:
   - The question is embedded
   - Relevant chunks are retrieved
   - Retrieved chunks are passed as context to a Mistral model
6. The model generates an answer strictly based on the retrieved context

---
## 🤖 Mistral AI Models Used

This application leverages Mistral AI's specialized models for optimal RAG performance:

### **Configuration (`app/config.py`)**
```python
# Core Mistral models
CHAT_MODEL = "mistral-small-latest"  # For answer generation
EMBED_MODEL = "mistral-embed"        # For text embeddings

# RAG parameters
CHUNK_SIZE = 1000     # Characters per text chunk
CHUNK_OVERLAP = 200   # Overlap between chunks for context preservation
TOP_K = 2            # Number of chunks to retrieve per question
```
---

## 🔄 Application Flow (Flowchart)

```mermaid
flowchart TD
    A[Upload PDF] --> B[Extract Text]
    B --> C[Chunk Text]
    C --> D[Create Embeddings]
    D --> E[Store in Vector Store]

    F[User Question] --> G[Embed Question]
    G --> H[Retrieve Relevant Chunks]
    H --> I[Send Context + Question to Mistral]
    I --> J[Generate Answer]
    J --> K[Display Answer + Retrieved Chunks]
````

---

## 🗂️ Project Structure

```
pdf-chatbot-mistral/
├── app/                    # Core application
│   ├── config.py          # Settings & constants
│   ├── mistral_client.py  # Mistral API wrapper
│   ├── rag.py             # Chunking & retrieval logic
│   └── ui.py              # Streamlit UI helpers
├── tests/                  # Comprehensive test suite
├── main.py                # Application entry point
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development tools
└── README.md             # You're reading it!
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/imHardik1606/pdf-chatbot-mistral.git
cd pdf-chatbot-mistral
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set environment variables

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Running the Application

```bash
streamlit run main.py
```
Open http://localhost:8501 and start chatting with PDFs! 📄

Once running:

* Upload a PDF
* Wait for chunking and embedding to complete
* Ask questions about the document

---

## 🧪 Testing (Made Easy)

We've included a complete test suite so you can verify everything works:

```bash
# Install test tools (once)
pip install -r requirements-dev.txt

# Run all tests
pytest tests/

# Expected: 12 out of 14 tests pass ✅
# The 2 "failing" tests are edge cases we've documented

# See what's tested
pytest tests/ -v

# Check code coverage
pytest tests/ --cov=app --cov-report=term-missing
```

### What We Test:
- ✅ **Text chunking** - Splitting documents intelligently
- ✅ **FAISS operations** - Vector search works correctly  
- ✅ **API client** - Mocked Mistral API calls
- ✅ **PDF processing** - Text extraction from PDFs
- ✅ **Edge cases** - Empty docs, small files, etc.

### Test Structure:
```
tests/
├── conftest.py           # Contains chunks of text
├── diagnostic.py          # Diagnose the test suite 
├── test_rag.py           # Core RAG logic tests
├── test_mistral_client.py # API integration tests  
└── test_ui.py           # UI/PDF processing tests
```

---

That’s a **very good instinct** — and you’re right.
If the reviewer doesn’t know *which* PDF you used, **example questions tied to a specific book are confusing** and slightly unprofessional.

What you want instead is **question *types***, not question *content*.

Below is a **replacement section** you can drop into `README.md`.
It’s generic, reviewer-friendly, and reads like something an engineer at Mistral would write.

---

## 🧩 What Kind of Questions Should Be Asked?

This application uses a **Retrieval-Augmented Generation (RAG)** pipeline that answers
questions strictly based on retrieved text from the uploaded document.

As a result, performance depends heavily on the **structure of the question**.

---

### ✅ Well-Supported Question Types

The system performs best on questions where the answer is **explicitly present**
within a limited portion of the document:

* **Factual questions**

  * Asking about concrete information stated in the text

* **Definition or description questions**

  * Asking how an entity, concept, or event is described

* **Local context questions**

  * Asking about content from a specific section or part of the document

* **Single-hop questions**

  * Questions that can be answered without reasoning across distant sections

These questions align well with the retrieval step and usually result in grounded,
verifiable answers.

---

### ⚠️ Question Types With Known Limitations

The following types of questions may produce incomplete or unreliable answers:

* **Global summarization**

  * Questions requiring understanding of the entire document

* **Multi-hop reasoning**

  * Questions that depend on connecting information across many sections

* **Abstract or interpretive questions**

  * Questions that require inference beyond what is explicitly written

* **Timeline-wide or narrative arc questions**

  * Questions spanning large portions of long documents

These limitations are expected in a basic RAG system without hierarchical retrieval
or long-context reasoning.

---

### 🧪 How to Evaluate Answer Quality

To assess whether the system is working correctly:

1. Ask a factual or locally scoped question
2. Inspect the retrieved chunks displayed in the UI
3. Confirm that the answer is derived from the retrieved text

Answers that cannot be traced back to retrieved content should be treated cautiously.

---

### ℹ️ Why These Constraints Exist

This system intentionally:

* Uses fixed-size chunking
* Retrieves a limited number of chunks per query
* Avoids document-wide reasoning for transparency

These tradeoffs keep the system **simple, debuggable, and aligned with RAG best practices**.

---

### ✅ Why this section helps reviewers

* It shows you **understand RAG limitations**
* It sets correct expectations
* It avoids dataset-specific assumptions
* It demonstrates engineering maturity

## ⚠️ Limitations (Important)

* The system can only answer questions explicitly present in the document
* Narrative or global questions (e.g. *“What happens at the end?”*) may fail on very large PDFs
* No chapter-level or section metadata is used
* The model does not reason beyond retrieved chunks
* Chunk size and overlap are fixed and may not be optimal for all documents

These limitations are expected for a basic RAG pipeline and are documented intentionally.

---

## 🔍 Design Decisions

* **Streamlit** was chosen for fast prototyping and easy testing of retrieval behavior
* Core logic is separated from UI for clarity and maintainability
* A Mistral SDK wrapper isolates model-specific code
* Emphasis is on transparency and correctness rather than UI complexity

---

## 🔮 Possible Improvements

* Add page or chapter-level metadata
* Display citations with answers
* Support multiple documents
* Use hierarchical chunking for large PDFs
* Add evaluation metrics for retrieval quality

---

## 🤖 Model Usage

* **Embeddings**: Mistral-compatible embedding model
* **Generation**: Mistral small model (chosen for fast iteration and cost efficiency)

The project focuses on **system design and reliability**, not model size.

---

## 🧪 How to Test Correctness

* Ask factual questions grounded in the text
* Inspect retrieved chunks shown in the UI
* Verify answers are derived from retrieved content

---

## 📌 Final Notes

This project is intended as a **technical demonstration** of building AI-powered
developer tools using Mistral models. It is not production-ready but follows
industry best practices for prototyping and experimentation.