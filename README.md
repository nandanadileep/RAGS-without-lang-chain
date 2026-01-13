# RAGS-without-lang-chain

**Live Demo:** [https://genesiss.streamlit.app/](https://genesiss.streamlit.app/)

Genesis is a Retrieval-Augmented Generation (RAG) application that provides biblical guidance. It demonstrates a custom RAG pipeline using Python, Streamlit, and Google's Gemini 2.0 Flash model, implemented without reliance on orchestration frameworks like LangChain.

## Technical Overview

The system indexes the King James Version (KJV) Bible to provide context-aware responses.

* **Vector Search**: Uses `sentence-transformers` for embeddings and NumPy for cosine similarity.
* **Generation**: Retrieval context is passed to Google's Gemini 2.0 Flash model to generate solemn, wise responses.
* **Data Pipeline**: Custom scripts fetch raw JSON text, process verses, and manage local asset storage.

## Tech Stack

* **Frontend**: Streamlit
* **LLM**: Google Gemini 2.0 Flash
* **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Operations**: NumPy

## Local Setup

**1. Clone the repository**

```bash
git clone <repository-url>
cd <repository-folder>
