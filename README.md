# SEBI Regulatory RAG Chatbot

🔗 **Live Demo:** https://sebi-chatbot.streamlit.app/

## Overview

The **SEBI Regulatory RAG Chatbot** is a Retrieval-Augmented Generation (RAG) application that enables users to ask questions about SEBI regulations and circulars in natural language. Instead of manually searching through lengthy regulatory documents, the chatbot retrieves the most relevant sections and generates context-aware responses using a Large Language Model (LLM).

This project demonstrates the practical application of semantic search, vector databases, and LLMs for document question answering.

---

## Features

- 📄 Upload and process SEBI regulatory PDF documents
- 💬 Ask questions in natural language
- 🔍 Semantic search using vector embeddings
- 🤖 Context-aware responses powered by a Hugging Face LLM
- ⚡ Fast document retrieval using ChromaDB
- 🌐 Interactive web interface built with Streamlit

---

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Hugging Face Inference API
- PyPDFLoader
- RecursiveCharacterTextSplitter
- python-dotenv

---

## Project Workflow

```text
                PDF Documents
                      │
                      ▼
              PyPDFLoader
                      │
                      ▼
      RecursiveCharacterTextSplitter
                      │
                      ▼
        Sentence Transformer Embeddings
                      │
                      ▼
                 ChromaDB
                      │
                      ▼
             User Question
                      │
                      ▼
           Similarity Search
                      │
                      ▼
       Retrieved Relevant Chunks
                      │
                      ▼
          Hugging Face LLM
                      │
                      ▼
            Final AI Response
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Devanshimann/RAG.git
cd RAG
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
HF_API_KEY=your_huggingface_api_key
```

### Run the application

```bash
streamlit run rag_app.py
```

---

## Example Questions

- What is ASBA?
- What are the timelines for a Rights Issue?
- What are the disclosure requirements for an IPO?
- What compensation is available to Retail Individual Investors?
- What are the eligibility criteria for the Innovators Growth Platform?

---

## Project Structure

```text
RAG/
│── rag_app.py
│── embedding.py
│── vectorstore.py
│── retriever.py
│── rag.py
│── requirements.txt
│── .gitignore
│── README.md
│── data/
└── vectorstore/      # Generated automatically (ignored in Git)
```

---

## Live Demo

Try the application here:

**https://sebi-chatbot.streamlit.app/**

B.Tech Data Science Student | Machine Learning & AI Enthusiast

If you found this project useful, consider giving it a ⭐ on GitHub!
