# ⚖️ Judicial RAG System

A Persian legal assistant based on Retrieval-Augmented Generation (RAG).

---

## 📖 Overview

Judicial RAG System is an intelligent question-answering system designed for Persian legal documents.

The system retrieves relevant judicial decisions using semantic vector search and generates grounded answers using Large Language Models (LLMs). It integrates document preprocessing, vector indexing, retrieval, reranking, and LLM-based generation into a unified pipeline.

Its goal is to provide structured, context-aware, and legally grounded responses to user queries.

---

## 🔎 What Happens When You Ask a Question?

When a user submits a legal question, the system performs the following steps:

1. **Query Processing**  
   The user’s question is cleaned and prepared for embedding.

2. **Query Embedding**  
   The question is converted into a vector representation using the selected embedding model.

3. **Semantic Retrieval**  
   The system searches the vector database (Qdrant) to retrieve the most relevant legal documents based on similarity.

4. **Optional Reranking**  
   If enabled, a reranker reorders the retrieved documents to improve relevance.

5. **Context Construction**  
   The top retrieved documents are combined into a structured context block.

6. **LLM Generation**  
   The context and the original question are sent to the language model, which generates a grounded legal answer.

7. **Response Delivery**  
   The final answer is returned to the user through the interface.

This pipeline ensures that responses are grounded in actual legal documents rather than being purely generative.

---

## 📁 Project Structure

```
judicial-rag-system/
│
├── .streamlit/               # Streamlit configuration files
├── config/                   # System configuration settings
├── data/                     # Raw and processed legal documents
├── DBs/                      # Database files and local storage
├── experiments/              # Experimental configurations
├── indexing/                 # Document indexing logic
├── logs/                     # System logs and execution outputs
├── preprocess/               # Text cleaning and preprocessing modules
├── qdrant_db_multilingual/   # Qdrant vector database storage
├── rag_llm/                  # RAG pipeline and LLM integration
├── retrieval/                # Retrieval and reranking modules
├── scraper/                  # Web scraping and data collection scripts
├── ui/                       # User interface components
├── .env                      # Environment variables
└── .gitignore                # Git ignored files
```

---

## 👥 Authors

- MohammadMahdi Sadeghi  
- Amirhossein KargarFard  

---

MCI Generative AI Bootcamp – 2026
