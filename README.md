# ⚖️ Judicial RAG System  
### A Persian Legal Question-Answering Assistant Based on Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Academic%20Project-green)
![Architecture](https://img.shields.io/badge/Architecture-RAG-orange)

---

## 📖 Project Overview

This project implements an intelligent Persian legal assistant based on a **Retrieval-Augmented Generation (RAG)** architecture.

The system retrieves relevant judicial decisions and generates grounded answers using Large Language Models (LLMs). The primary goal is to evaluate different RAG configurations in a legal domain setting and analyze their impact on retrieval quality, generation performance, and system latency.

---

## 🎯 Objectives

- Build a legal-domain question answering system in Persian
- Compare different embedding models
- Evaluate the impact of reranking
- Compare different LLMs
- Measure trade-offs between accuracy and latency
- Analyze retrieval vs generation performance

---

## 🏗️ System Architecture

The system follows a standard RAG pipeline:

### 1️⃣ Data Processing
- Crawling Persian judicial decisions
- Cleaning and preprocessing legal texts
- Extracting metadata (laws, legal articles, case numbers, references)

### 2️⃣ Embedding & Indexing
- Vector embedding models:
  - E5-base
  - ParsBERT
- Vector database indexing

### 3️⃣ Retrieval
- Top-k semantic retrieval
- Optional cross-encoder reranking

### 4️⃣ Answer Generation
- Prompt-based answer generation
- Context grounding from retrieved documents
- Citation-aware responses

### 5️⃣ Evaluation
- Retrieval metrics
- Generation metrics
- Latency measurement

---

## 🧪 Experimental Setup

The system was evaluated across **8 experimental configurations**, varying:

| Component | Variations |
|-----------|------------|
| Embedding Model | E5 / ParsBERT |
| Reranker | Enabled / Disabled |
| Language Model | GPT4oMini / qwen2.5-3b |

Evaluation was conducted on **20 manually designed legal questions** with gold-standard answers.

---

## 📊 Evaluation Metrics

### 🔎 Retrieval Metrics
- Recall@5
- Recall@10
- MRR (Mean Reciprocal Rank)
- NDCG@10

### ✍️ Generation Metrics
- Exact Match (EM)
- F1 Score
- ROUGE-1
- ROUGE-L

### ⚡ Efficiency Metric
- Average Latency

---

## 📈 Key Findings

- Reranking significantly improves retrieval metrics (Recall, MRR, NDCG).
- Improved retrieval does not always guarantee better final answer quality.
- GPT4oMini shows more stable generation performance than qwen2.5-3b.
- There is a clear trade-off between retrieval accuracy and latency.
- Exact Match remained zero due to paraphrased responses, highlighting the limitations of strict lexical matching.

---

## 📁 Project Structure

```
judicial-rag-system/
│
├── data/                  # Processed judicial documents
├── preprocessing/         # Text cleaning and metadata extraction
├── embeddings/            # Embedding generation modules
├── retrieval/             # Retrieval + reranker modules
├── generation/            # Prompting & LLM interaction
├── evaluation/            # Metrics and evaluation scripts
├── experiments/           # Experimental configurations
├── results/               # Evaluation outputs and logs
└── main.py                # Entry point
```

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/mahdisdg/judicial-rag-system.git
cd judicial-rag-system
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the pipeline

```bash
python main.py
```

---

## 🔮 Future Improvements

- Graph-RAG implementation
- Domain-specific fine-tuned embeddings
- Hybrid retrieval (Vector + BM25)
- Legal knowledge graph construction
- Human evaluation layer
- Query rewriting fine-tuning
- Citation verification module

---

## 👥 Authors

- Mohammadmehdi Sadeghi  
- Amirhossein KargarFard  

---

## 📌 Repository

GitHub Repository:  
https://github.com/mahdisdg/judicial-rag-system

---

## 📜 License

This project was developed for academic and research purposes.
