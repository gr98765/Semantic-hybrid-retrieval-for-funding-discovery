# Semantic-hybrid-retrieval-for-funding-discovery

🧠 Semantic Hybrid Retrieval System for Academic Funding Discovery

This project builds an intelligent retrieval system that helps researchers discover relevant academic funding opportunities (e.g., NSF grants) based on their research ideas.
By combining keyword-based retrieval (BM25) and semantic retrieval (Sentence-BERT + FAISS), it creates a hybrid Retrieval-Augmented Generation (RAG) pipeline for smarter and explainable funding discovery.

🧩 Project Overview

Finding funding opportunities can be tedious when relying on keyword search.
This system uses semantic embeddings and vector databases to go beyond simple word matching — it understands meaning and context.
By integrating a Large Language Model (LLM), it provides human-readable explanations for why certain grants match a given query.

🎯 Objectives

Develop a hybrid grant retrieval system combining sparse and dense retrieval.

Improve accuracy over traditional BM25 keyword search.

Integrate RAG and LLMs for contextual understanding and interpretability.

Evaluate retrieval performance using standard IR metrics (Precision, nDCG, MRR).

⚙️ Technical Workflow
1. Data Preparation

Source: NSF Award Abstracts Dataset (Kaggle)

Tasks:

Load and clean abstracts (remove formatting, symbols, etc.)

Standardize NSF program categories (BIO, CNS, IIS, OTHER)

Store processed text for downstream retrieval

2. Baseline Sparse Retrieval (BM25)

Implement TF-IDF/BM25 to retrieve top-k grants for sample researcher queries.

Evaluate ranking with metrics:

Precision@5

nDCG@5

MRR@5

Acts as a benchmark for semantic retrieval improvements.

3. Dense Semantic Retrieval

Use Sentence-BERT (SBERT) to embed both abstracts and queries into vector space.

Store embeddings in a FAISS vector database for efficient similarity search.

Compare semantic retrieval performance against BM25 baseline.

4. Hybrid RAG Pipeline

Combine semantic retrieval results with an LLM (like GPT) for contextual reasoning.

The LLM:

Generates short explanations for why a grant is relevant.

Helps align user’s research proposal ideas with grant descriptions.

5. Evaluation Framework

Quantitative: Precision@k, nDCG, MRR.

Qualitative: Faithfulness and relevance of LLM-generated explanations.

Comparison: BM25 vs SBERT vs Hybrid retrieval results.

🧪 Workflow Summary
Dataset (NSF Abstracts)
        ↓
Data Cleaning & Preprocessing
        ↓
BM25 Baseline Retrieval
        ↓
Evaluate Baseline (Precision, nDCG, MRR)
        ↓
SBERT Embeddings + FAISS Index
        ↓
Semantic Retrieval (Top-k Results)
        ↓
RAG Integration (LLM Generates Explanations)
        ↓
Hybrid Evaluation (Quantitative + Qualitative)

📂 Folder Structure
semantic-retrieval-funding/
│
├── data/
│   ├── raw/                # Original NSF dataset from Kaggle
│   └── processed/          # Cleaned and categorized abstracts
│
├── notebooks/
│   └── INFO556_Project_Update.ipynb  # Main Google Colab notebook
│
├── src/
│   ├── bm25_baseline.py            # Keyword retrieval
│   ├── semantic_retrieval.py       # SBERT + FAISS retrieval
│   ├── rag_pipeline.py             # LLM integration for explanation
│   ├── evaluation.py               # Precision, nDCG, MRR computations
│   └── utils.py                    # Helper functions (cleaning, etc.)
│
├── results/
│   ├── bm25_metrics.csv
│   ├── sbert_metrics.csv
│   └── visualizations/             # Graphs and comparison charts
│
├── requirements.txt
├── README.md
└── LICENSE


