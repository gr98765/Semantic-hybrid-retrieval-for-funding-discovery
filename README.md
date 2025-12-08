# 🧠 Semantic Hybrid Retrieval System for Academic Funding Discovery

This project builds an intelligent retrieval system that helps researchers discover relevant NSF funding opportunities based on their research ideas.  
It combines **keyword-based retrieval (BM25)** with **semantic retrieval (SBERT + FAISS)** and adds a **Large Language Model (LLM) for explainability**.

This hybrid approach produces more meaningful search results and provides natural-language explanations for *why* the retrieved grants are relevant.

---

## 🔍 Why This Project?

Finding research funding often depends on exact keyword matches, which miss many relevant opportunities.  
This system goes **beyond keyword matching** by understanding:

- Semantic meaning  
- Research context  
- Topic relevance  

The LLM then explains *why* a result is relevant — solving a real problem researchers face.

---

# 🎯 Objectives

- Build a **hybrid semantic search engine** for NSF grants  
- Improve over BM25 baseline using embeddings  
- Add **explainable retrieval** using an LLM  
- Evaluate retrieval performance using:
  - Precision@5  
  - nDCG@5  
  - MRR  
  - Human vs LLM agreement  

---

# ⚙️ Technical Workflow

### **1. Data Preparation**
- Source: *NSF Award Abstracts Dataset (Kaggle)*  
- Tasks:
  - Clean abstracts  
  - Remove duplicates / missing text  
  - Standardize NSF program categories → **BIO, IIS, CNS, OTHER**  
- Output saved as: `nsf_grants_clean.csv` (clean processed dataset used by the app)

---

### **2. BM25 Baseline Retrieval**
- Tokenize abstracts  
- Build BM25 index  
- Retrieve top-k grants  
- Compute baseline IR metrics:
  - Precision@5  
  - MRR  
  - nDCG  

---

### **3. Semantic Retrieval (SBERT + FAISS)**
- Encode abstracts using **Sentence-BERT (all-MiniLM-L6-v2)**  
- Store vectors in FAISS index  
- Perform dense retrieval  
- Compare to BM25 performance  

---

### **4. Hybrid Retrieval**
BM25 + SBERT combined:


This improves both **recall** and **semantic matching**, especially when query wording differs from the grant abstract.

---

### **5. LLM Explainability (RAG Pipeline)**
For each retrieved grant, the LLM:

- Assigns a binary relevance label  
- Generates a short explanation  
- Justifies its label  

This adds **interpretability**, which is essential for researchers.

---

### **6. Evaluation Framework**
Includes both **quantitative** and **qualitative** evaluation:

#### Quantitative
- Precision@5  
- MRR  
- nDCG  
- Human vs LLM agreement score  

#### Qualitative
- LLM-generated natural-language explanations  
- Faithfulness and clarity  

---

# 📁 Folder Structure



