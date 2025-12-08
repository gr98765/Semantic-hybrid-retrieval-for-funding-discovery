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

```plaintext
Semantic-hybrid-retrieval-for-funding-discovery/
│
├── Data/
│   └── nsf_grants_clean.csv        # Clean processed dataset (small + Git-safe)
│
├── app.py                          # Streamlit UI for search + evaluation
├── retrieval_core.py               # Hybrid retrieval + evaluation logic
│
├── notebook/
│   └── evaluation.ipynb      # Full evaluation notebook (BM25, SBERT, metrics)
│
├── requirements.txt                # Python dependencies
├── README.md                       
└── .gitignore  


---

# Why Two Code Components?

### ✔️ `retrieval_core.py`
**Purpose:**  
Contains all retrieval logic (backend) needed by the Streamlit UI.

Includes:
- BM25 setup  
- SBERT + FAISS  
- Hybrid ranking  
- LLM relevance + explanations  
- Evaluation metrics  

---

### ✔️ `app.py`
**Purpose:**  
Interactive Streamlit interface enabling (frontend):

- Query search  
- Real-time retrieval  
- LLM-generated explanations  
- Evaluation page (metrics + human labels + LLM labels)

This file contains only **UI logic**, with all computation done in `retrieval_core.py`.

---

### ✔️ `evaluation.ipynb` (Evaluation Notebook)
**Why this exists separately:**

The notebook documents the **full research workflow**:

- Data cleaning  
- Category mapping  
- BM25 baseline experiments  
- SBERT embedding generation  
- Hybrid ranking analysis  
- Metric comparison  

The app focuses on *deployment*, while the notebook focuses on *methodology and evaluation*.  
They are not duplicates — they serve different purposes.

---

# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
# -------------------------
# Mac / Linux
export OPENAI_API_KEY="your-key-here"

# Windows PowerShell
setx OPENAI_API_KEY "your-key-here"

# (Restart terminal after running setx)

# 3. Run the Streamlit app
streamlit run app.py

You will see:

- Search tab → semantic grant search  
- Evaluation tab → metrics + human/LLM labels  

---

# ✅ Summary

This project delivers:

- A hybrid semantic retrieval engine  
- Explainable funding recommendations  
- Evaluation pipeline with human + LLM labels  
- A clean, interactive UI for demonstration  

---



