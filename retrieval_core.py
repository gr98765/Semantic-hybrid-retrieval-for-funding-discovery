"""
retrieval_core.py
-----------------
This module contains all backend logic for the NSF Hybrid Retrieval System:
- Load cleaned dataset
- Build BM25 index
- Build SBERT embeddings + FAISS index
- Implement hybrid ranking (BM25 + SBERT)
- LLM relevance classification
- LLM explanation generation
- Evaluation functions for UI
"""

import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import os

# ===========================
# 1.LOAD CLEANED DATA
# ===========================
df_eval = pd.read_csv("nsf_grants_clean.csv")

# ===========================
# 2. BM25 KEYWORD RETRIEVAL SETUP (Sparse Retrieval)
# ===========================
def tokenize(text):
    return text.lower().split()

corpus = df_eval["abstract"].tolist()
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# ===========================
# 3. SBERT + FAISS SEMANTIC RETRIEVAL (Dense Retrieval)
# ===========================

model = SentenceTransformer("all-MiniLM-L6-v2")
dense_embeddings = model.encode(df_eval["abstract"].tolist(), convert_to_numpy=True)
# Normalize vectors for cosine similarity
faiss.normalize_L2(dense_embeddings)

index = faiss.IndexFlatIP(dense_embeddings.shape[1])
index.add(dense_embeddings)

def get_dense_scores(query, top_k=200):
    """Return FAISS semantic retrieval scores and indices."""
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    return D[0], I[0]

# =================================
# 4. HYBRID RANKING (BM25 + SBERT)
# =================================

def hybrid_rank(query, top_k=5, alpha=0.5):
    """
    Combine BM25 and SBERT scores using weighted fusion.
    alpha = weight for BM25, (1-alpha) = weight for SBERT.
    """
    # BM25 scores
    q_tokens = tokenize(query)
    bm_vals = bm25.get_scores(q_tokens)

    
    # Semantic scores from FAISS
    D, I = get_dense_scores(query, top_k=200)
    candidates = I.tolist()

    # Normalize scores
    bm_norm = (bm_vals[candidates] - bm_vals[candidates].min()) / (bm_vals[candidates].max() - bm_vals[candidates].min() + 1e-9)
    dense_norm = (D - D.min()) / (D.max() - D.min() + 1e-9)

    # Weighted fusion
    hybrid = alpha * bm_norm + (1 - alpha) * dense_norm
    sorted_idx = np.argsort(hybrid)[::-1][:top_k]

    return [(candidates[i], hybrid[i]) for i in sorted_idx]

# =========================================================
# 5. HUMAN RELEVANCE LABELS (Manually Added for Evaluation)
# =========================================================

human_labels = {
    "bio_cancer_detection": [0,0,0,0,0],
    "bio_protein_interactions": [0,1,0,0,0],
    "cns_zero_day": [0,1,1,1,0],
    "iis_recommendation": [0,1,0,0,0]
}

# ===========================================
#  6. METRIC FUNCTIONS (Precision, MRR, nDCG)
# ===========================================

def precision_at_5(h):
    """Compute Precision@5 using human relevance labels."""
    return sum(h)/5

def mrr(h):
    
    """Return the reciprocal rank of the first relevant document."""
    for i, v in enumerate(h):
        if v == 1:
            return 1/(i+1)
    return 0

def ndcg(h):
    """Compute normalized Discounted Cumulative Gain."""
    import numpy as np
    ideal = sorted(h, reverse=True)
    def dcg(vals):
        return sum((2**v - 1) / np.log2(i+2) for i,v in enumerate(vals))
    return dcg(h)/dcg(ideal) if dcg(ideal)>0 else 0

# ==========================================
#  7. LLM RELEVANCE + EXPLANATION
# ==========================================

client = OpenAI(api_key==os.getenv("OPENAI_API_KEY"))

def llm_relevance_label(query, abstract):
    """
    Ask LLM for binary (0/1) relevance for consistency with evaluation.
    """
     
    prompt = f"""
Does this grant match the query? Respond only with 1 or 0.

Query: {query}
Abstract: {abstract[:1500]}
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5
    )
    text = res.choices[0].message.content.strip()
    return int(text[0]) if text[0] in ["0","1"] else 0

def explain_with_llm(query, title, abstract, relevance_label):
    """
    Explanation that is consistent with the LLM relevance label.
    If relevance_label = 1 → explain why relevant
    If relevance_label = 0 → explain why NOT relevant
    """
    prompt = f"""
The LLM has already labeled this grant as: {relevance_label}
(1 = relevant, 0 = irrelevant).

Write 2 sentences explaining WHY this label is correct.

Query: {query}
Title: {title}
Abstract: {abstract[:2000]}
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120
    )
    return res.choices[0].message.content

# =============================================
# 8. EVALUATION FUNCTION (Used in Streamlit UI)
# =============================================

def evaluate_query(query, expected_cat, query_key):
    """
    Retrieve top-5 docs, compute all metrics, LLM relevance, and explanations.
    Used ONLY for UI evaluation tab.
    """
    retrieved = [idx for idx,_ in hybrid_rank(query)]
    retrieved_cats = df_eval.iloc[retrieved]["category"].tolist()

    human = human_labels[query_key]

    llm_labels = []
    llm_explanations = []

    for idx in retrieved:
        abstract = df_eval.iloc[idx]["abstract"]
        title = df_eval.iloc[idx]["award_title"]

        llm_label = llm_relevance_label(query, abstract)
        llm_labels.append(llm_label)

        explanation = explain_with_llm(query, title, abstract, llm_label)
        llm_explanations.append(explanation)

    metrics = {
        "precision": precision_at_5(human),
        "mrr": mrr(human),
        "ndcg": ndcg(human)
    }

    return retrieved, retrieved_cats, human, llm_labels, llm_explanations, metrics

# =================================
# 9. RETRIEVAL FOR UI (Search Tab)
# =================================

def search_and_explain(query, top_k=5, alpha=0.5):
    """
    Main retrieval function for Streamlit UI
    Each result contains: title, category, abstract, llm_label, explanation.
    """
    ranked = hybrid_rank(query, top_k=top_k, alpha=alpha)
    results = []

    for idx, score in ranked:
        row = df_eval.iloc[idx]

        # compute relevance label from LLM
        llm_label = llm_relevance_label(query, row["abstract"])

        # explanation consistent with label
        explanation = explain_with_llm(
            query,
            row["award_title"],
            row["abstract"],
            llm_label
        )

        results.append({
            "title": row["award_title"],
            "category": row["category"],
            "abstract": row["abstract"],
            "llm_label": llm_label,
            "explanation": explanation
        })

    return results

# Expose df_eval to app.py
df_eval_global = df_eval
