"""
app.py
------
Streamlit Frontend UI for the NSF Hybrid Retrieval System.

This interface provides:
1. A Search tab — retrieves relevant grants and shows LLM explanations.
2. An Evaluation tab — runs full evaluation using:
   - Human labels
   - LLM labels
   - Precision@5, MRR, nDCG
   - Human–LLM agreement score
"""
# =====================================
# Streamlit Page Setup
# =====================================

import streamlit as st
from retrieval_core import search_and_explain, evaluate_query, df_eval_global

st.set_page_config(page_title="NSF Grant Semantic Search", layout="wide")
# Two tabs: SEARCH and EVALUATION
tab1, tab2 = st.tabs(["🔍 Search Grants", "📊 Evaluation"])

# =====================================
# TAB 1 — SEARCH PAGE
# =====================================

with tab1:

    st.title("NSF Grant Semantic Search Demo")
    st.write(
        "This tool retrieves relevant NSF grants using Hybrid BM25 + SBERT and "
        "explains relevance using an LLM."
    )
    # Input box for research idea
    query = st.text_area("Enter your research idea:", height=120)
    top_k = st.selectbox("Number of results:", [3, 5], index=1)

    # Run search
    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a query first.")
        else:
            with st.spinner("Searching grants and generating explanations..."):
                results = search_and_explain(query, top_k=top_k)

            st.subheader("Results")
            if not results:
                st.write("No results found.")
            else:
                #display each retrieved result
                for i, r in enumerate(results, start=1):
                    st.markdown(f"### {i}. {r['title']}")
                    st.write(f"**Category:** {r['category']}")
                    
                    #Show abstract expandable
                    with st.expander("Show abstract"):
                        st.write(r["abstract"])
                    
                    # LLM explanation
                    st.markdown("**Why is this relevant?**")
                    st.write(r["explanation"])
                    st.markdown("---")


# =====================================
# TAB 2 — EVALUATION PAGE
# =====================================

with tab2:

    st.title("Model Evaluation")
    st.write("""
    This section evaluates retrieval using:
    - Human labels (binary relevance)
    - LLM relevance labels
    - IR metrics: Precision@5, MRR, nDCG
    - Human–LLM agreement score
    """)
    
    #Query mapping for evaluation
    query_map = {
        "BIO – Early cancer detection": (
            "Developing AI-based models for early detection of cancer using blood biomarkers.",
            "BIO",
            "bio_cancer_detection"
        ),
        "BIO – Protein interactions": (
            "Developing machine learning models to study protein interactions and cellular processes in biological systems.",
            "BIO",
            "bio_protein_interactions"
        ),
        "CNS – Zero-day cloud attacks": (
            "Designing advanced cybersecurity methods to protect cloud-based systems from zero-day attacks and data breaches.",
            "CNS",
            "cns_zero_day"
        ),
        "IIS – Recommendation systems": (
            "Building machine learning algorithms to enhance personalized recommendation systems for e-commerce platforms.",
            "IIS",
            "iis_recommendation"
        )
    }

    st.subheader("Select Query to Evaluate")
    choice = st.selectbox("Choose one:", list(query_map.keys()))

    if st.button("Run Evaluation"):
        query_text, expected_cat, query_key = query_map[choice]

        with st.spinner("Running evaluation..."):
            (
                retrieved_indices,
                retrieved_categories,
                human_labels,
                llm_labels,
                explanations,
                metrics
            ) = evaluate_query(query_text, expected_cat, query_key)

        df_eval = df_eval_global  # <--- FIXED HERE

        # -------------------------------------------------
        # DISPLAY RETRIEVED RESULTS + HUMAN +LLM LABELS
        # -------------------------------------------------
        st.subheader("Retrieved Grants (Top-5)")

        for i, idx in enumerate(retrieved_indices):
            st.markdown(f"### {i+1}. {df_eval.iloc[idx]['award_title']}")
            st.write(f"**Category:** {retrieved_categories[i]}")
            st.write(f"**Human relevance:** {human_labels[i]}")
            st.write(f"**LLM relevance:** {llm_labels[i]}")
            st.write(f"**Explanation:** {explanations[i]}")
            st.markdown("---")

        # ------------------------
        # EVALUATION METRICS
        # ------------------------
        st.subheader("Evaluation Metrics")
        st.write(f"**Precision@5:** {metrics['precision']:.3f}")
        st.write(f"**MRR:** {metrics['mrr']:.3f}")
        st.write(f"**nDCG:** {metrics['ndcg']:.3f}")
        
        #Human-LLM Agreement(score)
        agreement = sum(
            1 for i in range(5) if human_labels[i] == llm_labels[i]
        ) / 5

        st.write(f"**Human–LLM Agreement Score:** {agreement:.2f}")
 
