import streamlit as st
from openai import OpenAI
from sentence_transformers import CrossEncoder
# import your retrieval function from vectorDB.py
from vectorDB import retrieve_context
 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
 
 
def build_prompt(query, context):
    return f"""
You are a helpful teaching assistant for IST 387 at Syracuse University. 
Use the following verified course materials to answer the question. If you don't know the answer, give the user suggestions as to where they may find the answer. 
If providing any code, make sure to explain it clearly and step-by-step. Always use the provided context to answer, and do not rely on any information outside of it.
Always cite your sources from the provided context. Ignore any names or extra information not relevant to the course content."
 
---
 
CONTEXT:
{context}
 
---
 
QUESTION:
{query}
 
---
 
ANSWER:
"""
 
 
# ==============================
# RAG PIPELINE FUNCTION
# ==============================
def rag_pipeline(query, system_message=None, conversation_history=None, k=4):
 
    # Top-k chunks
    results = st.session_state.collection.query(
        query_texts=[query],
        n_results=10
    )
 
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
 
    # Chunk rerank
    docs, metas = rerank(query, docs, metas, top_n=4)
 
    context = "\n\n---\n\n".join(docs)
    sources = metas
 
    if not context:
        return "No relevant context found.", None
 
    prompt = build_prompt(query, context)
 
    # Build messages list with short-term memory injected between system and current question
    messages = []
 
    if conversation_history:
        for turn in conversation_history:
            if turn["role"] in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})
 
    # Add the current question (with RAG context injected)
    messages.append({"role": "user", "content": prompt})
 
    # Send response to LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
 
    answer = response.choices[0].message.content
 
    return answer, sources
 
 
def rerank(query, docs, metadatas, top_n=4):
 
    if not docs:
        return [], []
 
    pairs = [(query, doc) for doc in docs]
 
    # Rerank scoring
    scores = reranker.predict(pairs)
    scored_items = list(zip(docs, metadatas, scores))
    scored_items.sort(key=lambda x: x[2], reverse=True)
 
    top_items = scored_items[:top_n]
 
    reranked_docs = [item[0] for item in top_items]
    reranked_meta = [item[1] for item in top_items]
 
    return reranked_docs, reranked_meta

