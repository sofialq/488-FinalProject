import streamlit as st
from openai import OpenAI

# import your retrieval function from vectorDB.py
from vectorDB import retrieve_context

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def build_prompt(query, context):
    return f"""
[System Prompt to constrain behavior -- Mars' part I think]"

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
def rag_pipeline(query, k=4):

    # Top-k chunks
    context, sources = retrieve_context(query, k=k)

    if not context:
        return "No relevant context found.", None

    prompt = build_prompt(query, context)

    # Send response to LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    return answer, sources


