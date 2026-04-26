import json
import streamlit as st
from openai import OpenAI
from sentence_transformers import CrossEncoder
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


# tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "summarize_topic_from_memory",
            "description": (
                "Reference the user's study profile to understand what topics they struggle with, "
                "based on their long-term learning history. Use this when the user asks "
                "what they find confusing, what they should study, or asks about their "
                "progress or struggles with a specific concept."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The concept or topic to look up in the user's memory, e.g. 'joins', 'for loops', 'ggplot'"
                    }
                },
                "required": ["topic"]
            }
        }
    }
]


# tool executions
def summarize_topic_from_memory(topic, memories):
    """
    Searches the user's long-term memory and generated profile for struggles
    related to the given topic. Returns a plain-text summary for the LLM.
    """
    topic_lower = topic.lower()

    # search raw memories for topic matches
    matches = [m for m in memories if topic_lower in m.lower()] if memories else []

    # pull in the generated profile if one exists
    profile = st.session_state.get("profile", None)

    # build the result
    result_parts = []

    if matches:
        match_str = "\n".join([f"- {m}" for m in matches])
        result_parts.append(f"Recorded struggles related to '{topic}':\n{match_str}")

    if profile:
        result_parts.append(f"Overall student profile (use this for broader context):\n{profile}")

    if not result_parts:
        return (
            f"No recorded struggles found related to '{topic}' or no profile has been "
            f"generated yet. Encourage the user to keep chatting so a profile can be built."
        )

    return "\n\n".join(result_parts)


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

    # Build messages list with short-term memory injected
    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    if conversation_history:
        for turn in conversation_history:
            if turn["role"] in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

    # Add the current question with RAG context
    messages.append({"role": "user", "content": prompt})

    # First LLM call — may invoke a tool
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # Check if the LLM wants to call a tool
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # Execute the tool
        if tool_name == "summarize_topic_from_memory":
            memories = st.session_state.get("memories", [])
            tool_result = summarize_topic_from_memory(tool_args["topic"], memories)
        else:
            tool_result = "Tool not found."

        # Append the assistant's tool call and the tool result to messages
        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

        # Second LLM call — now with the tool result included
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        answer = second_response.choices[0].message.content

    else:
        # No tool call — use the first response directly
        answer = response_message.content

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