import streamlit as st
import vectorDB
from RAG_Pipeline import rag_pipeline
import json
import os

# load + save memory
def load_memory():
    if os.path.exists("memory.json"):
        with open("memory.json", "r") as f:
            return json.load(f)
    return []

memories = load_memory()

def save_memories(memories):
    with open("memory.json", "w") as f:
        json.dump(memories, f)

# system message with memory
system_message = "You are a helpful teaching assistant for IST 387 at Syracuse University."
if memories:
    memory_str = "\n".join([f"- {m}" for m in memories])
    system_message += (
        "\n\nHere are some things you've learned from previous interactions:\n"
        f"{memory_str}\n\n"
        "Use this information to help answer the user's questions, but do not rely on it exclusively. Always use the provided context from verified course materials to answer questions, and only use the memory as supplementary information."

    )

# bot description
st.title("IST 387 Code Helper")
st.caption("created by Andrew Champagne, Marcus Johnson, Sofia Quintero, and Mars Schrag")
st.write("This app is designed to help students in IST 387 by providing accurate answers to their questions based on verified course materials. " \
"Ask any question related to the course, and the assistant will search through the provided documents to give you a helpful response. " \
"If the answer isn't found in the materials, it will guide you on where to look next!")


# keep chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# chat input
question = st.chat_input("Ask any questions about IST 387...")

if question:
    # store user message
    st.session_state.messages.append({"role": "user", "content": question})

    # display user message immediately
    with st.chat_message("user"):
        st.write(question)

    # generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching verified documents..."):
            answer, _ = rag_pipeline(question) 

        st.write(answer)

    # store assistant message (no sources)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # memory extraction - only run if we have at least 2 messages (a user question and an assistant answer)
    if len(st.session_state.messages) >= 2:
        user_msg = st.session_state.messages[-2]["content"]
        assistant_msg = st.session_state.messages[-1]["content"]

        extraction_prompt = f"""
        Analyze the conversation and identify what concepts the user seems to struggle with.
        Extract 1–3 key concepts based on their questions and the assistant's answers.

        Already known memories:
        {json.dumps(memories)}

        User message: {user_msg}
        Assistant message: {assistant_msg}

        Return ONLY a JSON list of new facts. If no new facts, return [].
        """

        # model call
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}]
        )

        try:
            new_memories = json.loads(response.choices[0].message.content)
            if new_memories:
                memories.extend(new_memories)
                save_memories(memories)
        except json.JSONDecodeError:
            pass

    # refresh UI so answer appears immediately
    st.rerun()




