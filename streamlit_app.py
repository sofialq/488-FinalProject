import streamlit as st
import json
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from RAG_Pipeline import rag_pipeline


# load + save memory
def load_memory():
    if os.path.exists("memory.json"):
        with open("memory.json", "r") as f:
            return json.load(f)
    return []

def save_memories(memories):
    with open("memory.json", "w") as f:
        json.dump(memories, f)

memories = load_memory()

system_message = "You are a helpful teaching assistant for IST 387 at Syracuse University."

if memories:
    memory_str = "\n".join([f"- {m}" for m in memories])
    system_message += (
        "\n\nHere are some things you've learned from previous interactions:\n"
        f"{memory_str}\n\n"
        "Use this information to help answer the user's questions, but do not rely on it exclusively. "
        "Always use verified course materials as your primary source."
    )


# initialize chromaDB
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002"
)

chroma_client = chromadb.PersistentClient(
    path="./ChromaDB_for_HelpBot",
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(
    name="IST387Collection",
    embedding_function=embedding_fn
)

if "collection" not in st.session_state:
    st.session_state.collection = collection


# memory debug panel
with st.sidebar.expander(" Memory Debug Panel"):
    st.write("**memory.json exists:**", os.path.exists("memory.json"))
    st.write("**Current working directory:**", os.getcwd())
    st.write("**Directory writable:**", os.access(os.getcwd(), os.W_OK))
    st.write("**Loaded memories:**", memories)

    if "last_extracted_memories" in st.session_state:
        st.write("**Last extracted memories:**", st.session_state.last_extracted_memories)
    else:
        st.write("**Last extracted memories:** None yet")


# streamlit ui
st.title("IST 387 Code Helper")
st.caption("created by Andrew Champagne, Marcus Johnson, Sofia Quintero, and Mars Schrag")
st.write("""Ask any questions about IST 387 and get answers based on verified course materials. 
         If the answer isn't in the materials, I'll do my best to point you in the right direction!
         The assistant can also learn from the conversation to better assist you in the future!""")

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

    # generate answer using RAG + memory
    with st.chat_message("assistant"):
        with st.spinner("Searching verified documents..."):
            answer, _ = rag_pipeline(question, system_message)

        st.write(answer)

    # store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


    # memory extraction - only if we have at least 2 messages (1 user + 1 assistant)
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

        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}]
        )

        try:
            new_memories = json.loads(response.choices[0].message.content)
            st.session_state.last_extracted_memories = new_memories

            if new_memories:
                memories.extend(new_memories)
                save_memories(memories)

        except json.JSONDecodeError:
            st.session_state.last_extracted_memories = "JSON decode error"


    # refresh UI so answer appears immediately
    st.rerun()





