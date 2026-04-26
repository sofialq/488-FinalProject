import streamlit as st
import json
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from RAG_Pipeline import rag_pipeline
import openai

openai_api_key = st.secrets["OPENAI_API_KEY"]


# user selection for user-based memory
st.sidebar.header("User Settings")

username = st.sidebar.text_input("Enter your username:", key="username_input")

if not username:
    st.warning("Please enter a username to begin.")
    st.stop()

# normalize username for file naming
username = username.strip().lower().replace(" ", "_")
memory_file = f"memory_{username}.json"

st.sidebar.write(f"Active user: **{username}**")


# load + save memory
def load_memory(memory_file):
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            return json.load(f)
    return []

def save_memories(memory_file, memories):
    with open(memory_file, "w") as f:
        json.dump(memories, f)

memories = load_memory(memory_file)


# system message
system_message = "You are a helpful teaching assistant for IST 387 at Syracuse University."

if memories:
    memory_str = "\n".join([f"- {m}" for m in memories])
    system_message += (
        "\n\nHere are some things you've learned from previous interactions with this user:\n"
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


# initialize OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = openai.OpenAI(api_key=openai_api_key)


# memory debug panel
#with st.sidebar.expander("Memory Debug Panel"):
    #st.write("**Active user:**", username)
    #st.write("**Memory file:**", memory_file)
    #st.write("**File exists:**", os.path.exists(memory_file))
    #st.write("**Current working directory:**", os.getcwd())
    #st.write("**Directory writable:**", os.access(os.getcwd(), os.W_OK))
    #st.write("**Loaded memories:**", memories)

    #if "last_extracted_memories" in st.session_state:
        #st.write("**Last extracted memories:**", st.session_state.last_extracted_memories)
    #else:
        #st.write("**Last extracted memories:** None yet")


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

    # build short-term memory from session history - excludes current message
    conversation_history = st.session_state.messages[:-1]

    # token cap
    max_interactions = 6
    conversation_history = conversation_history[-(max_interactions * 2):]

    # generate answer using RAG + short-term memory + long-term memory
    with st.chat_message("assistant"):
        with st.spinner("Searching verified documents..."):
            answer, _ = rag_pipeline(question, system_message, conversation_history=conversation_history)

        st.write(answer)

    # store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


    # memory extraction
    if len(st.session_state.messages) >= 2:
        user_msg = st.session_state.messages[-2]["content"]
        assistant_msg = st.session_state.messages[-1]["content"]

        extraction_prompt = f"""
        You are extracting long-term learning signals from a tutoring conversation.

        Your task:
        - Identify 1–3 concepts the user appears to struggle with.
        - ONLY return a JSON list of short phrases.
        - If nothing new is learned, return [].

        Already known memories:
        {json.dumps(memories)}

        User message: {user_msg}
        Assistant message: {assistant_msg}

        Return ONLY valid JSON. No explanation.
        Example: ["confused about joins", "struggling with map functions"]
        """

        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}]
        )

        try:
            new_memories = json.loads(response.choices[0].message.content)
            st.session_state.last_extracted_memories = new_memories

            new_memories = [m for m in new_memories if m not in memories]
            
            if new_memories:
                memories.extend(new_memories)
                save_memories(memory_file, memories)

        except json.JSONDecodeError:
            st.session_state.last_extracted_memories = "JSON decode error"


    # refresh UI so answer appears immediately
    st.rerun()






