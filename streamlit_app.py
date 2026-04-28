import streamlit as st
import json
import os
import sys
import chromadb
import openai
import pdfplumber
from pathlib import Path
from openai import OpenAI
from sentence_transformers import CrossEncoder
import re

# user selection for user-based memory
st.sidebar.header("User Settings")

username = st.sidebar.text_input("Enter your username:", key="username_input")

# api key input
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    placeholder="sk-proj-...",
    key="api_key_input"
)

if not openai_api_key and not username:
    st.warning("Please enter your OpenAI API key and username to begin.")
    st.stop()

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to begin.")
else:
    st.session_state["openai_api_key"] = openai_api_key

# always store key in session_state
st.session_state["openai_api_key"] = openai_api_key


if username:
    st.sidebar.caption("Refresh the application if looking to change users.")

if not username:
    st.warning("Please enter a username to begin.")
    st.stop()

# normalize username for file naming
username = username.strip().lower().replace(" ", "_")
memory_file = f"memory_{username}.json"

st.sidebar.write(f"Active user: **{username}**")

## vectorDB 
# SQLite fix for Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# chromaDB client — no embedding function needed
chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_HelpBot")
collection = chroma_client.get_or_create_collection(name="IST387Collection")

if "collection" not in st.session_state:
    st.session_state.collection = collection


# text processing
def clean_text(text):
    return " ".join(text.split())


def extract_text_from_pdf_path(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return clean_text(text)


def chunk_text(text, chunk_size=800, overlap=150):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_embedding(text, api_key):
    client = OpenAI(api_key=st.session_state["openai_api_key"])
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# ingestion functions
def get_ingested_sources(collection):
    existing = collection.get()["metadatas"]
    return set(m["source"] for m in existing if m)


def add_to_collection(collection, text, file_name, api_key):
    chunks = chunk_text(text)
    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]

    embeddings = [get_embedding(chunk, st.session_state["openai_api_key"]) for chunk in chunks]

    try:
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"Added {len(chunks)} chunks from {file_name}")
    except Exception as e:
        print(f"Error adding {file_name}: {e}")


def load_pdfs(folder_path, collection, api_key):
    folder = Path(folder_path)
    already_ingested = get_ingested_sources(collection)

    newly_ingested = []
    skipped = []

    for pdf_file in folder.glob("*.pdf"):
        if pdf_file.name in already_ingested:
            skipped.append(pdf_file.name)
            print(f"Skipping (already ingested): {pdf_file.name}")
            continue

        print(f"Ingesting: {pdf_file.name}")
        text = extract_text_from_pdf_path(pdf_file)

        add_to_collection(collection, text, pdf_file.name, st.session_state["openai_api_key"])
        newly_ingested.append(pdf_file.name)

    return newly_ingested, skipped


# ingestion - runs once per session
if "ingestion_done" not in st.session_state:
    with st.spinner("Checking and ingesting documents..."):
        newly_ingested, skipped = load_pdfs(
            "./IST387_documents",
            st.session_state.collection,
            st.session_state["openai_api_key"]
        )
    st.session_state.ingestion_done = True


# retrieval with manual embeddings
def retrieve_context(query, k=4):
    collection = st.session_state.collection

    query_embedding = get_embedding(query, st.session_state["openai_api_key"])

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return None, None

    return docs, metas


## rag pipeline
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


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
                "Look up what this user has previously struggled with on a given topic, "
                "based on their long-term learning history."
            ),
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_practice_question",
            "description": (
                "Generate a personalized practice question for the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]}
                },
                "required": ["topic", "difficulty"]
            }
        }
    }
]


def summarize_topic_from_memory(topic, memories):
    topic_lower = topic.lower()
    topic_words = [w for w in topic_lower.split() if len(w) > 3]

    matches = [
        m for m in memories
        if any(re.search(rf'\b{re.escape(word)}\b', m.lower()) for word in topic_words)
    ] if memories else []

    if matches:
        match_str = "\n".join([f"- {m}" for m in matches])
        return f"Recorded struggles related to '{topic}':\n{match_str}"
    else:
        return (
            f"NO_MATCH: No recorded struggles found for '{topic}'. "
            f"You MUST respond with exactly: 'You have no recorded history of struggling with {topic}.' "
        )


def generate_practice_question(topic, difficulty, memories, context, api_key=""):
    client = OpenAI(api_key=st.session_state["openai_api_key"])

    topic_lower = topic.lower()
    matches = [m for m in memories if topic_lower in m.lower()] if memories else []
    memory_context = "\n".join([f"- {m}" for m in matches]) if matches else "No specific struggles recorded for this topic."

    prompt = f"""
        You are generating a practice question for an IST 387 student at Syracuse University.

        Topic: {topic}
        Difficulty: {difficulty}

        The student has the following recorded struggles related to this topic:
        {memory_context}

        Use the following course material as the basis for your question:
        {context[:2000]}

        Format your response exactly like this:

        **Question:**
        <the question here>

        **Answer Key:**
        <step-by-step answer here>

        **Why this question:**
        <one sentence explaining why this targets the student's struggles>
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ==============================
# RAG PIPELINE FUNCTION
# ==============================
def rag_pipeline(query, system_message=None, conversation_history=None, k=4, api_key=""):

    client = OpenAI(api_key=st.session_state["openai_api_key"])

    query_embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = query_embedding_response.data[0].embedding

    results = st.session_state.collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    docs, metas = rerank(query, docs, metas, top_n=4)

    context = "\n\n---\n\n".join(docs)
    sources = metas

    if not context:
        return "No relevant context found.", None

    prompt = build_prompt(query, context)

    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    if conversation_history:
        for turn in conversation_history:
            if turn["role"] in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": prompt})

    memory_keywords = ["struggling", "struggle", "confused", "where have i", "where am i",
                    "what have i", "what am i", "focus on", "should i study", "my weakness"]
    is_memory_query = any(kw in query.lower() for kw in memory_keywords)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "summarize_topic_from_memory"}
        } if is_memory_query else "auto"
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        memories = st.session_state.get("memories", [])

        if tool_name == "summarize_topic_from_memory":
            tool_result = summarize_topic_from_memory(tool_args["topic"], memories)

            if tool_result.startswith("NO_MATCH:"):
                return f"You have no recorded history of struggling with {tool_args['topic']}.", sources

        elif tool_name == "generate_practice_question":
            tool_result = generate_practice_question(
                topic=tool_args["topic"],
                difficulty=tool_args["difficulty"],
                memories=memories,
                context=context,
                api_key=st.session_state["openai_api_key"] 
            )

        else:
            tool_result = "Tool not found."

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_call.function.arguments
                    }
                }
            ]
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        answer = second_response.choices[0].message.content

    else:
        answer = response_message.content

    return answer, sources


def rerank(query, docs, metadatas, top_n=4):

    if not docs:
        return [], []

    pairs = [(query, doc) for doc in docs]

    scores = reranker.predict(pairs)
    scored_items = list(zip(docs, metadatas, scores))
    scored_items.sort(key=lambda x: x[2], reverse=True)

    top_items = scored_items[:top_n]

    reranked_docs = [item[0] for item in top_items]
    reranked_meta = [item[1] for item in top_items]

    return reranked_docs, reranked_meta

## streamlit app
# load + save memory
def load_memory(memory_file):
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data, None
            return data.get("memories", []), data.get("profile", None)
    return [], None


def save_memories(memory_file, memories, profile=None):
    with open(memory_file, "w") as f:
        json.dump({"memories": memories, "profile": profile}, f)


memories, saved_profile = load_memory(memory_file)
st.session_state.memories = memories

if saved_profile and "profile" not in st.session_state:
    st.session_state.profile = saved_profile


# study profile generator
def generate_profile(memories, username):
    if not memories:
        return "No learning data yet. Keep chatting to build your profile!"

    memory_str = "\n".join([f"- {m}" for m in memories])

    response = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""
            You are summarizing a student's learning profile for IST 387 at Syracuse University.

            Based on the following observed struggles, write a short, encouraging 3-part profile:
            1. Concepts to focus on
            2. Strengths
            3. Growth
            4. One personalized study tip

            Observed struggles:
            {memory_str}
            """
        }]
    )
    return response.choices[0].message.content


# system message
system_message = (
    "You are a helpful teaching assistant for IST 387 at Syracuse University. "
    "If reporting on a student's struggles or learning history, only reference what is explicitly recorded in their memory. "
    "Never infer or guess struggles that are not directly recorded."
)

if memories:
    memory_str = "\n".join([f"- {m}" for m in memories])
    system_message += (
        "\n\nHere are some things you've learned from previous interactions with this user:\n"
        f"{memory_str}\n\n"
        "Use this information to help answer the user's questions, but always rely on verified course materials first."
    )


# initialize chromaDB (no embedding function needed — manual embeddings used)
chroma_client = chromadb.PersistentClient(
    path="./ChromaDB_for_HelpBot"
)

collection = chroma_client.get_or_create_collection(
    name="IST387Collection"
)

if "collection" not in st.session_state:
    st.session_state.collection = collection


# initialize OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = openai.OpenAI(api_key=st.session_state["openai_api_key"])


# study profile - created with long-term memory
st.sidebar.divider()
with st.sidebar.expander("My Study Profile"):
    if st.button("Generate My Profile", key="gen_profile"):
        with st.spinner("Building your profile..."):
            st.session_state.profile = generate_profile(memories, username)
            save_memories(memory_file, memories, st.session_state.profile)

    if "profile" in st.session_state:
        st.markdown(st.session_state.profile)
    elif not memories:
        st.info("Chat with the assistant to build your profile!")


# streamlit ui
st.title("IST 387 Code Helper")
st.caption("created by Andrew Champagne, Marcus Johnson, Sofia Quintero, and Mars Schrag")

# keep chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# greet user once per session
if "greeted" not in st.session_state:
    st.session_state.greeted = True
    is_returning = bool(memories)

    if is_returning:
        if saved_profile:
            profile_section = f"\n\nHere's your study profile from last time:\n\n{saved_profile}"
        else:
            profile_section = "\n\nYou don't have a generated study profile yet — click **Generate My Profile** in the sidebar anytime!"

        welcome_msg = (
            f"Welcome back, **{username}**!"
            f"{profile_section}\n\n"
            "Feel free to pick up where you left off — what would you like to work on today?"
        )
    else:
        welcome_msg = (
            f"Welcome {username}! \nAsk any questions about IST 387 and get answers based on verified course materials.\n\n"
            "If the answer isn't in the materials, I'll point you in the right direction! "
            "The assistant can also learn from the conversation to better assist you in the future!"
        )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})


# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# chat input
question = st.chat_input("Ask any questions about IST 387...")

if question:
    # store user message
    st.session_state.messages.append({"role": "user", "content": question})

    # display user message immediately
    with st.chat_message("user"):
        st.write(question)

    # build short-term memory from session history - exclude the current message (last item) since it's passed separately as `question`
    conversation_history = st.session_state.messages[:-1]
    
    # cap history to last 6 interactions to avoid token overflows
    max_interactions = 6
    conversation_history = conversation_history[-(max_interactions * 2):]

    # generate answer using RAG + short-term memory + long-term memory
    with st.chat_message("assistant"):
        with st.spinner("Searching verified documents..."):
            answer, _ = rag_pipeline(
                question,
                system_message,
                conversation_history=conversation_history,
                api_key=st.session_state["openai_api_key"]
            )

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

        Return ONLY valid JSON.
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
                st.session_state.memories = memories
                save_memories(memory_file, memories, st.session_state.get("profile"))

        except json.JSONDecodeError:
            st.session_state.last_extracted_memories = "JSON decode error"


    # refresh UI so answer appears immediately
    st.rerun()
