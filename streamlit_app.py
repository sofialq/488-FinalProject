import streamlit as st
import json
import os
import chromadb
import openai

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

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to begin.")
    st.stop()

# make key available to vectorDB + RAG
st.session_state["openai_api_key"] = openai_api_key

from RAG_Pipeline import rag_pipeline


if username:
    st.sidebar.caption("Refresh the application if looking to change users.")

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
            data = json.load(f)
            # handle both old format (plain list) and new format (dict)
            if isinstance(data, list):
                return data, None
            return data.get("memories", []), data.get("profile", None)
    return [], None


def save_memories(memory_file, memories, profile=None):
    with open(memory_file, "w") as f:
        json.dump({"memories": memories, "profile": profile}, f)


memories, saved_profile = load_memory(memory_file)
st.session_state.memories = memories  # make memories available to tools in RAG_Pipeline

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
    st.session_state.openai_client = openai.OpenAI(api_key=openai_api_key)

# memory debug panel - uncomment for debugging
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
    is_returning = bool(memories) # returning if they have saved memories

    # welcome-back message with saved profile if available
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
                api_key=openai_api_key
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
