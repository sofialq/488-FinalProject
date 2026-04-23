import streamlit as st
import vectorDB
from RAG_Pipeline import rag_pipeline

st.title("IST 387 Code Helper")
st.write("created by Andrew Champagne, Marcus Johnson, Sofia Quintero, and Mars Schrag")

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

    # refresh UI so answer appears immediately
    st.rerun()



