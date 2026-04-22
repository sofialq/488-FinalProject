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
        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(source)

# chat input
if question := st.chat_input("Ask any questions about IST 387..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching verified documents..."):
            answer, sources = rag_pipeline(question)

        # st.write("**DEBUG: Collection count:**", st.session_state.collection.count())

    st.session_state.messages.append({"role": "assistant", "content": answer})

