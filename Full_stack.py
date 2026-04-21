import streamlit as st
import sys
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader
from vectorDB import retrieve_context
from sentence_transformers import CrossEncoder
from openai import OpenAI
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ==============================
# SQLite fix for Streamlit Cloud
# ==============================
if "streamlit.runtime.scriptrunner.script_runner" in sys.modules:
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        st.warning("pysqlite3 not available locally; using system sqlite3")

# Streamlit setup
st.set_page_config(page_title="RAG Retriever", layout="wide")

# Chromadb setup
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="./ChromaDB_for_HelpBot",
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(
    name="IST387Collection",
    embedding_function=embedding_fn
)

# Instantiate db
if "collection" not in st.session_state:
    st.session_state.collection = collection


# Data preprocessing
def clean_text(text):
    return " ".join(text.split())


def extract_text_from_pdf_path(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return clean_text(text)


# Chunk PDFs
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


# Add to db
def add_to_collection(collection, text, file_name):

    chunks = chunk_text(text)

    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]

    try:
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
    except Exception:
        pass


# PDF load
def load_pdfs(folder_path, collection):
    folder = Path(folder_path)

    for pdf_file in folder.glob("*.pdf"):
        text = extract_text_from_pdf_path(pdf_file)
        add_to_collection(collection, text, pdf_file.name)


# db load
if st.session_state.collection.count() == 0:
    with st.spinner("Loading documents into vector database..."):
        load_pdfs("./IST387_documents", st.session_state.collection)


# Retrieve context w/ metadata function
def retrieve_context(query, k=4):

    collection = st.session_state.collection

    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return None, None

    context = "\n\n---\n\n".join(docs)
    return context, metas


# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("Controls")

if st.sidebar.button("Rebuild Vector DB"):

    chroma_client = chromadb.PersistentClient(
        path="./ChromaDB_for_HelpBot",
        settings=Settings(anonymized_telemetry=False)
    )

    st.session_state.collection = chroma_client.get_or_create_collection(
        name="IST387Collection",
        embedding_function=embedding_fn
    )

    collection = st.session_state.collection

    for pdf_file in Path("./IST387_documents").glob("*.pdf"):
        text = extract_text_from_pdf_path(pdf_file)
        add_to_collection(collection, text, pdf_file.name)

    st.sidebar.success("Rebuilt successfully!")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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

    # Send response to LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
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


