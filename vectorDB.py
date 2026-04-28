import streamlit as st
import sys
from pathlib import Path
import pdfplumber
import openai
import chromadb
from chromadb.utils import embedding_functions

# use the api key entered by the user in the sidebar
openai_api_key = st.session_state.get("api_key_input", "")


# SQLite fix for Streamlit Cloud
if "streamlit.runtime.scriptrunner.script_runner" in sys.modules:
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass


# OpenAI client setup
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = openai.OpenAI(api_key=openai_api_key)


# chromaDB setup
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-ada-002"
)

chroma_client = chromadb.PersistentClient(
    path="./ChromaDB_for_HelpBot"
)

collection = chroma_client.get_or_create_collection(
    name="IST387Collection",
    embedding_function=embedding_fn
)

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


# ingestion functions
def get_ingested_sources(collection):
    existing = collection.get()["metadatas"]
    return set(m["source"] for m in existing if m)


def add_to_collection(collection, text, file_name):
    chunks = chunk_text(text)
    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]

    try:
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"Added {len(chunks)} chunks from {file_name}")
    except Exception as e:
        print(f"Error adding {file_name}: {e}")


def load_pdfs(folder_path, collection):
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
        add_to_collection(collection, text, pdf_file.name)
        newly_ingested.append(pdf_file.name)

    return newly_ingested, skipped


# ingestion - runs once per session
if "ingestion_done" not in st.session_state:
    with st.spinner("Checking and ingesting documents..."):
        newly_ingested, skipped = load_pdfs("./IST387_documents", st.session_state.collection)

    st.session_state.ingestion_done = True

# retrieval
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

    return docs, metas


## querying collection for testing - uncomment to test
#topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')

#if topic:
    #client = st.session_state.openai_client
    #response = client.embeddings.create(
        #input=topic,
        #model='text-embedding-3-small'
    #)

    # get the embedding
    #query_embedding = response.data[0].embedding

    # get text related to this question (this prompt)
    #results = collection.query(
        #query_embeddings=[query_embedding],
        #n_results=3
    #)

    # display the results
    #st.subheader(f'Results for: {topic}')

    #for i in range(len(results['documents'][0])):
        #doc = results['documents'][0][i]
        #doc_id = results['ids'][0][i]

        #st.write(f'**{i+1}. {doc_id}**')
        #st.write(doc)

#else:
    #st.info('Enter a topic in the sidebar to search the collection')