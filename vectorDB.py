import streamlit as st
import sys
from pathlib import Path
from PyPDF2 import PdfReader
import openai
import chromadb
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

# client
if 'openai_client' not in st.session_state:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

# Streamlit setup
st.set_page_config(page_title="RAG Retriever", layout="wide")

# Chromadb setup
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
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