import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path
from PyPDF2 import PdfReader

# working with chromadb on streamlit community cloud
if "streamlit.runtime.scriptrunner.script_runner" in sys.modules:
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        st.warning("pysqlite3 not available locally; using system sqlite3")

from chromadb.config import Settings
import chromadb

# create client - openai used for embeddings for chromadb
if 'openai_client' not in st.session_state:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

# extract text from pdf
def extract_text_from_pdf_path(pdf_path):

    '''
    this function extracts text from each assignment and document 
    to pass to add_to_collection
    '''

    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # basic cleaning (helps embeddings)
    text = " ".join(text.split())

    return text


# split text into chunks for better retrieval
def chunk_text(text, chunk_size=500, overlap=100):

    '''
    splits large text into smaller overlapping chunks
    improves embedding + retrieval quality
    '''

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# create chromadb client
chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_HelpBot")
collection = chroma_client.get_or_create_collection("IST387Collection")


# using chromadb with openai embeddings 
def add_to_collection(collection, text, file_name):

    '''
    function to add documents to collections
    
    collection: chromadb collection (already established)
    text: extracted text from pdf files
    
    embeddings inserted into the collection from openai
    '''

    client = st.session_state.openai_client

    # split into chunks
    chunks = chunk_text(text)

    # create embeddings in batch (faster)
    response = client.embeddings.create(
        input=chunks,
        model='text-embedding-3-small'
    )

    embeddings = [item.embedding for item in response.data]

    # add each chunk separately
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name} for _ in chunks]

    collection.add(
        documents=chunks,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )


# populate collection with pdfs
def load_pdfs_to_collection(folder_path, collection):

    '''
    this function uses extract_text_from_pdf and 
    add_to_collection to put assignments and documents in chromadb collection
    '''

    loaded_files = []

    folder = Path(folder_path)

    # loop through all PDF files in the folder
    for pdf_file in folder.glob("*.pdf"):

        # extract text from the PDF
        text = extract_text_from_pdf_path(pdf_file)

        # add to ChromaDB
        add_to_collection(collection, text, pdf_file.name)

        loaded_files.append(pdf_file.name)

    return loaded_files


# check if collection is empty and load PDFs
if collection.count() == 0:
    loaded = load_pdfs_to_collection('./IST387_documents', collection)


def get_rag_context(query):

    '''
    query chromadb for relevant information based on user query
    '''

    # create embedding for query
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )

    # get embedding
    query_embedding = response.data[0].embedding

    # get text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    # combine the retrieved documents into context
    if results['documents'][0]:
        context = "\n\n---\n\n".join(results['documents'][0])
        source_files = [m["source"] for m in results["metadatas"][0]]
        return context, source_files
    else:
        return None, None
    

## querying collection for testing - uncomment to test
topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')

if topic:
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=topic,
        model='text-embedding-3-small'
    )

    # get the embedding
    query_embedding = response.data[0].embedding

    # get text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    # display the results
    st.subheader(f'Results for: {topic}')

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        doc_id = results['metadatas'][0][i]["source"]

        st.write(f'**{i+1}. {doc_id}**')
        st.write(doc)

else:
    st.info('Enter a topic in the sidebar to search the collection')