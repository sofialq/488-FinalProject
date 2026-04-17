import streamlit as st
from anthropic import Anthropic
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

st.title("IST 387 Code Helper")
st.write("created by Andrew Champagne, Marcus Johnson, Sofia Quintero, and Mars Schrag")

#api key
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
client = Anthropic(api_key=anthropic_api_key)

