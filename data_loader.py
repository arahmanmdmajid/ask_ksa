# data_loader.py
import json
import os
from pathlib import Path

import chromadb
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL_NAME, VECTOR_DB_DIR, CHROMA_COLLECTION_NAME

def _check_files_exist(paths):
    """Raise a clear error if any of the needed files is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))


@st.cache_resource(show_spinner=False)
def load_resources():
    """
    Load the embedding model and Chroma collection used for RAG.
    Called once per Streamlit session and cached.
    """

    # Load embedding model (for query encoding only)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 2) Connect to the persistent Chroma DB
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    # return embed_model, index, all_chunks, all_chunks_metadata
    return embed_model, collection