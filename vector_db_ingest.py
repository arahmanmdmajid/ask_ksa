import os
import torch
import chromadb
import shutil
from config import VECTOR_DB_DIR, EMBED_MODEL_NAME
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import load_all_publications, slugify

_embedding_model = None


def initialize_db(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
    delete_existing: bool = False,
) -> chromadb.Collection:
    """
    Initialize a ChromaDB instance and persist it to disk.

    Args:
        persist_directory (str): The directory where ChromaDB will persist data. Defaults to "./vector_db"
        collection_name (str): The name of the collection to create/get. Defaults to "publications"
        delete_existing (bool): Whether to delete the existing database if it exists. Defaults to False
    Returns:
        chromadb.Collection: The ChromaDB collection instance
    """
    if os.path.exists(persist_directory) and delete_existing:
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or get a collection
    try:
        # Try to get existing collection first
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: {collection_name}")
    except Exception:
        # If collection doesn't exist, create it
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:batch_size": 10000,
            },  # Use cosine distance for semantic search
        )
        print(f"Created new collection: {collection_name}")

    print(f"ChromaDB initialized with persistent storage at: {persist_directory}")

    return collection


def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
) -> chromadb.Collection:
    """
    Get a ChromaDB client instance.

    Args:
        persist_directory (str): The directory where ChromaDB persists data
        collection_name (str): The name of the collection to get

    Returns:
        chromadb.PersistentClient: The ChromaDB client instance
    """
    return chromadb.PersistentClient(path=persist_directory).get_collection(
        name=collection_name
    )


def chunk_publication(
    content: str,
    title: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(content)

    title_slug = slugify(title)

    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append(
            {
                "content": chunk,
                "title": title,
                "chunk_id": f"{title_slug}_{i}",
            }
        )

    return chunk_data

# Global embedding model instance (lazy-loaded)
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from config import EMBED_MODEL_NAME

_embedding_model = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": device},
        )
    return _embedding_model

def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.embed_documents(texts)


def insert_publications(collection, publications: list[dict]):
    """
    Insert documents into a ChromaDB collection.

    publications: list of dicts with at least:
      - title
      - content
      - (optionally) source_url, path, scraped_at
    """
    for pub in publications:
        title = pub["title"]
        content = pub["content"]

        # 1) Chunk with metadata
        chunk_data = chunk_publication(content=content, title=title)

        # 2) Extract pieces for Chroma
        documents = [c["content"] for c in chunk_data]   # <- strings
        ids = [c["chunk_id"] for c in chunk_data]
        metadatas = [
            {
                "title": c["title"],
                "source_url": pub.get("source_url"),
                "path": pub.get("path"),
                "scraped_at": pub.get("scraped_at"),
            }
            for c in chunk_data
        ]

        # 3) Embed only the content strings
        embeddings = embed_documents(documents)

        # 4) Add to Chroma
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )



def main():
    print(VECTOR_DB_DIR)
    collection = initialize_db(
        persist_directory=VECTOR_DB_DIR,
        collection_name="publications",
        delete_existing=True,
    )
    publications = load_all_publications()
    insert_publications(collection, publications)

    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()
