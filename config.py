from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

# Old FAISS config (you can keep it for now, but it won’t be used)
INDEX_PATH = os.getenv("INDEX_PATH", str(BASE_DIR / "faiss_index_ip.bin"))
CHUNKS_PATH = os.getenv("CHUNKS_PATH", str(BASE_DIR / "chunks.json"))
META_PATH  = os.getenv("META_PATH",  str(BASE_DIR / "chunks_metadata.json"))

# New Chroma config
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", str(BASE_DIR / "vector_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "publications")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", str(BASE_DIR / "outputs"))
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))