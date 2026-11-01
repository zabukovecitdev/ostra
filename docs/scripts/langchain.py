"""Split a text file, embed the chunks with a local Ollama model and persist them in sqlite-vec."""

from pathlib import Path

from langchain_community.vectorstores.sqlitevec import SQLiteVec
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = SCRIPT_DIR / "sample.txt"
DB_FILE = SCRIPT_DIR / "sample.db"
TABLE = "sample_chunks"

EMBEDDING_MODEL = "mxbai-embed-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Read the source text and wrap it in a Document so metadata travels with the chunks.
document = Document(page_content=SOURCE.read_text(encoding="utf-8"), metadata={"source": str(SOURCE)})

# Split on blank lines first, falling back to character-based chunks of CHUNK_SIZE.
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents([document])
print(f"Split {SOURCE.name} into {len(chunks)} chunks")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# SQLiteVec appends on every insert and never deduplicates, so rebuild the database from scratch.
if DB_FILE.exists():
    DB_FILE.unlink()

store = SQLiteVec.from_texts(
    texts=[chunk.page_content for chunk in chunks],
    embedding=embeddings,
    metadatas=[chunk.metadata for chunk in chunks],
    table=TABLE,
    db_file=str(DB_FILE),
)
print(f"Persisted {len(chunks)} embedded chunks to {DB_FILE} (table '{TABLE}')")

# Sanity check: run a similarity search against the persisted store.
query = "What is this document about?"
for hit in store.similarity_search(query, k=3):
    print("---")
    print(hit.page_content[:200])
