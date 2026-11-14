"""Stage 0: read the 10-K filings and store their chunks in Qdrant.

Run with:  uv run python -m backend.ingest
"""
import csv
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

load_dotenv()  # reads the keys from .env into environment variables

DATA = Path(__file__).resolve().parent.parent / "data"
OPENROUTER = "https://openrouter.ai/api/v1"
COLLECTION = os.environ.get("QDRANT_COLLECTION", "tenk")

# Each Item starts with a heading such as "## Item 1A. Risk Factors".
ITEM = re.compile(r"^## Item (\w+)\. (.+)$", re.M)

# A chunk is about 1,500 characters. Neighbouring chunks share 150 characters,
# so a sentence cut at the edge still appears whole in one of them.
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)


def read_items(path: Path) -> list[dict]:
    """Return one dict per Item: code, title, and text."""
    text = path.read_text(encoding="utf-8")
    heads = list(ITEM.finditer(text))
    items = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        items.append({"code": h.group(1), "title": h.group(2), "text": text[h.end():end].strip()})
    return items


def chunk_id(company: str, year: int, item: str, n: int) -> str:
    """A stable ID: the same chunk always gets the same ID, so a re-run overwrites it."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{company}/{year}/{item}/{n}"))


def make_chunks(row: dict) -> tuple[list[Document], list[str]]:
    """Split one filing into chunks. Each Item is split on its own, so a chunk never crosses Items."""
    docs, ids = [], []
    year = int(row["fiscal_year"])
    for item in read_items(DATA / row["file"]):
        for n, piece in enumerate(splitter.split_text(item["text"])):
            docs.append(Document(page_content=piece, metadata={
                "company": row["ticker"], "year": year, "item": item["code"],
                "item_title": item["title"], "url": row["source_url"],
            }))
            ids.append(chunk_id(row["ticker"], year, item["code"], n))
    return docs, ids


def get_client() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",  # 1536 numbers per vector
        base_url=OPENROUTER,
        api_key=os.environ["OPENROUTER_API_KEY"],
        check_embedding_ctx_length=False,  # required for OpenRouter
    )


def make_filter(company: str, year: int) -> models.Filter:
    return models.Filter(must=[
        models.FieldCondition(key="metadata.company", match=models.MatchValue(value=company)),
        models.FieldCondition(key="metadata.year", match=models.MatchValue(value=year)),
    ])


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection once. The indexes make filtering by company, year, and Item fast."""
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        COLLECTION, vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))
    client.create_payload_index(COLLECTION, "metadata.company", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.item", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.year", models.PayloadSchemaType.INTEGER)


def main() -> None:
    client = get_client()
    ensure_collection(client)
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=get_embeddings())

    with open(DATA / "filings.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        name = f"{row['ticker']} {row['fiscal_year']}"
        docs, ids = make_chunks(row)
        stored = client.count(
            COLLECTION, count_filter=make_filter(row["ticker"], int(row["fiscal_year"])), exact=True).count
        if stored == len(docs):
            print(f"{name}: skipped, all {stored} chunks are already stored")
            continue
        # Embedding costs money, so we only do it for filings that are not complete.
        store.add_documents(docs, ids=ids, batch_size=200)
        print(f"{name}: stored {len(docs)} chunks (there were {stored})")

    total = client.count(COLLECTION, exact=True).count
    print(f"Collection '{COLLECTION}' now holds {total} chunks.")


if __name__ == "__main__":
    main()
