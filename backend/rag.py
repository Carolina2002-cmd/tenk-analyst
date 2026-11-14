"""Shared RAG tools: the models and the Qdrant search. Later stages add the reranker here."""
import functools
import os

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import models

from backend.ingest import COLLECTION, OPENROUTER, get_client, get_embeddings

load_dotenv()


def get_llm() -> ChatOpenAI:
    """The chat model. temperature=0 means the same question gives (almost) the same answer."""
    return ChatOpenAI(
        model=os.environ.get("CHAT_MODEL", "anthropic/claude-haiku-5.5"),
        base_url=OPENROUTER,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        max_tokens=1500,
    )


@functools.lru_cache(maxsize=1)
def get_store() -> QdrantVectorStore:
    """Connect to Qdrant once and reuse the connection."""
    return QdrantVectorStore(client=get_client(), collection_name=COLLECTION, embedding=get_embeddings())


def make_filter(companies=None, years=None, items=None):
    """Build a Qdrant filter. LangChain stores metadata under 'metadata.', so keys start with that."""
    must = []
    if companies:
        must.append(models.FieldCondition(key="metadata.company", match=models.MatchAny(any=companies)))
    if years:
        must.append(models.FieldCondition(key="metadata.year", match=models.MatchAny(any=years)))
    if items:
        must.append(models.FieldCondition(key="metadata.item", match=models.MatchAny(any=items)))
    return models.Filter(must=must) if must else None


@functools.lru_cache(maxsize=1)
def get_ranker() -> Ranker:
    """Load the small FlashRank model once (about 3 MB, runs on the CPU)."""
    return Ranker(model_name="ms-marco-TinyBERT-L-2-v2")


# @traceable makes this call show up as its own step in LangSmith, with its input, output, and time.
@traceable(run_type="retriever", name="qdrant_search")
def search(question: str, k: int = 5, flt=None) -> list:
    """Return the k chunks whose meaning is closest to the question."""
    return get_store().similarity_search(question, k=k, filter=flt)


@traceable(name="flashrank_rerank")
def rerank(question: str, docs: list, top_n: int = 5) -> list:
    """Score each chunk against the question with FlashRank and keep the best top_n."""
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    ranked = get_ranker().rerank(RerankRequest(query=question, passages=passages))  # best first
    return [docs[r["id"]] for r in ranked[:top_n]]
