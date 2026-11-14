"""Stage 3 tests: search 20 chunks, rerank with FlashRank, keep 5."""
import uuid

from langchain_core.documents import Document

from backend.pipelines import answer
from backend.rag import rerank, search, make_filter


def ask(question: str, companies=None, years=None) -> dict:
    return answer(question, "rerank", companies or [], years or [], str(uuid.uuid4()))


def test_rerank_keeps_five_sources_from_the_filtered_company_and_year():
    r = ask("What does GM say about cybersecurity in 2024?")
    assert len(r["sources"]) == 5
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in r["sources"])


def test_rerank_keeps_the_most_relevant_chunk_first_and_drops_the_rest():
    docs = [Document(page_content=t, metadata={}) for t in [
        "The cafeteria menu changes every Tuesday.",
        "Our lithium-ion battery cell supply depends on a small number of suppliers.",
        "The company was incorporated in Delaware.",
    ]]
    kept = rerank("What are the battery supply risks?", docs, top_n=2)
    assert len(kept) == 2
    assert "battery" in kept[0].page_content


def test_search_returns_20_when_asked():
    assert len(search("tariff risks", k=20, flt=make_filter(["F"], [2025]))) == 20


def test_sidebar_message_still_applies_in_rerank():
    r = ask("What was GM's R&D expense in 2025?", companies=["TSLA", "F"])
    assert "not selected in the sidebar: General Motors." in r["answer"]
    assert r["sources"] == []
