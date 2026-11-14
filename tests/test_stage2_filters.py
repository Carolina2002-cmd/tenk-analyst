"""Stage 2 tests: the filters pipeline and the sidebar check."""
import uuid

from backend.pipelines import answer


def ask(question: str, companies=None, years=None, pipeline="filters") -> dict:
    return answer(question, pipeline, companies or [], years or [], str(uuid.uuid4()))


def test_filters_searches_only_gm_2024():
    r = ask("What does GM say about cybersecurity in 2024?")
    assert len(r["sources"]) == 5
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in r["sources"])


def test_sidebar_without_the_named_company_returns_the_sidebar_message():
    r = ask("What was GM's R&D expense in 2025?", companies=["TSLA", "F"])
    assert "not selected in the sidebar: General Motors." in r["answer"]
    assert r["sources"] == [] and r["tools_used"] == []


def test_sidebar_without_the_named_year_returns_the_sidebar_message():
    r = ask("What does Ford say about cybersecurity in 2024?", years=[2025])
    assert "not selected in the sidebar: FY2024." in r["answer"]
    assert r["sources"] == []


def test_sidebar_list_replaces_the_companies_of_the_question():
    # The question names no company, so the sidebar decides. The year still comes from the question.
    r = ask("What does the company say about cybersecurity in 2025?", companies=["F"])
    assert r["sources"] and all(s["company"] == "F" and s["year"] == 2025 for s in r["sources"])


def test_simple_ignores_the_sidebar():
    # simple searches everything, and has no sidebar check, so it does not return the sidebar message.
    r = ask("What was GM's R&D expense in 2025?", companies=["TSLA", "F"], pipeline="simple")
    assert "sidebar" not in r["answer"]
    assert len(r["sources"]) == 5


def test_out_of_scope_still_wins_in_filters():
    r = ask("What risks does Toyota report?")
    assert r["answer"].startswith("This app covers only Tesla")
    assert r["sources"] == []
