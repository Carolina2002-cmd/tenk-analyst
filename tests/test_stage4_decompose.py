"""Stage 4 tests: decompose into sub-questions, search each in parallel, answer from all."""
import uuid

from backend.pipelines import MAX_SUB_QUESTIONS, answer, split_question


def ask(question: str, companies=None, years=None) -> dict:
    return answer(question, "decompose", companies or [], years or [], str(uuid.uuid4()))


def test_comparison_has_sources_from_both_companies():
    r = ask("Compare Tesla's and Ford's tariff risks in 2025.")
    found = {(s["company"], s["year"]) for s in r["sources"]}
    assert ("TSLA", 2025) in found and ("F", 2025) in found
    assert r["answer"].strip() and r["tools_used"] == []


def test_split_gives_one_sub_question_per_company_with_its_own_ticker():
    subs = split_question("Compare Tesla's and Ford's tariff risks in 2025.")
    assert 2 <= len(subs) <= MAX_SUB_QUESTIONS
    assert {t for s in subs for t in s.tickers} == {"TSLA", "F"}
    assert all(s.years == [2025] for s in subs)


def test_never_more_than_four_sub_questions():
    q = ("Compare the cybersecurity, tariff, battery, labor, and recall risks of Tesla, GM, Ford, Rivian, "
         "Lucid, Aptiv, BorgWarner, Autoliv, Harley-Davidson, and Carvana in 2024 and 2025.")
    assert len(split_question(q)) <= MAX_SUB_QUESTIONS


def test_sidebar_limits_the_search_even_for_a_comparison():
    # The question names both, and both are in the sidebar, so the search works on both.
    r = ask("Compare Tesla's and Ford's tariff risks in 2025.", companies=["TSLA", "F"], years=[2025])
    assert {s["company"] for s in r["sources"]} <= {"TSLA", "F"}


def test_sidebar_message_when_a_named_company_is_not_selected():
    r = ask("Compare Tesla's and Ford's tariff risks in 2025.", companies=["TSLA"])
    assert "not selected in the sidebar: Ford." in r["answer"]
    assert r["sources"] == []


def test_out_of_scope_company_in_decompose():
    r = ask("Compare Tesla and Toyota tariff risks.")
    assert r["answer"].startswith("This app covers only Tesla")
    assert r["sources"] == []
