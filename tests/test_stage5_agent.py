"""Stage 5 tests: the agent and its two tools."""
import uuid

from langgraph.errors import GraphRecursionError

import backend.agent as agent_module
from backend.agent import MSG_NO_NUMBER, MSG_STEPS, make_tools
from backend.pipelines import MSG_COMPANY, answer


def ask(question: str, companies=None, years=None) -> dict:
    return answer(question, "agent", companies or [], years or [], str(uuid.uuid4()))


# ---- the tools, without the LLM (free and fast)

def test_sec_tool_returns_ford_revenue_2024():
    tools, sources, _ = make_tools([], [])
    text = tools[1].invoke({"ticker": "F", "metric": "revenue", "years": [2024]})
    assert "$185.0 billion (FY2024)" in text
    # SEC's exact value is 184,992,000,000, which rounds to 185.0 billion.
    assert round(sources[0]["value"] / 1e9, 1) == 185.0 and sources[0]["type"] == "sec"


def test_sec_tool_does_not_return_gm_rd_2025():
    tools, sources, _ = make_tools([], [])
    text = tools[1].invoke({"ticker": "GM", "metric": "rd_expense", "years": [2025]})
    assert "not in the SEC financial data" in text
    assert sources == []


def test_sec_tool_rejects_unsupported_input_and_lists_what_is_supported():
    tools, _, _ = make_tools([], [])
    for args in [{"ticker": "TM", "metric": "revenue", "years": [2025]},
                 {"ticker": "F", "metric": "profit", "years": [2025]},
                 {"ticker": "F", "metric": "revenue", "years": [2023]}]:
        text = tools[1].invoke(args)
        assert "Not supported" in text and "TSLA" in text and "rd_expense" in text


def test_sec_tool_respects_the_sidebar():
    tools, sources, _ = make_tools(["TSLA"], [2025])
    assert "sidebar does not include Ford" in tools[1].invoke({"ticker": "F", "metric": "revenue", "years": [2025]})
    assert "sidebar does not include these years" in tools[1].invoke(
        {"ticker": "TSLA", "metric": "revenue", "years": [2024]})
    assert sources == []


def test_search_tool_cannot_widen_the_sidebar():
    tools, sources, _ = make_tools(["TSLA"], [2025])
    text = tools[0].invoke({"query": "tariffs", "companies": ["F"], "years": [2025]})
    assert text.startswith("Nothing searched")
    assert sources == []
    # Asking for Ford and Tesla: only Tesla is searched, and the agent is told.
    text = tools[0].invoke({"query": "tariffs", "companies": ["F", "TSLA"], "years": [2025]})
    assert "does not include these companies: F" in text
    assert sources and all(s["company"] == "TSLA" and s["year"] == 2025 for s in sources)


def test_step_limit_gives_the_fixed_message(monkeypatch):
    class Looping:
        def invoke(self, *a, **k):
            raise GraphRecursionError("limit")
    monkeypatch.setattr(agent_module, "create_agent", lambda *a, **k: Looping())
    r = ask("What was Ford's R&D expense in 2025?")
    assert r["answer"] == MSG_STEPS


# ---- the agent with the LLM (test questions 4, 5, 6, 7, 10)

def test_q4_ford_rd_2025_uses_sec_data():
    r = ask("What was Ford's R&D expense in 2025?")
    assert "9.4" in r["answer"]
    assert "sec_financial_data" in r["tools_used"]
    assert {"type": "sec", "company": "F", "year": 2025} == {k: r["sources"][0][k] for k in ("type", "company", "year")}


def test_q5_gm_rd_2025_comes_from_the_10k_text():
    r = ask("What was GM's R&D expense in 2025?")
    assert "8.5" in r["answer"]
    assert any(s["type"] == "10-K" and s["company"] == "GM" and s["year"] == 2025 for s in r["sources"])


def test_q6_carvana_rd_is_not_reported():
    r = ask("What was Carvana's R&D expense in 2025?")
    assert MSG_NO_NUMBER in r["answer"]
    assert "sec_financial_data" in r["tools_used"]


def test_q7_toyota_is_out_of_scope_with_no_tools():
    r = ask("What risks does Toyota report?")
    assert r["answer"] == MSG_COMPANY
    assert r["sources"] == [] and r["tools_used"] == []


def test_q10_sidebar_conflict_calls_no_tools():
    r = ask("What was GM's R&D expense in 2025?", companies=["TSLA", "F"])
    assert r["answer"].startswith("This question asks about something that is not selected in the sidebar:")
    assert "General Motors" in r["answer"]
    assert r["sources"] == [] and r["tools_used"] == []
