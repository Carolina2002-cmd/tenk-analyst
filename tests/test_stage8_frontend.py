"""Stage 8 tests: the Streamlit page, run headless with Streamlit's AppTest and a fake back end."""
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")
CODE = "a-test-code"   # a value made up for these tests; it is not a real access code

ANSWER = {
    "answer": "Ford's R&D expense in FY2025 was $9.4 billion (Ford, FY2025, SEC data).",
    "pipeline": "agent", "chat_id": "x", "tools_used": ["sec_financial_data", "search_10k"],
    "sources": [
        {"type": "sec", "company": "F", "year": 2025, "metric": "rd_expense",
         "tag": "ResearchAndDevelopmentExpense", "value": 9400000000},
        {"type": "10-K", "company": "F", "year": 2025, "item": "7", "item_title": "MD&A",
         "excerpt": "Ford spent on research.", "url": "https://www.sec.gov/Archives/x"},
    ],
}


class FakeResponse:
    def __init__(self, status: int, data: dict):
        self.status_code, self._data = status, data

    def json(self) -> dict:
        return self._data


@pytest.fixture
def backend(monkeypatch):
    """Replace httpx.post. 'calls' records every request; 'replies' holds what to return, in order."""
    monkeypatch.setenv("ACCESS_CODE", CODE)
    calls, replies = [], [FakeResponse(200, ANSWER)]

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls, replies


def open_app(signed_in: bool = True) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=15)
    if signed_in:
        at.session_state["authed"] = True
    return at.run()


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def all_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " " + " ".join(e.value for e in at.error) \
        + " " + " ".join(i.value for i in at.info)


# ---- 1. the access code screen

def test_wrong_code_keeps_the_page_locked(backend):
    at = open_app(signed_in=False)
    assert not at.chat_input                       # nothing but the form is shown
    at.text_input[0].input("wrong").run()
    button(at, "Enter").click().run()
    assert at.error[0].value == "Wrong access code." and not at.chat_input


def test_right_code_unlocks_the_page_and_stays_unlocked(backend):
    at = open_app(signed_in=False)
    at.text_input[0].input(CODE)                   # typed text in a form only counts when the form is submitted
    button(at, "Enter").click().run()
    assert at.chat_input and at.session_state["authed"] is True
    at.run()                                       # another run in the same session: no second prompt
    assert at.chat_input


# ---- 2 and 3. the sidebar

def test_sidebar_has_filters_sources_toggle_new_chat_and_a_folded_learning_mode(backend):
    at = open_app()
    assert [m.label for m in at.sidebar.multiselect] == ["Companies", "Fiscal years"]
    assert len(at.sidebar.multiselect[0].value) == 10 and at.sidebar.multiselect[1].value == [2024, 2025]
    assert at.sidebar.toggle[0].label == "Show sources" and at.sidebar.toggle[0].value is True
    assert any(b.label == "New chat" for b in at.sidebar.button)
    assert at.sidebar.expander[0].label == "Learning mode"
    assert at.sidebar.radio[0].value == "agent"    # the agent is the default


# ---- 4. the examples

def test_three_example_buttons_send_the_exact_question(backend):
    calls, _ = backend
    at = open_app()
    labels = [b.label for b in at.button if b.label != "New chat"]
    assert labels == ["What risks does Tesla report about battery supply?",
                      "Compare GM and Ford R&D in 2025",
                      "How did Carvana's revenue change from 2024 to 2025?"]
    button(at, "Compare GM and Ford R&D in 2025").click().run()
    assert calls[0]["json"]["question"] == "Compare GM and Ford R&D in 2025"


# ---- 5. what is sent to the back end

def test_everything_selected_sends_empty_lists_and_the_access_code(backend):
    calls, _ = backend
    at = open_app()
    at.chat_input[0].set_value("What was Ford's R&D expense in 2025?").run()
    sent = calls[0]
    assert sent["json"]["companies"] == [] and sent["json"]["years"] == []
    assert sent["json"]["pipeline"] == "agent" and sent["json"]["chat_id"] == at.session_state["chat_id"]
    assert sent["headers"] == {"X-Access-Code": CODE}
    assert sent["timeout"] == 90 and sent["url"].endswith("/ask")


def test_selected_companies_and_years_and_the_learning_mode_pipeline_are_sent(backend):
    calls, _ = backend
    at = open_app()
    at.sidebar.multiselect[0].set_value(["TSLA", "F"])
    at.sidebar.multiselect[1].set_value([2025])
    at.sidebar.radio[0].set_value("simple")
    at.run()
    at.chat_input[0].set_value("hello").run()
    assert calls[0]["json"]["companies"] == ["TSLA", "F"]
    assert calls[0]["json"]["years"] == [2025]
    assert calls[0]["json"]["pipeline"] == "simple"


def test_the_back_end_address_comes_from_api_url_and_a_trailing_slash_is_harmless(backend, monkeypatch):
    calls, _ = backend
    monkeypatch.setenv("API_URL", "https://tenk-analyst-api.onrender.com/")   # as pasted into Render
    at = open_app()
    at.chat_input[0].set_value("hello").run()
    assert calls[0]["url"] == "https://tenk-analyst-api.onrender.com/ask"


# ---- 6. under each answer: tool labels (always) and sources (when switched on)

def test_answer_shows_tool_labels_and_a_source_list(backend):
    at = open_app()
    at.chat_input[0].set_value("What was Ford's R&D expense in 2025?").run()
    text = all_text(at)
    assert "$9.4 billion" in text
    assert ":blue-badge[sec_financial_data]" in text and ":blue-badge[search_10k]" in text
    assert [e.label for e in at.expander if e.label.startswith("Sources")] == ["Sources (2)"]
    assert "Ford · FY2025 · SEC data" in text and "Ford · FY2025 10-K · Item 7" in text
    assert "$9.4 billion" in text and "research.ac" not in text


def test_with_show_sources_off_the_labels_stay_and_the_sources_go(backend):
    at = open_app()
    at.sidebar.toggle[0].set_value(False).run()
    at.chat_input[0].set_value("What was Ford's R&D expense in 2025?").run()
    assert ":blue-badge[sec_financial_data]" in all_text(at)
    assert not [e for e in at.expander if e.label.startswith("Sources")]


def test_history_of_the_chat_stays_on_screen(backend):
    at = open_app()
    at.chat_input[0].set_value("first question").run()
    at.chat_input[0].set_value("second question").run()
    texts = [m.value for m in at.chat_message[0].markdown] + [m.value for m in at.chat_message[2].markdown]
    assert texts == ["first question", "second question"]
    assert len(at.chat_message) == 4                 # two questions and two answers


def test_new_chat_clears_the_screen_and_makes_a_new_chat_id(backend):
    at = open_app()
    at.chat_input[0].set_value("hello").run()
    old_id = at.session_state["chat_id"]
    next(b for b in at.sidebar.button if b.label == "New chat").click().run()
    assert at.session_state["chat_id"] != old_id and not at.chat_message


# ---- 7. errors and the sleeping server

def test_the_error_text_of_the_back_end_is_shown(backend):
    _, replies = backend
    replies[0] = FakeResponse(400, {"error": "The question is too long. Use at most 1,000 characters."})
    at = open_app()
    at.chat_input[0].set_value("x").run()
    assert "The question is too long. Use at most 1,000 characters." in all_text(at)


def test_a_timeout_is_retried_once_and_then_the_answer_is_shown(backend):
    calls, replies = backend
    replies[:] = [httpx.ReadTimeout("slow"), FakeResponse(200, ANSWER)]
    at = open_app()
    at.chat_input[0].set_value("What was Ford's R&D expense in 2025?").run()
    assert len(calls) == 2 and "$9.4 billion" in all_text(at)


def test_two_timeouts_give_a_plain_message_and_no_crash(backend):
    calls, replies = backend
    replies[:] = [httpx.ReadTimeout("slow")]
    at = open_app()
    at.chat_input[0].set_value("hello").run()
    assert len(calls) == 2 and not at.exception
    assert "The server did not answer. Try again in a minute." in all_text(at)
