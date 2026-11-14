"""Stage 7 tests: /health, /ask, the error table, and the start-up check."""
import logging
import os
import uuid

import pytest
from fastapi.testclient import TestClient

import backend.api as api

client = TestClient(api.app)  # without "with": the start-up code is tested on its own below


def headers() -> dict:
    return {"X-Access-Code": os.environ["ACCESS_CODE"]}


def body(**kw) -> dict:
    return {"question": "What was Ford's R&D expense in 2025?", "chat_id": str(uuid.uuid4()),
            "pipeline": "agent", "companies": [], "years": [], **kw}


# ---- /health

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


# ---- the error table (none of these calls a paid service)

def test_missing_access_code():
    r = client.post("/ask", json=body())
    assert r.status_code == 401 and r.json() == {"error": "Wrong or missing access code."}


def test_wrong_access_code():
    r = client.post("/ask", json=body(), headers={"X-Access-Code": "not-the-code"})
    assert r.status_code == 401 and r.json() == {"error": "Wrong or missing access code."}


def test_access_code_is_checked_before_the_question():
    r = client.post("/ask", json=body(question=""), headers={"X-Access-Code": "not-the-code"})
    assert r.status_code == 401


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question(question):
    r = client.post("/ask", json=body(question=question), headers=headers())
    assert r.status_code == 400 and r.json() == {"error": "The question is empty."}


def test_question_longer_than_1000_characters():
    r = client.post("/ask", json=body(question="a" * 1001), headers=headers())
    assert r.status_code == 400
    assert r.json() == {"error": "The question is too long. Use at most 1,000 characters."}


def test_unknown_pipeline():
    r = client.post("/ask", json=body(pipeline="magic"), headers=headers())
    assert r.status_code == 400
    assert r.json() == {"error": "Unknown pipeline. Use simple, filters, rerank, decompose, or agent."}


def test_body_that_is_not_json_is_a_400_not_a_stack_trace():
    r = client.post("/ask", content="not json", headers=headers())
    assert r.status_code == 400 and "error" in r.json()


# ---- a service fails: 503, a fixed message, and no secret in the log

def test_service_failure_gives_503_and_logs_no_secret(monkeypatch, caplog):
    secret = "postgresql://user:hunter2pass@db.example.com/neon"
    monkeypatch.setenv("NEON_DATABASE_URL", secret)

    def boom(*args, **kwargs):
        raise ConnectionError(f"could not connect to {secret}")

    monkeypatch.setattr(api, "answer", boom)
    with caplog.at_level(logging.ERROR, logger="tenk.api"):
        r = client.post("/ask", json=body(), headers=headers())
    assert r.status_code == 503
    assert r.json() == {"error": "The answer service is not available right now. Try again in a minute."}
    assert "hunter2pass" not in r.text and "hunter2pass" not in caplog.text
    assert "ConnectionError" in caplog.text  # but the error itself is logged


# ---- the start-up check prints names, never values

def test_startup_names_the_missing_variable(monkeypatch):
    monkeypatch.delenv("QDRANT_API_KEY")
    with pytest.raises(RuntimeError, match="QDRANT_API_KEY"):
        api.check_environment()


def test_startup_rejects_a_line_break_and_prints_only_the_name(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secretvalue\nextra-line")
    with pytest.raises(RuntimeError) as e:
        api.check_environment()
    assert "OPENROUTER_API_KEY" in str(e.value) and "secretvalue" not in str(e.value)


def test_startup_checks_the_optional_langsmith_key_for_a_line_break(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-secretvalue\n")
    with pytest.raises(RuntimeError) as e:
        api.check_environment()
    assert "LANGSMITH_API_KEY" in str(e.value) and "secretvalue" not in str(e.value)


def test_server_starts_with_the_real_environment():
    with TestClient(api.app) as c:   # "with" runs the start-up and shut-down code
        assert c.get("/health").json() == {"status": "ok"}


# ---- one real question through the API (calls OpenRouter, costs a fraction of a cent)

def test_ask_returns_the_documented_response():
    chat_id = str(uuid.uuid4())
    r = client.post("/ask", json=body(chat_id=chat_id), headers=headers())
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"answer", "pipeline", "chat_id", "tools_used", "sources"}
    assert data["pipeline"] == "agent" and data["chat_id"] == chat_id
    assert "9.4" in data["answer"] and "sec_financial_data" in data["tools_used"]
    assert data["sources"][0]["type"] == "sec" and data["sources"][0]["company"] == "F"
