"""Stage 6 tests: the agent remembers the conversation in Neon."""
import uuid

from backend.agent import get_checkpointer
from backend.pipelines import answer


def saved_messages(chat_id: str) -> list:
    """Read the saved conversation straight from Neon. Empty if nothing is saved under this chat ID."""
    saved = get_checkpointer().get_tuple({"configurable": {"thread_id": chat_id}})
    return saved.checkpoint["channel_values"].get("messages", []) if saved else []


def test_q8_follow_up_refers_to_the_previous_question():
    chat_id = str(uuid.uuid4())  # a new chat on every run, so old runs cannot help
    first = answer("What was Ford's revenue in 2024?", "agent", [], [], chat_id)
    assert "185.0" in first["answer"]
    second = answer("And in 2025?", "agent", [], [], chat_id)
    assert "187.3" in second["answer"]
    assert second["sources"] and all(s["company"] == "F" for s in second["sources"])


def test_conversation_is_saved_in_neon_under_the_chat_id_and_not_under_another():
    chat_id = str(uuid.uuid4())
    answer("What was Tesla's net income in 2025?", "agent", [], [], chat_id)
    texts = [str(m.content) for m in saved_messages(chat_id)]
    assert any("Tesla's net income" in t for t in texts)          # the question was saved
    assert any(type(m).__name__ == "ToolMessage" for m in saved_messages(chat_id))  # and the tool result
    assert saved_messages(str(uuid.uuid4())) == []               # another chat has nothing


def test_out_of_scope_questions_are_not_saved():
    chat_id = str(uuid.uuid4())
    answer("What risks does Toyota report?", "agent", [], [], chat_id)
    assert saved_messages(chat_id) == []
