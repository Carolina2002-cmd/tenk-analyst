"""Stage 8: the Streamlit front end.

Start the back end first, then:  uv run python -m streamlit run frontend/app.py

Streamlit runs this whole file again after every click or key press.
Things that must survive between runs (the chat, the chat ID) live in st.session_state.
"""
import os
import secrets
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# On your computer the back end runs on port 8000. On Render, API_URL is the back end's public address.
BACKEND_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_SECONDS = 90
WAKING = "The server is waking up. This takes about 30 seconds."

COMPANIES = {
    "TSLA": "Tesla", "GM": "General Motors", "F": "Ford", "RIVN": "Rivian", "LCID": "Lucid",
    "APTV": "Aptiv", "BWA": "BorgWarner", "ALV": "Autoliv", "HOG": "Harley-Davidson", "CVNA": "Carvana",
}
YEARS = [2024, 2025]

# The pipelines, for the "Learning mode" section. The default is the agent.
PIPELINES = {
    "agent": "Agent: uses tools (SEC numbers and 10-K search) and remembers the chat",
    "decompose": "Decompose: splits the question into parts, searches each part",
    "rerank": "Rerank: searches 20 chunks, keeps the best 5",
    "filters": "Filters: searches only the companies and years of the question",
    "simple": "Simple: searches all chunks, no filters",
}

EXAMPLES = [
    "What risks does Tesla report about battery supply?",
    "Compare GM and Ford R&D in 2025",
    "How did Carvana's revenue change from 2024 to 2025?",
]

st.set_page_config(page_title="10-K Analyst", page_icon="📊")

# ---------------------------------------------------------------- state

st.session_state.setdefault("authed", False)
st.session_state.setdefault("chat_id", str(uuid.uuid4()))   # one random ID per chat
st.session_state.setdefault("messages", [])                 # the chat on screen


# ---------------------------------------------------------------- access code screen

def access_screen() -> None:
    """Ask for the access code. A form submits with Enter or with the button."""
    st.title("10-K Analyst")
    expected = os.environ.get("ACCESS_CODE", "")
    if not expected:
        st.error("ACCESS_CODE is not set. Fill in the ACCESS_CODE line of your .env file and restart.")
        st.stop()
    with st.form("access"):
        code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if secrets.compare_digest(code.encode(), expected.encode()):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong access code.")
    st.stop()  # nothing below this line is shown until the code is right


if not st.session_state.authed:
    access_screen()


# ---------------------------------------------------------------- talking to the back end

def ask_backend(payload: dict, notice) -> tuple[dict | None, str | None]:
    """Send one question. Returns (answer data, error text); one of them is None.

    A server on a free plan sleeps when idle. If it does not answer within 90 seconds,
    we say that it is waking up and try once more."""
    for attempt in (1, 2):
        try:
            r = httpx.post(f"{BACKEND_URL}/ask", json=payload, timeout=TIMEOUT_SECONDS,
                           headers={"X-Access-Code": os.environ["ACCESS_CODE"]})
        except httpx.TransportError:   # timeout, or the server is not reachable
            if attempt == 1:
                notice.info(WAKING)
                continue
            return None, "The server did not answer. Try again in a minute."
        try:
            data = r.json()
        except ValueError:
            return None, "The server sent an answer that could not be read."
        if r.status_code == 200:
            return data, None
        return None, data.get("error", "Something went wrong.")   # show the back end's own error text
    return None, "The server did not answer. Try again in a minute."


# ---------------------------------------------------------------- showing an answer

def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1e9:.1f} billion"


def show_source(s: dict) -> None:
    name = COMPANIES.get(s["company"], s["company"])
    if s["type"] == "sec":
        st.markdown(f"**{name} · FY{s['year']} · SEC data** — {s['metric']}: {money(s['value'])} "
                    f"(tag `{s['tag']}`)")
    else:
        st.markdown(f"**{name} · FY{s['year']} 10-K · Item {s['item']}** — {s['item_title']}  \n"
                    f"[Open the filing on sec.gov]({s['url']})")
        st.text(s["excerpt"])


def show_message(m: dict, show_sources: bool) -> None:
    st.markdown(m["content"])
    if m["role"] != "assistant":
        return
    if m.get("tools_used"):   # the tool labels are always shown
        st.markdown("Tools used: " + " ".join(f":blue-badge[{t}]" for t in m["tools_used"]))
    if show_sources and m.get("sources"):
        with st.expander(f"Sources ({len(m['sources'])})"):
            for s in m["sources"]:
                show_source(s)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Filters")
    companies = st.multiselect("Companies", list(COMPANIES), default=list(COMPANIES),
                               format_func=lambda t: f"{COMPANIES[t]} ({t})")
    years = st.multiselect("Fiscal years", YEARS, default=YEARS, format_func=lambda y: f"FY{y}")
    st.caption("Clearing a list is the same as selecting all.")
    show_sources = st.toggle("Show sources", value=True)
    if st.button("New chat"):
        st.session_state.chat_id = str(uuid.uuid4())   # a new chat ID means a fresh memory
        st.session_state.messages = []
        st.rerun()
    with st.expander("Learning mode"):
        pipeline = st.radio("Pipeline", list(PIPELINES), format_func=lambda p: PIPELINES[p], index=0)

# ---------------------------------------------------------------- the chat

st.title("10-K Analyst")
st.caption("Ask about the 10-K filings of Tesla, GM, Ford, Rivian, Lucid, Aptiv, BorgWarner, Autoliv, "
           "Harley-Davidson, and Carvana (fiscal years 2024 and 2025).")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        show_message(m, show_sources)

clicked = None
examples = st.empty()   # a placeholder: unlike a container, it can be cleared
if not st.session_state.messages:   # the examples are shown only at the start of a chat
    with examples.container():
        for example in EXAMPLES:
            if st.button(example):
                clicked = example

question = st.chat_input("Ask a question about the filings") or clicked
if question:
    examples.empty()   # hide the buttons at once, also in the run that answers the question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        notice = st.empty()   # a place for the "waking up" message
        with st.spinner("Searching the filings..."):
            data, error = ask_backend({
                "question": question,
                "chat_id": st.session_state.chat_id,
                "pipeline": pipeline,
                # An empty list means "not set". We send it when everything is selected.
                "companies": [] if len(companies) in (0, len(COMPANIES)) else companies,
                "years": [] if len(years) in (0, len(YEARS)) else years,
            }, notice)
        notice.empty()
        message = ({"role": "assistant", "content": error} if error else
                   {"role": "assistant", "content": data["answer"],
                    "tools_used": data["tools_used"], "sources": data["sources"]})
        show_message(message, show_sources)
    st.session_state.messages.append(message)
